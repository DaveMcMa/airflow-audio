import base64
import boto3
from io import BytesIO
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context
import subprocess

# -----------------------------
# Helper Functions
# -----------------------------
def get_token():
    """Fetch JWT token from Kubernetes secret."""
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        namespace = f.read()
    from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
    k8s_hook = KubernetesHook()
    secret = k8s_hook.core_v1_client.read_namespaced_secret("access-token", namespace)
    token_encoded = secret.data["AUTH_TOKEN"]  # type: ignore
    return base64.b64decode(token_encoded).decode("utf-8")


def get_s3_client():
    """Return boto3 S3 client configured with JWT auth."""
    endpoint_url = "http://local-s3-service.ezdata-system.svc.cluster.local:30000"
    jwt_token = get_token()
    return boto3.client(
        "s3",
        aws_access_key_id=jwt_token,
        aws_secret_access_key="s3",
        endpoint_url=endpoint_url,
        use_ssl=False,
    )

# -----------------------------
# DAG Definition
# -----------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
}

with DAG(
    dag_id='process_audio_all_files',
    default_args=default_args,
    schedule_interval=None,
    tags=['audio', 'processing'],
    access_control={'Admin': {'can_read', 'can_edit', 'can_delete'}},
) as dag:

    @task
    def list_raw_files():
        """List all WAV files in the raw bucket"""
        s3 = get_s3_client()
        bucket = "audio-raw"
        resp = s3.list_objects_v2(Bucket=bucket)
        files = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].lower().endswith(".wav")]
        print(f"Found {len(files)} WAV files: {files}")
        return files

    @task
    def process_file(file_key: str):
        """Process a single WAV file, installing packages if needed"""
        # Install packages at runtime with full pip logging
        result = subprocess.run(
            ["pip", "install", "librosa", "noisereduce", "soundfile", "numpy"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(result.stderr)

        import librosa
        import noisereduce as nr
        import soundfile as sf

        s3 = get_s3_client()
        input_bucket = "audio-raw"
        output_bucket = "audio-processed"
        output_key = file_key.replace(".wav", "_processed.wav")

        # Skip if already processed
        try:
            s3.head_object(Bucket=output_bucket, Key=output_key)
            print(f"Skipping {file_key}, already processed.")
            return
        except s3.exceptions.ClientError:
            pass

        # Download
        audio_bytes = BytesIO(s3.get_object(Bucket=input_bucket, Key=file_key)['Body'].read())
        audio, sr = librosa.load(audio_bytes, sr=None)
        # Process
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
        audio_norm = librosa.util.normalize(audio_denoised)
        audio_final = librosa.effects.preemphasis(audio_norm, coef=0.95)
        # Upload
        buf = BytesIO()
        sf.write(buf, audio_final, sr, format='WAV')
        buf.seek(0)
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=buf.getvalue())
        print(f"Processed and uploaded: {output_key}")

    # -----------------------------
    # DAG Flow
    # -----------------------------
    raw_files = list_raw_files()
    process_file.expand(file_key=raw_files)
