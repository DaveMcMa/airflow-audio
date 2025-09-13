import base64
import boto3
from io import BytesIO
import subprocess
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context

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

def get_s3_client(endpoint_host: str, ssl_enabled: bool):
    """Return boto3 S3 client configured with JWT auth."""
    endpoint_url = f"http{'s' if ssl_enabled else ''}://{endpoint_host}"
    jwt_token = get_token()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=jwt_token,
        aws_secret_access_key="s3",
        endpoint_url=endpoint_url,
        use_ssl=ssl_enabled,
    )
    return s3

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
    dag_id='process_single_audio_file',
    default_args=default_args,
    schedule_interval=None,
    tags=['audio', 'processing'],
    access_control={'Admin': {'can_read', 'can_edit', 'can_delete'}},  # ✅ valid permissions
) as dag:

    @task
    def process_audio_file(file_key: str):
        """Process a single audio file from S3."""
        # Install missing packages at runtime
        for pkg in ["librosa", "noisereduce", "soundfile", "numpy"]:
            try:
                __import__(pkg)
            except ImportError:
                print(f"Installing missing package: {pkg}")
                subprocess.check_call(["python3", "-m", "pip", "install", pkg])

        import librosa
        import noisereduce as nr
        import soundfile as sf

        context = get_current_context()
        s3_endpoint = "local-s3-service.ezdata-system.svc.cluster.local:30000"
        s3_ssl = False
        input_bucket = "audio-raw"
        output_bucket = "audio-processed"

        s3 = get_s3_client(s3_endpoint, s3_ssl)
        output_key = file_key.replace(".wav", "_processed.wav")

        # Skip if already processed
        try:
            s3.head_object(Bucket=output_bucket, Key=output_key)
            print(f"Skipping {file_key}, already processed.")
            return
        except:
            pass

        print(f"Downloading {file_key} from {input_bucket}")
        audio_bytes = BytesIO(s3.get_object(Bucket=input_bucket, Key=file_key)['Body'].read())
        audio, sr = librosa.load(audio_bytes, sr=None)

        print(f"Processing {file_key}")
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
        audio_norm = librosa.util.normalize(audio_denoised)
        audio_final = librosa.effects.preemphasis(audio_norm, coef=0.95)

        buf = BytesIO()
        sf.write(buf, audio_final, sr, format='WAV')
        buf.seek(0)
        print(f"Uploading processed file to {output_bucket}/{output_key}")
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=buf.getvalue())
        print(f"Finished processing {file_key}")

    # Example: replace "test1.wav" with the actual key you want to process
    process_audio_file("test1.wav")
