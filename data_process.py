import base64
import boto3
from io import BytesIO
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.utils.dates import days_ago
import subprocess
import sys
import importlib.util

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

def ensure_packages(packages):
    """Install Python packages dynamically if not already installed."""
    for pkg in packages:
        if importlib.util.find_spec(pkg) is None:
            print(f"Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

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
    dag_id='process_audio',
    default_args=default_args,
    schedule_interval=None,
    tags=['audio', 'processing'],
    access_control={'Admin': {'can_read', 'can_edit', 'can_delete'}},
    params={
        's3_endpoint': Param("local-s3-service.ezdata-system.svc.cluster.local:30000", type="string"),
        's3_endpoint_ssl_enabled': Param(False, type="boolean"),
        's3_bucket_raw': Param("audio-raw", type="string"),
        's3_bucket_processed': Param("audio-processed", type="string"),
        's3_files_prefix_raw': Param("", type="string"),
    }
) as dag:

    @task
    def list_raw_wav_files():
        """Return list of WAV files in raw bucket"""
        context = get_current_context()
        bucket = context['params']['s3_bucket_raw']
        prefix = context['params']['s3_files_prefix_raw'] or ""
        s3 = get_s3_client(context['params']['s3_endpoint'], context['params']['s3_endpoint_ssl_enabled'])
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if "Contents" in resp:
            wav_files = [obj["Key"] for obj in resp["Contents"] if obj["Key"].lower().endswith(".wav")]
            if wav_files:
                print(f"Found {len(wav_files)} WAV file(s) in '{bucket}': {wav_files}")
                return wav_files
        print(f"No WAV files found in bucket '{bucket}' with prefix '{prefix}'.")
        return []

    @task
    def process_audio_file(input_key: str):
        """Process a single WAV file: denoise, normalize, preemphasis, upload to processed bucket"""
        # Install required packages dynamically
        ensure_packages(["librosa", "noisereduce", "soundfile", "numpy", "boto3"])

        import librosa, noisereduce as nr, soundfile as sf
        import numpy as np
        from io import BytesIO

        # S3 client
        s3 = get_s3_client("local-s3-service.ezdata-system.svc.cluster.local:30000", ssl_enabled=False)

        input_bucket = "audio-raw"
        output_bucket = "audio-processed"
        output_key = input_key.replace(".wav", "_processed.wav")

        # Skip if already processed
        try:
            s3.head_object(Bucket=output_bucket, Key=output_key)
            print(f"No new files to process. Skipping '{input_key}' as it already exists in processed bucket.")
            return
        except s3.exceptions.ClientError:
            pass  # file does not exist, proceed

        # Download, process, and upload
        audio_bytes = BytesIO(s3.get_object(Bucket=input_bucket, Key=input_key)['Body'].read())
        audio, sr = librosa.load(audio_bytes, sr=None)
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
        audio_norm = librosa.util.normalize(audio_denoised)
        audio_final = librosa.effects.preemphasis(audio_norm, coef=0.95)

        buf = BytesIO()
        sf.write(buf, audio_final, sr, format='WAV')
        buf.seek(0)
        s3.put_object(Bucket=output_bucket, Key=output_key, Body=buf.getvalue())
        print(f"Processed and uploaded '{output_key}' successfully!")

    # -----------------------------
    # DAG Flow
    # -----------------------------
    raw_files = list_raw_wav_files()
    processed_tasks = process_audio_file.expand(input_key=raw_files)
    raw_files >> processed_tasks
