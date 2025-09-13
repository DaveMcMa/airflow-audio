import base64
import boto3
import botocore.exceptions
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context
from io import BytesIO
import librosa
import noisereduce as nr
import soundfile as sf

# -----------------------------
# Helper Functions
# -----------------------------
def get_token():
    """Fetch auth token from K8s secret."""
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


def preprocess_audio_for_transcription_s3(s3_client, input_bucket: str, input_key: str, 
                                           output_bucket: str, output_key: str):
    """
    Read a WAV file from S3, preprocess (denoise, normalize, preemphasis),
    and write the processed audio back to S3. Skips if output exists.
    """
    # Check if output already exists
    try:
        s3_client.head_object(Bucket=output_bucket, Key=output_key)
        print(f"Skipping '{input_key}' as it already exists in '{output_bucket}'.")
        return
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            raise  # unexpected error

    print(f"Processing '{input_key}'...")
    audio_obj = s3_client.get_object(Bucket=input_bucket, Key=input_key)
    audio_bytes = BytesIO(audio_obj['Body'].read())

    audio, sample_rate = librosa.load(audio_bytes, sr=None)
    audio_denoised = nr.reduce_noise(y=audio, sr=sample_rate, stationary=False, prop_decrease=0.8)
    audio_normalized = librosa.util.normalize(audio_denoised)
    audio_final = librosa.effects.preemphasis(audio_normalized, coef=0.95)

    output_buffer = BytesIO()
    sf.write(output_buffer, audio_final, sample_rate, format='WAV')
    output_buffer.seek(0)
    s3_client.put_object(Bucket=output_bucket, Key=output_key, Body=output_buffer.getvalue())
    print(f"Uploaded processed file to s3://{output_bucket}/{output_key}")


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
    params={
        's3_endpoint': Param("minio-service.ezdata-system.svc.cluster.local:30000", type="string"),
        's3_endpoint_ssl_enabled': Param(False, type="boolean"),
        's3_bucket_raw': Param("audio-raw", type="string"),
        's3_bucket_processed': Param("audio-processed", type="string"),
        's3_files_prefix_raw': Param("", type="string"),
    }
) as dag:

    @task
    def list_raw_wav_files():
        """List all WAV files in the raw audio bucket."""
        context = get_current_context()
        bucket_name = context['params']['s3_bucket_raw']
        prefix = context['params']['s3_files_prefix_raw']
        s3 = get_s3_client(context['params']['s3_endpoint'], context['params']['s3_endpoint_ssl_enabled'])
        try:
            resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if "Contents" in resp:
                wav_keys = [obj["Key"] for obj in resp["Contents"] if obj["Key"].lower().endswith(".wav")]
                print(f"Found WAV files: {wav_keys}")
                return wav_keys
            print("No WAV files found.")
            return []
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Error listing S3 objects: {str(e)}")

    @task
    def process_wav_file(key: str):
        """Process a single WAV file."""
        context = get_current_context()
        s3 = get_s3_client(context['params']['s3_endpoint'], context['params']['s3_endpoint_ssl_enabled'])
        input_bucket = context['params']['s3_bucket_raw']
        output_bucket = context['params']['s3_bucket_processed']
        output_key = key.replace(".wav", "_processed.wav")
        preprocess_audio_for_transcription_s3(s3, input_bucket, key, output_bucket, output_key)

    # DAG flow
    wav_files = list_raw_wav_files()
    process_wav_file.expand(key=wav_files)
