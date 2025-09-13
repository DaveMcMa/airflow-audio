import base64
import boto3
import botocore.exceptions
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context

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
    dag_id='test_audio_dag',
    default_args=default_args,
    schedule_interval=None,
    tags=['s3', 'audio', 'test'],
    access_control={'Admin': {'can_read', 'can_edit', 'can_delete'}},  # ✅ valid permissions
    params={
        's3_endpoint': Param("minio-service.ezdata-system.svc.cluster.local:30000", type="string"),
        's3_endpoint_ssl_enabled': Param(False, type="boolean"),
        's3_bucket_raw': Param("audio-raw", type="string"),
        's3_bucket_processed': Param("audio-processed", type="string"),
        's3_files_prefix_raw': Param("", type="string"),
        's3_files_prefix_processed': Param("", type="string"),
    }
) as dag:

    @task
    def list_s3_objects(bucket_param: str, prefix_param: str):
        context = get_current_context()
        bucket_name = context['params'][bucket_param]
        prefix = context['params'][prefix_param]
        s3 = get_s3_client(context['params']['s3_endpoint'], context['params']['s3_endpoint_ssl_enabled'])
        try:
            resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if "Contents" in resp:
                keys = [obj["Key"] for obj in resp["Contents"]]
                print(f"Objects in bucket '{bucket_name}' with prefix '{prefix}':")
                for k in keys:
                    print(f" - {k}")
                return keys
            print(f"No objects found in bucket '{bucket_name}' with prefix '{prefix}'.")
            return []
        except botocore.exceptions.ClientError as e:
            raise RuntimeError(f"Error listing S3 objects: {str(e)}")

    # Tasks
    list_raw_files = list_s3_objects("s3_bucket_raw", "s3_files_prefix_raw")
    list_processed_files = list_s3_objects("s3_bucket_processed", "s3_files_prefix_processed")
