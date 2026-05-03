import base64
import boto3
import logging
import subprocess
from io import BytesIO

import pendulum
from airflow import DAG
from airflow.decorators import task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_BUCKET = "audio-raw"
OUTPUT_BUCKET = "audio-processed"
S3_ENDPOINT = "http://local-s3-service.ezdata-system.svc.cluster.local:30000"
MIN_FILE_SIZE_MB = 0.1
MAX_FILE_SIZE_MB = 500
MIN_PROCESSED_SIZE_BYTES = 1000
PROCESSING_VERSION = "2.0"

REQUIRED_PACKAGES = [
    "torch==2.2.2",
    "torchaudio==2.2.2",
    "librosa==0.10.1",
    "noisereduce==3.0.0",
    "soundfile==0.12.1",
    "numpy==1.26.4",        # numpy 1.24.x doesn't support Python 3.12
]



def get_token():
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            namespace = f.read().strip()
        from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
        k8s_hook = KubernetesHook()
        secret = k8s_hook.core_v1_client.read_namespaced_secret("access-token", namespace)
        return base64.b64decode(secret.data["AUTH_TOKEN"]).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to get token: {e}")
        raise


def get_s3_client():
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=get_token(),
            aws_secret_access_key="s3",
            endpoint_url=S3_ENDPOINT,
            use_ssl=False,
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        raise


def validate_audio(audio_data, sample_rate, min_duration=1.0, max_duration=3600.0):
    duration = len(audio_data) / sample_rate
    if not (min_duration <= duration <= max_duration):
        raise ValueError(f"Audio duration {duration:.2f}s out of range [{min_duration}, {max_duration}]")
    if sample_rate < 8000:
        raise ValueError(f"Sample rate too low: {sample_rate}Hz (min 8000Hz)")
    if audio_data.ndim > 1 and audio_data.shape[1] > 2:
        raise ValueError(f"Too many channels: {audio_data.shape[1]} (max 2)")
    logger.info(f"Audio validated: {duration:.2f}s at {sample_rate}Hz")


def install_packages():
    logger.info("Installing required packages...")
    result = subprocess.run(
        ["pip", "install", "--no-cache-dir"] + REQUIRED_PACKAGES,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Package installation failed:\n{result.stderr}")
    logger.info("Packages installed successfully.")


with DAG(
    dag_id="masterDAG",
    schedule="0 */12 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    tags=["audio", "processing"],
    catchup=False,
    access_control={"Admin": {"DAGs": {"can_read", "can_edit", "can_delete"}}},
) as dag:

    @task
    def list_raw_files():
        s3 = get_s3_client()
        resp = s3.list_objects_v2(Bucket=INPUT_BUCKET)

        if "Contents" not in resp:
            logger.warning("No files found in raw bucket.")
            return []

        files = []
        for obj in resp["Contents"]:
            key = obj["Key"]
            if not key.lower().endswith(".wav"):
                continue
            size_mb = obj["Size"] / (1024 * 1024)
            if MIN_FILE_SIZE_MB <= size_mb <= MAX_FILE_SIZE_MB:
                files.append({
                    "key": key,
                    "size_mb": round(size_mb, 2),
                    "last_modified": obj["LastModified"].isoformat(),
                })
                logger.info(f"Queued: {key} ({size_mb:.2f}MB)")
            else:
                logger.warning(f"Skipping {key}: size {size_mb:.2f}MB out of range.")

        logger.info(f"Total files queued: {len(files)}")
        return files

    @task
    def process_all_files(file_list):
        if not file_list:
            logger.warning("No files to process.")
            return []

        try:
            install_packages()
        except Exception as e:
            logger.error(str(e))
            return [{"status": "failed", "error": str(e), "file_key": f.get("key")} for f in file_list]

        import librosa
        import noisereduce as nr
        import numpy as np
        import soundfile as sf

        s3 = get_s3_client()
        results = []

        for file_info in file_list:
            file_key = file_info["key"]
            output_key = file_key.replace(".wav", "_processed.wav")

            try:
                existing = s3.head_object(Bucket=OUTPUT_BUCKET, Key=output_key)
                if existing["ContentLength"] > MIN_PROCESSED_SIZE_BYTES:
                    logger.info(f"Already processed, skipping: {file_key}")
                    results.append({"status": "skipped", "file_key": file_key, "output_key": output_key})
                    continue
            except s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    logger.error(f"S3 head error for {file_key}: {e}")
                    results.append({"status": "failed", "error": str(e), "file_key": file_key})
                    continue

            try:
                logger.info(f"Downloading: {file_key}")
                obj = s3.get_object(Bucket=INPUT_BUCKET, Key=file_key)
                audio_bytes = BytesIO(obj["Body"].read())
                original_size = len(audio_bytes.getvalue())

                audio, sr = librosa.load(audio_bytes, sr=None)
                validate_audio(audio, sr)

                audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
                if not np.isfinite(audio).all():
                    raise ValueError("Noise reduction produced NaN/Inf values.")

                audio = librosa.util.normalize(audio)
                audio = librosa.effects.preemphasis(audio, coef=0.95)
                validate_audio(audio, sr)

                buf = BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
                buf.seek(0)
                processed_data = buf.getvalue()
                processed_size = len(processed_data)

                if processed_size < MIN_PROCESSED_SIZE_BYTES:
                    raise ValueError(f"Processed file too small: {processed_size} bytes")

                s3.put_object(
                    Bucket=OUTPUT_BUCKET,
                    Key=output_key,
                    Body=processed_data,
                    Metadata={
                        "original_file": file_key,
                        "original_size": str(original_size),
                        "processed_size": str(processed_size),
                        "sample_rate": str(sr),
                        "processing_version": PROCESSING_VERSION,
                    },
                )

                logger.info(f"✓ Done: {output_key} ({processed_size} bytes)")
                results.append({
                    "status": "success",
                    "file_key": file_key,
                    "output_key": output_key,
                    "original_size": original_size,
                    "processed_size": processed_size,
                    "sample_rate": sr,
                })

            except Exception as e:
                logger.error(f"✗ Failed: {file_key} — {e}")
                try:
                    s3.delete_object(Bucket=OUTPUT_BUCKET, Key=output_key)
                except Exception:
                    pass
                results.append({"status": "failed", "file_key": file_key, "error": str(e)})

        return results

    @task
    def summarize_results(results):
        total = len(results)
        successful = sum(1 for r in results if r.get("status") == "success")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        failed = sum(1 for r in results if r.get("status") == "failed")

        logger.info("===== Processing Summary =====")
        logger.info(f"  Total:      {total}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped:    {skipped}")
        logger.info(f"  Failed:     {failed}")

        for r in results:
            if r.get("status") == "failed":
                logger.error(f"  FAILED: {r.get('file_key')} — {r.get('error')}")

        if successful:
            orig_total = sum(r.get("original_size", 0) for r in results if r.get("status") == "success")
            proc_total = sum(r.get("processed_size", 0) for r in results if r.get("status") == "success")
            if orig_total:
                logger.info(f"  Compression ratio: {proc_total / orig_total:.2f}")

        return {"total": total, "successful": successful, "skipped": skipped, "failed": failed}

    raw_files = list_raw_files()
    process_results = process_all_files(raw_files)
    summarize_results(process_results)
