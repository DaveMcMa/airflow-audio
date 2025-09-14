import base64
import boto3
import logging
import hashlib
from io import BytesIO
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Helper Functions
# -----------------------------
def get_token():
    """Fetch JWT token from Kubernetes secret."""
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            namespace = f.read()
        from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
        k8s_hook = KubernetesHook()
        secret = k8s_hook.core_v1_client.read_namespaced_secret("access-token", namespace)
        token_encoded = secret.data["AUTH_TOKEN"]
        return base64.b64decode(token_encoded).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to get token: {str(e)}")
        raise

def get_s3_client():
    """Return boto3 S3 client configured with JWT auth."""
    endpoint_url = "http://local-s3-service.ezdata-system.svc.cluster.local:30000"
    try:
        jwt_token = get_token()
        return boto3.client(
            "s3",
            aws_access_key_id=jwt_token,
            aws_secret_access_key="s3",
            endpoint_url=endpoint_url,
            use_ssl=False,
        )
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        raise

def validate_audio_file(audio_data, sample_rate, min_duration=1.0, max_duration=3600.0):
    """Validate audio file properties"""
    duration = len(audio_data) / sample_rate
    
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)")
    
    if duration > max_duration:
        raise ValueError(f"Audio too long: {duration:.2f}s (maximum: {max_duration}s)")
    
    if sample_rate < 8000:
        raise ValueError(f"Sample rate too low: {sample_rate}Hz (minimum: 8000Hz)")
    
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 2:
        raise ValueError(f"Too many channels: {audio_data.shape[1]} (maximum: 2)")
    
    logger.info(f"Audio validation passed: {duration:.2f}s at {sample_rate}Hz")
    return True

def get_file_checksum(s3_client, bucket, key):
    """Get MD5 checksum of S3 file"""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        etag = response['ETag'].strip('"')
        return etag
    except Exception as e:
        logger.error(f"Failed to get checksum for {key}: {str(e)}")
        return None

# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id='process_audio_all_files_improved',
    schedule_interval='0 */12 * * *',  # Fixed cron expression
    start_date=days_ago(1),
    tags=['audio', 'processing', 'improved'],
    catchup=False,
    access_control={'Admin': {'can_read', 'can_edit', 'can_delete'}},
) as dag:

    @task
    def list_raw_files():
        """List all WAV files in the raw bucket with validation"""
        try:
            s3 = get_s3_client()
            bucket = "audio-raw"
            
            logger.info(f"Listing files in bucket: {bucket}")
            resp = s3.list_objects_v2(Bucket=bucket)
            
            if 'Contents' not in resp:
                logger.warning("No files found in raw bucket")
                return []
            
            files = []
            for obj in resp["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".wav"):
                    # Check file size (basic validation)
                    size_mb = obj["Size"] / (1024 * 1024)
                    if 0.1 <= size_mb <= 500:  # Between 100KB and 500MB
                        files.append({
                            'key': key,
                            'size_mb': size_mb,
                            'last_modified': obj['LastModified'].isoformat()
                        })
                        logger.info(f"Found valid WAV file: {key} ({size_mb:.2f}MB)")
                    else:
                        logger.warning(f"Skipping {key}: invalid size {size_mb:.2f}MB")
            
            logger.info(f"Found {len(files)} valid WAV files")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list raw files: {str(e)}")
            raise

    @task
    def process_file(file_info: dict):
        """Process a single WAV file with comprehensive error handling"""
        file_key = file_info['key']
        
        try:
            # Install packages with error handling
            logger.info("Installing required packages...")
            result = subprocess.run(
                ["pip", "install", "--quiet", "librosa==0.10.1", "noisereduce==3.0.0", "soundfile==0.12.1", "numpy==1.24.3"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Package installation failed: {result.stderr}")
                raise RuntimeError(f"Package installation failed: {result.stderr}")
            
            import librosa
            import noisereduce as nr
            import soundfile as sf
            import numpy as np
            
            s3 = get_s3_client()
            input_bucket = "audio-raw"
            output_bucket = "audio-processed"
            output_key = file_key.replace(".wav", "_processed.wav")
            
            # Check if already processed with checksum validation
            try:
                existing_obj = s3.head_object(Bucket=output_bucket, Key=output_key)
                existing_size = existing_obj['ContentLength']
                
                # If processed file exists and is reasonable size, skip
                if existing_size > 1000:  # At least 1KB
                    logger.info(f"Skipping {file_key}, already processed ({existing_size} bytes)")
                    return {"status": "skipped", "reason": "already_processed"}
                else:
                    logger.warning(f"Existing processed file too small ({existing_size} bytes), reprocessing")
                    
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    logger.error(f"Error checking existing file: {str(e)}")
                    raise
                # File doesn't exist, continue processing
                pass
            
            # Download with validation
            logger.info(f"Downloading {file_key} from {input_bucket}")
            try:
                obj = s3.get_object(Bucket=input_bucket, Key=file_key)
                audio_bytes = BytesIO(obj['Body'].read())
                original_size = len(audio_bytes.getvalue())
                logger.info(f"Downloaded {original_size} bytes")
                
            except Exception as e:
                logger.error(f"Failed to download {file_key}: {str(e)}")
                raise
            
            # Load and validate audio
            logger.info("Loading audio file...")
            try:
                audio, sr = librosa.load(audio_bytes, sr=None)
                validate_audio_file(audio, sr)
                
            except Exception as e:
                logger.error(f"Failed to load/validate audio {file_key}: {str(e)}")
                raise
            
            # Processing steps with validation
            logger.info("Starting audio processing...")
            
            # Step 1: Noise reduction
            try:
                logger.info("Applying noise reduction...")
                audio_denoised = nr.reduce_noise(
                    y=audio, 
                    sr=sr, 
                    stationary=False, 
                    prop_decrease=0.8
                )
                
                # Validate output
                if np.any(np.isnan(audio_denoised)) or np.any(np.isinf(audio_denoised)):
                    raise ValueError("Noise reduction produced invalid values")
                    
            except Exception as e:
                logger.error(f"Noise reduction failed for {file_key}: {str(e)}")
                raise
            
            # Step 2: Normalization
            try:
                logger.info("Normalizing audio...")
                audio_norm = librosa.util.normalize(audio_denoised)
                
                # Validate normalization
                if np.max(np.abs(audio_norm)) > 1.1:
                    logger.warning(f"Normalization may have failed: max amplitude {np.max(np.abs(audio_norm))}")
                    
            except Exception as e:
                logger.error(f"Normalization failed for {file_key}: {str(e)}")
                raise
            
            # Step 3: Preemphasis
            try:
                logger.info("Applying preemphasis...")
                audio_final = librosa.effects.preemphasis(audio_norm, coef=0.95)
                
                # Final validation
                validate_audio_file(audio_final, sr)
                
            except Exception as e:
                logger.error(f"Preemphasis failed for {file_key}: {str(e)}")
                raise
            
            # Save processed audio
            logger.info("Saving processed audio...")
            try:
                buf = BytesIO()
                sf.write(buf, audio_final, sr, format='WAV', subtype='PCM_16')
                buf.seek(0)
                processed_data = buf.getvalue()
                processed_size = len(processed_data)
                
                # Validate processed file size
                if processed_size < 1000:
                    raise ValueError(f"Processed file too small: {processed_size} bytes")
                
                # Upload with metadata
                s3.put_object(
                    Bucket=output_bucket, 
                    Key=output_key, 
                    Body=processed_data,
                    Metadata={
                        'original_file': file_key,
                        'original_size': str(original_size),
                        'processed_size': str(processed_size),
                        'sample_rate': str(sr),
                        'processing_version': '2.0'
                    }
                )
                
                logger.info(f"Successfully processed and uploaded: {output_key} ({processed_size} bytes)")
                
                return {
                    "status": "success",
                    "original_size": original_size,
                    "processed_size": processed_size,
                    "sample_rate": sr,
                    "output_key": output_key
                }
                
            except Exception as e:
                logger.error(f"Failed to save processed audio for {file_key}: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Processing failed for {file_key}: {str(e)}")
            
            # Try to clean up any partial uploads
            try:
                s3.delete_object(Bucket="audio-processed", Key=output_key)
                logger.info(f"Cleaned up partial upload: {output_key}")
            except:
                pass
                
            return {
                "status": "failed",
                "error": str(e),
                "file_key": file_key
            }

    @task
    def summarize_results(process_results):
        """Summarize processing results"""
        total_files = len(process_results)
        successful = sum(1 for r in process_results if r.get('status') == 'success')
        skipped = sum(1 for r in process_results if r.get('status') == 'skipped')
        failed = sum(1 for r in process_results if r.get('status') == 'failed')
        
        logger.info(f"Processing Summary:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")
        
        # Log failed files
        for result in process_results:
            if result.get('status') == 'failed':
                logger.error(f"Failed: {result.get('file_key')} - {result.get('error')}")
        
        return {
            "total": total_files,
            "successful": successful,
            "skipped": skipped,
            "failed": failed
        }

    # -----------------------------
    # DAG Flow
    # -----------------------------
    raw_files = list_raw_files()
    process_results = process_file.expand(file_info=raw_files)
    summary = summarize_results(process_results)
