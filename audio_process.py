import base64
import boto3
import logging
import hashlib
from io import BytesIO
from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_token():
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

def install_audio_packages():
    packages = [
        "https://files.pythonhosted.org/packages/py3/p/py_webrtcvad_wheels/py_webrtcvad_wheels-2.0.10-py3-none-any.whl",
        "torch==2.0.1", 
        "librosa==0.10.1", 
        "soundfile==0.12.1", 
        "numpy==1.24.3",
        "scipy==1.10.1"
    ]
    
    result = subprocess.run(
        ["pip", "install", "--user", "--no-cache-dir"] + packages,
        capture_output=True, text=True, timeout=900
    )
    
    if result.returncode != 0:
        logger.error(f"Package installation failed: {result.stderr}")
        raise RuntimeError(f"Package installation failed: {result.stderr}")
    
    logger.info("Audio packages installed successfully")

def enhanced_voice_activity_detection(audio, sr, frame_duration=20):
    """Enhanced VAD using webrtcvad with fallback"""
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    
    try:
        import webrtcvad
        
        vad = webrtcvad.Vad(2)
        audio_16bit = (audio * 32767).astype(np.int16)
        frame_size = int(sr * frame_duration / 1000)
        
        padding = frame_size - (len(audio_16bit) % frame_size)
        if padding != frame_size:
            audio_16bit = np.pad(audio_16bit, (0, padding), mode='constant')
        
        voice_frames = []
        for i in range(0, len(audio_16bit), frame_size):
            frame = audio_16bit[i:i+frame_size].tobytes()
            try:
                is_speech = vad.is_speech(frame, sr)
            except:
                is_speech = False
            voice_frames.append(is_speech)
        
        voice_mask = np.repeat(voice_frames, frame_size)[:len(audio)]
        voice_mask = binary_dilation(voice_mask, iterations=2)
        voice_mask = binary_erosion(voice_mask, iterations=1)
        
        logger.info("Using WebRTC VAD")
        return voice_mask
        
    except ImportError:
        logger.warning("webrtcvad not available, using fallback VAD")
        
        # Fallback VAD implementation
        import librosa
        from scipy.signal import medfilt
        
        frame_length = int(sr * frame_duration / 1000)
        hop_length = frame_length // 2
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Simple thresholds
        energy_threshold = np.percentile(rms, 25)
        zcr_min, zcr_max = np.percentile(zcr, [15, 85])
        centroid_threshold = np.percentile(spectral_centroid, 30)
        
        # Voice activity detection
        energy_voice = rms > energy_threshold
        spectral_voice = (zcr >= zcr_min) & (zcr <= zcr_max) & (spectral_centroid > centroid_threshold)
        
        # Simple voting
        voice_frames = energy_voice | spectral_voice
        voice_frames = medfilt(voice_frames.astype(int), kernel_size=3).astype(bool)
        
        # Convert to sample-level mask
        voice_mask = np.repeat(voice_frames, hop_length)[:len(audio)]
        voice_mask = binary_dilation(voice_mask, iterations=2)
        voice_mask = binary_erosion(voice_mask, iterations=1)
        
        return voice_mask

def bandpass_filter(audio, sr, low_freq=300, high_freq=3400):
    from scipy.signal import butter, filtfilt
    import numpy as np
    
    nyquist = sr / 2
    low = max(0.001, min(low_freq / nyquist, 0.999))
    high = max(low + 0.001, min(high_freq / nyquist, 0.999))
    
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, audio)

def adaptive_spectral_subtraction(audio, sr, alpha=1.5, beta=0.02):
    import librosa
    import numpy as np
    
    n_fft = 512
    hop_length = 256
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from quietest 20% of frames
    frame_energy = np.mean(magnitude**2, axis=0)
    noise_threshold = np.percentile(frame_energy, 20)
    noise_frames = frame_energy < noise_threshold
    
    if np.sum(noise_frames) > 2:
        noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    else:
        noise_frames = min(5, magnitude.shape[1])
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction
    clean_magnitude = magnitude - alpha * noise_spectrum
    clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
    
    # Reconstruct
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
    
    return clean_audio

def automatic_gain_control(audio, target_rms=0.1, sr=16000):
    import numpy as np
    from scipy.signal import medfilt
    
    window_size = 512
    hop_size = 256
    
    rms_values = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    rms_values = medfilt(rms_values, kernel_size=3)
    
    gains = []
    current_gain = 1.0
    
    for rms in rms_values:
        if rms > 0:
            target_gain = np.clip(target_rms / rms, 0.1, 10.0)
            current_gain = current_gain + (target_gain - current_gain) * 0.1
        gains.append(current_gain)
    
    # Interpolate gains
    gain_indices = np.arange(0, len(audio), hop_size)[:len(gains)]
    audio_indices = np.arange(len(audio))
    interpolated_gains = np.interp(audio_indices, gain_indices, gains)
    
    return audio * interpolated_gains

def speaker_aware_silence_removal(audio, voice_mask, min_gap_ms=200, padding_ms=100, sr=16000):
    import numpy as np
    
    min_gap_samples = int(min_gap_ms * sr / 1000)
    padding_samples = int(padding_ms * sr / 1000)
    
    # Find voice regions
    voice_changes = np.diff(voice_mask.astype(int))
    voice_starts = np.where(voice_changes == 1)[0] + 1
    voice_ends = np.where(voice_changes == -1)[0] + 1
    
    # Handle edge cases
    if len(voice_starts) == 0 and len(voice_ends) == 0:
        return audio
    
    if len(voice_starts) > 0 and (len(voice_ends) == 0 or voice_starts[0] < voice_ends[0]):
        voice_ends = np.append(voice_ends, len(voice_mask))
    
    if len(voice_ends) > 0 and (len(voice_starts) == 0 or voice_ends[0] < voice_starts[0]):
        voice_starts = np.insert(voice_starts, 0, 0)
    
    # Create enhanced mask
    enhanced_mask = np.zeros_like(voice_mask, dtype=bool)
    
    for i in range(len(voice_starts)):
        start = voice_starts[i]
        end = voice_ends[i] if i < len(voice_ends) else len(voice_mask)
        
        # Add padding
        pad_start = max(0, start - padding_samples)
        pad_end = min(len(voice_mask), end + padding_samples)
        enhanced_mask[pad_start:pad_end] = True
        
        # Connect small gaps
        if i < len(voice_starts) - 1:
            next_start = voice_starts[i + 1]
            gap_size = next_start - end
            if gap_size < min_gap_samples:
                enhanced_mask[end:next_start] = True
    
    return audio[enhanced_mask]

def quality_improvement_processing(audio, sr):
    import librosa
    import numpy as np
    
    logger.info("Step 1: Quality improvement processing...")
    
    # Bandpass filter
    audio_filtered = bandpass_filter(audio, sr)
    
    # Spectral subtraction
    audio_denoised = adaptive_spectral_subtraction(audio_filtered, sr)
    
    # Automatic gain control
    audio_agc = automatic_gain_control(audio_denoised, sr=sr)
    
    # Normalization
    audio_final = librosa.util.normalize(audio_agc)
    
    # Validate
    if np.any(np.isnan(audio_final)) or np.any(np.isinf(audio_final)):
        raise ValueError("Step 1 processing produced invalid values")
    
    logger.info("Step 1 complete")
    return audio_final

def enhanced_diarization_processing(audio, sr):
    import librosa
    import numpy as np
    
    logger.info("Step 2: Diarization processing...")
    
    # Voice Activity Detection
    voice_mask = enhanced_voice_activity_detection(audio, sr)
    voice_ratio = np.sum(voice_mask) / len(voice_mask)
    logger.info(f"Voice activity: {voice_ratio:.1%}")
    
    # Apply AGC again for speaker balancing
    audio_agc = automatic_gain_control(audio, sr=sr)
    
    # Remove silence intelligently
    if len(audio_agc) != len(voice_mask):
        if len(audio_agc) < len(voice_mask):
            voice_mask = voice_mask[:len(audio_agc)]
        else:
            padding = len(audio_agc) - len(voice_mask)
            voice_mask = np.pad(voice_mask, (0, padding), mode='constant', constant_values=False)
    
    audio_final = speaker_aware_silence_removal(audio_agc, voice_mask, sr=sr)
    
    # Final normalization
    audio_final = librosa.util.normalize(audio_final)
    
    if np.any(np.isnan(audio_final)) or np.any(np.isinf(audio_final)):
        raise ValueError("Step 2 processing produced invalid values")
    
    logger.info("Step 2 complete")
    return audio_final

with DAG(
    dag_id='fixed_audio_processing',
    schedule_interval='0 */12 * * *',
    start_date=days_ago(1),
    tags=['audio', 'processing', 'fixed'],
    catchup=False,
) as dag:

    @task
    def list_raw_files():
        try:
            s3 = get_s3_client()
            resp = s3.list_objects_v2(Bucket="audio-raw")
            
            if 'Contents' not in resp:
                return []
            
            files = []
            for obj in resp["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".wav"):
                    size_mb = obj["Size"] / (1024 * 1024)
                    if 0.1 <= size_mb <= 500:
                        files.append({'key': key, 'size_mb': size_mb})
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise

    @task
    def process_step1(file_list):
        if not file_list:
            return []
        
        try:
            install_audio_packages()
        except Exception as e:
            return [{"status": "failed", "error": str(e), "step": "install"}]
        
        import librosa
        import soundfile as sf
        import numpy as np
        
        s3 = get_s3_client()
        results = []
        
        for file_info in file_list:
            file_key = file_info['key']
            output_key = file_key.replace(".wav", "_improved.wav")
            
            try:
                # Check if exists
                try:
                    existing = s3.head_object(Bucket="audio-improved", Key=output_key)
                    if existing['ContentLength'] > 1000:
                        results.append({"status": "skipped", "file_key": file_key, "output_key": output_key})
                        continue
                except:
                    pass
                
                # Process
                obj = s3.get_object(Bucket="audio-raw", Key=file_key)
                audio_bytes = BytesIO(obj['Body'].read())
                
                audio, sr = librosa.load(audio_bytes, sr=16000)
                audio_improved = quality_improvement_processing(audio, sr)
                
                # Save
                buf = BytesIO()
                sf.write(buf, audio_improved, sr, format='WAV', subtype='PCM_16')
                buf.seek(0)
                
                s3.put_object(Bucket="audio-improved", Key=output_key, Body=buf.getvalue())
                
                results.append({"status": "success", "file_key": file_key, "output_key": output_key})
                
            except Exception as e:
                results.append({"status": "failed", "file_key": file_key, "error": str(e)})
        
        return results

    @task
    def process_step2(step1_results):
        if not step1_results:
            return []
        
        successful = [r for r in step1_results if r.get('status') == 'success']
        if not successful:
            return step1_results
        
        try:
            install_audio_packages()
        except Exception as e:
            return [{"status": "failed", "error": str(e), "step": "install"}]
        
        import librosa
        import soundfile as sf
        import numpy as np
        
        s3 = get_s3_client()
        results = []
        
        for result in step1_results:
            if result.get('status') != 'success':
                results.append(result)
                continue
                
            file_key = result['file_key']
            improved_key = result['output_key']
            output_key = file_key.replace(".wav", "_enhanced.wav")
            
            try:
                # Check if exists
                try:
                    existing = s3.head_object(Bucket="audio-enhanced", Key=output_key)
                    if existing['ContentLength'] > 1000:
                        results.append({"status": "skipped", "file_key": file_key, "output_key": output_key})
                        continue
                except:
                    pass
                
                # Process
                obj = s3.get_object(Bucket="audio-improved", Key=improved_key)
                audio_bytes = BytesIO(obj['Body'].read())
                
                audio, sr = librosa.load(audio_bytes, sr=16000)
                audio_enhanced = enhanced_diarization_processing(audio, sr)
                
                # Save
                buf = BytesIO()
                sf.write(buf, audio_enhanced, sr, format='WAV', subtype='PCM_16')
                buf.seek(0)
                
                s3.put_object(Bucket="audio-enhanced", Key=output_key, Body=buf.getvalue())
                
                results.append({"status": "success", "file_key": file_key, "output_key": output_key})
                
            except Exception as e:
                results.append({"status": "failed", "file_key": file_key, "error": str(e)})
        
        return results

    @task
    def summarize_results(results):
        total = len(results)
        success = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        skipped = sum(1 for r in results if r.get('status') == 'skipped')
        
        logger.info(f"Processing complete: {success} success, {failed} failed, {skipped} skipped")
        return {"total": total, "success": success, "failed": failed, "skipped": skipped}

    # DAG Flow
    files = list_raw_files()
    step1 = process_step1(files)
    step2 = process_step2(step1)
    summary = summarize_results(step2)
