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

def install_audio_packages():
    """Install required audio processing packages"""
    packages = [
        "torch==2.0.1", 
        "torchaudio==2.0.2", 
        "librosa==0.10.1", 
        "noisereduce==3.0.0", 
        "soundfile==0.12.1", 
        "numpy==1.24.3",
        "scipy==1.10.1",
        "webrtcvad==2.0.10"  # Added for enhanced VAD
    ]
    
    logger.info("Installing required audio packages...")
    result = subprocess.run(
        ["pip", "install", "--quiet", "--no-cache-dir"] + packages,
        capture_output=True,
        text=True,
        timeout=900
    )
    
    if result.returncode != 0:
        logger.error(f"Package installation failed: {result.stderr}")
        raise RuntimeError(f"Package installation failed: {result.stderr}")
    
    logger.info("Audio packages installed successfully")

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

# -----------------------------
# Enhanced Audio Processing Functions
# -----------------------------
def enhanced_voice_activity_detection(audio, sr, frame_duration=20):
    """Enhanced VAD with multiple aggressiveness levels and smoothing"""
    import webrtcvad
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    
    vad_levels = [1, 2, 3]
    audio_16bit = (audio * 32767).astype(np.int16)
    frame_size = int(sr * frame_duration / 1000)
    
    # Pad audio
    padding = frame_size - (len(audio_16bit) % frame_size)
    if padding != frame_size:
        audio_16bit = np.pad(audio_16bit, (0, padding), mode='constant')
    
    # Collect votes from different VAD levels
    vad_votes = []
    for level in vad_levels:
        vad = webrtcvad.Vad(level)
        voice_frames = []
        
        for i in range(0, len(audio_16bit), frame_size):
            frame = audio_16bit[i:i+frame_size].tobytes()
            try:
                is_speech = vad.is_speech(frame, sr)
            except:
                is_speech = False
            voice_frames.append(is_speech)
        
        vad_votes.append(voice_frames)
    
    # Majority voting
    final_voice_frames = []
    for i in range(len(vad_votes[0])):
        votes = sum(vad_vote[i] for vad_vote in vad_votes)
        final_voice_frames.append(votes >= 2)  # At least 2 out of 3 agree
    
    # Convert to sample-level mask
    voice_mask = np.repeat(final_voice_frames, frame_size)[:len(audio)]
    
    # Morphological operations to clean up mask
    voice_mask = binary_dilation(voice_mask, iterations=2)
    voice_mask = binary_erosion(voice_mask, iterations=1)
    
    return voice_mask

def bandpass_filter(audio, sr, low_freq=300, high_freq=3400):
    """Apply bandpass filter optimized for speech frequencies"""
    from scipy.signal import butter, filtfilt
    import numpy as np
    
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, audio)

def adaptive_spectral_subtraction(audio, sr, alpha=1.5, beta=0.02):
    """Memory-efficient adaptive spectral subtraction"""
    import librosa
    import numpy as np
    
    chunk_size = sr * 10  # Process 10 seconds at a time
    n_fft = 512
    hop_length = 256
    
    if len(audio) <= chunk_size:
        return _process_spectral_chunk(audio, sr, alpha, beta, n_fft, hop_length)
    
    # Process large files in chunks
    processed_chunks = []
    overlap = n_fft
    
    for start in range(0, len(audio), chunk_size - overlap):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        
        processed_chunk = _process_spectral_chunk(chunk, sr, alpha, beta, n_fft, hop_length)
        
        if start > 0:
            processed_chunk = processed_chunk[overlap//2:]
        
        processed_chunks.append(processed_chunk)
    
    return np.concatenate(processed_chunks)

def _process_spectral_chunk(audio, sr, alpha, beta, n_fft, hop_length):
    """Process a single chunk with spectral subtraction"""
    import librosa
    import numpy as np
    
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

def multi_band_dynamic_range_compression(audio, sr):
    """Apply frequency-specific compression"""
    import numpy as np
    
    # Split into frequency bands
    low_band = bandpass_filter(audio, sr, 150, 500)
    mid_band = bandpass_filter(audio, sr, 500, 2000)
    high_band = bandpass_filter(audio, sr, 2000, 3400)
    
    def compress_band(band, threshold=0.2, ratio=3.0):
        compressed = band.copy()
        above_threshold = np.abs(band) > threshold
        compressed[above_threshold] = np.sign(band[above_threshold]) * (
            threshold + (np.abs(band[above_threshold]) - threshold) / ratio
        )
        return compressed
    
    low_compressed = compress_band(low_band, threshold=0.15, ratio=2.5)
    mid_compressed = compress_band(mid_band, threshold=0.25, ratio=3.0)
    high_compressed = compress_band(high_band, threshold=0.3, ratio=4.0)
    
    return 0.3 * low_compressed + 0.5 * mid_compressed + 0.2 * high_compressed

def automatic_gain_control(audio, target_rms=0.1, attack_time=0.005, release_time=0.05, sr=16000):
    """Fast AGC specifically tuned for speech"""
    import numpy as np
    from scipy.signal import medfilt
    
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
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
            target_gain = target_rms / rms
            target_gain = np.clip(target_gain, 0.1, 10.0)
            
            if target_gain < current_gain:
                current_gain = current_gain + (target_gain - current_gain) / attack_samples
            else:
                current_gain = current_gain + (target_gain - current_gain) / release_samples
        
        gains.append(current_gain)
    
    # Interpolate gains to match audio length
    gain_indices = np.arange(0, len(audio), hop_size)[:len(gains)]
    audio_indices = np.arange(len(audio))
    interpolated_gains = np.interp(audio_indices, gain_indices, gains)
    
    return audio * interpolated_gains

def speaker_transition_enhancement(audio, sr):
    """Enhance speaker transitions by emphasizing spectral changes"""
    import librosa
    import numpy as np
    
    # Compute spectral centroid over time
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
    
    # Find rapid changes (potential speaker transitions)
    centroid_diff = np.abs(np.diff(centroids))
    transition_threshold = np.percentile(centroid_diff, 80)
    
    # Create enhancement mask
    enhancement_mask = np.ones_like(audio)
    
    # For each significant transition, apply local enhancement
    for i, diff in enumerate(centroid_diff):
        if diff > transition_threshold:
            start_sample = i * 512
            end_sample = min(start_sample + 1024, len(audio))
            enhancement_mask[start_sample:end_sample] *= 1.1
    
    return audio * enhancement_mask

def speaker_aware_silence_removal(audio, voice_mask, min_gap_ms=200, padding_ms=100, sr=16000):
    """Intelligent silence removal that preserves speaker transitions"""
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
        
        # Add padding around voice regions
        pad_start = max(0, start - padding_samples)
        pad_end = min(len(voice_mask), end + padding_samples)
        enhanced_mask[pad_start:pad_end] = True
        
        # If gap to next voice region is small, include it
        if i < len(voice_starts) - 1:
            next_start = voice_starts[i + 1]
            gap_size = next_start - end
            if gap_size < min_gap_samples:
                enhanced_mask[end:next_start] = True
    
    return audio[enhanced_mask]

def quality_improvement_processing(audio, sr):
    """
    Step 1: Enhanced quality improvement processing
    """
    import librosa
    import numpy as np
    
    logger.info("Step 1: Applying enhanced quality improvements...")
    
    # 1.1: Enhanced bandpass filter
    logger.info("  - Applying enhanced bandpass filter...")
    audio_filtered = bandpass_filter(audio, sr, low_freq=300, high_freq=3400)
    
    # 1.2: Adaptive spectral subtraction (better than simple noise reduction)
    logger.info("  - Applying adaptive spectral subtraction...")
    audio_denoised = adaptive_spectral_subtraction(audio_filtered, sr)
    
    # 1.3: Multi-band compression
    logger.info("  - Applying multi-band compression...")
    audio_compressed = multi_band_dynamic_range_compression(audio_denoised, sr)
    
    # 1.4: Enhanced normalization
    logger.info("  - Applying enhanced normalization...")
    audio_normalized = librosa.util.normalize(audio_compressed)
    
    # 1.5: Preemphasis for speech clarity
    logger.info("  - Applying preemphasis...")
    audio_final = librosa.effects.preemphasis(audio_normalized, coef=0.95)
    
    # Validate output
    if np.any(np.isnan(audio_final)) or np.any(np.isinf(audio_final)):
        raise ValueError("Step 1 processing produced invalid values")
    
    logger.info("Step 1 complete: Enhanced quality improvement finished")
    return audio_final

def enhanced_diarization_processing(audio, sr):
    """
    Step 2: Enhanced processing for speaker diarization with advanced techniques
    """
    import librosa
    import numpy as np
    
    logger.info("Step 2: Applying advanced diarization enhancements...")
    
    # 2.1: Enhanced Voice Activity Detection (on original improved audio)
    logger.info("  - Performing enhanced voice activity detection...")
    voice_mask = enhanced_voice_activity_detection(audio, sr)
    voice_ratio = np.sum(voice_mask) / len(voice_mask)
    logger.info(f"  - Voice activity detected: {voice_ratio:.1%} of audio")
    
    # 2.2: Speaker transition enhancement
    logger.info("  - Enhancing speaker transitions...")
    audio_transitions = speaker_transition_enhancement(audio, sr)
    
    # 2.3: Fast automatic gain control for speaker level balancing
    logger.info("  - Applying fast automatic gain control...")
    audio_agc = automatic_gain_control(audio_transitions, sr=sr)
    
    # 2.4: Speaker-aware silence removal
    logger.info("  - Applying speaker-aware silence removal...")
    # Adjust voice mask if audio length changed
    if len(audio_agc) != len(voice_mask):
        logger.info(f"  - Adjusting voice mask: original {len(voice_mask)}, current {len(audio_agc)}")
        if len(audio_agc) < len(voice_mask):
            voice_mask = voice_mask[:len(audio_agc)]
        else:
            padding = len(audio_agc) - len(voice_mask)
            voice_mask = np.pad(voice_mask, (0, padding), mode='constant', constant_values=False)
    
    audio_final = speaker_aware_silence_removal(audio_agc, voice_mask, sr=sr)
    
    # Final normalization
    audio_final = librosa.util.normalize(audio_final)
    
    # Validate output
    if np.any(np.isnan(audio_final)) or np.any(np.isinf(audio_final)):
        raise ValueError("Step 2 processing produced invalid values")
    
    logger.info("Step 2 complete: Advanced diarization enhancements finished")
    return audio_final

# -----------------------------
# DAG Definition
# -----------------------------
with DAG(
    dag_id='enhanced_two_step_audio_processing',
    schedule_interval='0 */12 * * *',
    start_date=days_ago(1),
    tags=['audio', 'processing', 'enhanced', 'diarization'],
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
                    size_mb = obj["Size"] / (1024 * 1024)
                    if 0.1 <= size_mb <= 500:
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
    def process_all_files_step1(file_list):
        """Step 1: Enhanced quality improvement processing"""
        if not file_list:
            logger.warning("No files to process")
            return []
        
        # Install packages once for all files
        try:
            install_audio_packages()
        except Exception as e:
            logger.error(f"Package installation failed: {str(e)}")
            return [{"status": "failed", "error": f"Package installation failed: {str(e)}", "step": "package_install"}]
        
        import librosa
        import soundfile as sf
        import numpy as np
        
        s3 = get_s3_client()
        input_bucket = "audio-raw"
        output_bucket = "audio-improved"
        
        results = []
        
        for file_info in file_list:
            file_key = file_info['key']
            output_key = file_key.replace(".wav", "_improved.wav")
            
            try:
                # Check if already processed
                try:
                    existing_obj = s3.head_object(Bucket=output_bucket, Key=output_key)
                    existing_size = existing_obj['ContentLength']
                    
                    if existing_size > 1000:
                        logger.info(f"Step 1: Skipping {file_key}, already improved")
                        results.append({
                            "status": "skipped", 
                            "reason": "already_improved", 
                            "output_key": output_key,
                            "file_key": file_key
                        })
                        continue
                        
                except s3.exceptions.ClientError as e:
                    if e.response['Error']['Code'] != '404':
                        logger.error(f"Error checking existing file: {str(e)}")
                        results.append({
                            "status": "failed",
                            "error": f"S3 check failed: {str(e)}",
                            "file_key": file_key,
                            "step": "step1"
                        })
                        continue
                
                # Download and load audio
                logger.info(f"Step 1: Processing {file_key}")
                obj = s3.get_object(Bucket=input_bucket, Key=file_key)
                audio_bytes = BytesIO(obj['Body'].read())
                original_size = len(audio_bytes.getvalue())
                
                logger.info(f"  - Loading audio file...")
                audio, sr = librosa.load(audio_bytes, sr=16000)
                validate_audio_file(audio, sr)
                
                # Apply Enhanced Step 1 processing
                audio_improved = quality_improvement_processing(audio, sr)
                
                # Save improved audio
                logger.info(f"  - Saving improved audio...")
                buf = BytesIO()
                sf.write(buf, audio_improved, sr, format='WAV', subtype='PCM_16')
                buf.seek(0)
                improved_data = buf.getvalue()
                improved_size = len(improved_data)
                
                if improved_size < 1000:
                    raise ValueError(f"Improved file too small: {improved_size} bytes")
                
                s3.put_object(
                    Bucket=output_bucket, 
                    Key=output_key, 
                    Body=improved_data,
                    Metadata={
                        'original_file': file_key,
                        'original_size': str(original_size),
                        'improved_size': str(improved_size),
                        'sample_rate': str(sr),
                        'processing_step': 'step1_enhanced_quality',
                        'processing_version': '2.0'
                    }
                )
                
                logger.info(f"  - Successfully improved: {output_key} ({improved_size} bytes)")
                
                results.append({
                    "status": "success",
                    "original_size": original_size,
                    "improved_size": improved_size,
                    "sample_rate": sr,
                    "output_key": output_key,
                    "file_key": file_key
                })
                
            except Exception as e:
                logger.error(f"Step 1 processing failed for {file_key}: {str(e)}")
                results.append({
                    "status": "failed",
                    "error": str(e),
                    "file_key": file_key,
                    "step": "step1"
                })
        
        return results

    @task
    def process_all_files_step2(step1_results):
        """Step 2: Advanced diarization optimization processing"""
        if not step1_results:
            logger.warning("No step1 results to process")
            return []
        
        # Filter for successful step1 results
        successful_files = [r for r in step1_results if r.get('status') == 'success']
        
        if not successful_files:
            logger.warning("No successful step1 files to process in step2")
            return step1_results
        
        # Install packages
        try:
            install_audio_packages()
        except Exception as e:
            logger.error(f"Package installation failed in step2: {str(e)}")
            failed_results = []
            for result in step1_results:
                if result.get('status') == 'success':
                    failed_results.append({
                        "status": "failed",
                        "error": f"Step2 package installation failed: {str(e)}",
                        "file_key": result.get('file_key'),
                        "step": "step2_package_install"
                    })
                else:
                    failed_results.append(result)
            return failed_results
        
        import librosa
        import soundfile as sf
        import numpy as np
        
        s3 = get_s3_client()
        input_bucket = "audio-improved"
        output_bucket = "audio-enhanced"
        
        results = []
        
        for file_result in step1_results:
            if file_result.get('status') != 'success':
                results.append(file_result)
                continue
            
            file_key = file_result['file_key']
            improved_key = file_result['output_key']
            output_key = file_key.replace(".wav", "_enhanced.wav")
            
            try:
                # Check if already processed
                try:
                    existing_obj = s3.head_object(Bucket=output_bucket, Key=output_key)
                    existing_size = existing_obj['ContentLength']
                    
                    if existing_size > 1000:
                        logger.info(f"Step 2: Skipping {file_key}, already enhanced")
                        results.append({
                            "status": "skipped", 
                            "reason": "already_enhanced", 
                            "output_key": output_key,
                            "file_key": file_key
                        })
                        continue
                        
                except s3.exceptions.ClientError as e:
                    if e.response['Error']['Code'] != '404':
                        logger.error(f"Error checking existing file: {str(e)}")
                        results.append({
                            "status": "failed",
                            "error": f"S3 check failed: {str(e)}",
                            "file_key": file_key,
                            "step": "step2"
                        })
                        continue
                
                # Download improved audio from step 1
                logger.info(f"Step 2: Processing {file_key}")
                obj = s3.get_object(Bucket=input_bucket, Key=improved_key)
                audio_bytes = BytesIO(obj['Body'].read())
                improved_size = len(audio_bytes.getvalue())
                
                logger.info(f"  - Loading improved audio file...")
                audio, sr = librosa.load(audio_bytes, sr=16000)
                validate_audio_file(audio, sr)
                
                # Apply Enhanced Step 2 processing
                audio_enhanced = enhanced_diarization_processing(audio, sr)
                
                # Save enhanced audio
                logger.info(f"  - Saving enhanced audio...")
                buf = BytesIO()
                sf.write(buf, audio_enhanced, sr, format='WAV', subtype='PCM_16')
                buf.seek(0)
                enhanced_data = buf.getvalue()
                enhanced_size = len(enhanced_data)
                
                if enhanced_size < 1000:
                    raise ValueError(f"Enhanced file too small: {enhanced_size} bytes")
                
                s3.put_object(
                    Bucket=output_bucket, 
                    Key=output_key, 
                    Body=enhanced_data,
                    Metadata={
                        'original_file': file_key,
                        'improved_file': improved_key,
                        'improved_size': str(improved_size),
                        'enhanced_size': str(enhanced_size),
                        'sample_rate': str(sr),
                        'processing_step': 'step2_advanced_diarization',
                        'processing_version': '2.0'
                    }
                )
                
                logger.info(f"  - Successfully enhanced: {output_key} ({enhanced_size} bytes)")
                
                results.append({
                    "status": "success",
                    "improved_size": improved_size,
                    "enhanced_size": enhanced_size,
                    "sample_rate": sr,
                    "output_key": output_key,
                    "file_key": file_key,
                    "step1_output": improved_key
                })
                
            except Exception as e:
                logger.error(f"Step 2 processing failed for {file_key}: {str(e)}")
                results.append({
                    "status": "failed",
                    "error": str(e),
                    "file_key": file_key,
                    "step": "step2"
                })
        
        return results

    @task
    def summarize_results(process_results):
        """Summarize enhanced processing results"""
        total_files = len(process_results)
        step1_successful = sum(1 for r in process_results if r.get('status') == 'success' and 'improved_size' in r)
        step2_successful = sum(1 for r in process_results if r.get('status') == 'success' and 'enhanced_size' in r)
        skipped = sum(1 for r in process_results if r.get('status') == 'skipped')
        failed = sum(1 for r in process_results if r.get('status') == 'failed')
        
        logger.info(f"Enhanced Two-Step Processing Summary:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Step 1 (Enhanced Quality) successful: {step1_successful}")
        logger.info(f"  Step 2 (Advanced Diarization) successful: {step2_successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")
        
        # Calculate processing improvements
        total_original_size = sum(r.get('original_size', 0) for r in process_results if r.get('original_size'))
        total_enhanced_size = sum(r.get('enhanced_size', 0) for r in process_results if r.get('enhanced_size'))
        
        if total_original_size > 0:
            compression_ratio = total_enhanced_size / total_original_size
            logger.info(f"  Audio compression ratio: {compression_ratio:.2f}")
        
        # Log failed files with detailed step information
        for result in process_results:
            if result.get('status') == 'failed':
                step = result.get('step', 'unknown')
                logger.error(f"Failed at {step}: {result.get('file_key')} - {result.get('error')}")
        
        # Log quality improvements
        successful_results = [r for r in process_results if r.get('status') == 'success' and 'enhanced_size' in r]
        if successful_results:
            logger.info(f"Enhanced processing optimized {len(successful_results)} files for:")
            logger.info(f"  - Improved speaker diarization accuracy")
            logger.info(f"  - Better transcription quality")
            logger.info(f"  - Enhanced voice activity detection")
            logger.info(f"  - Speaker transition preservation")
        
        return {
            "total": total_files,
            "step1_successful": step1_successful,
            "step2_successful": step2_successful,
            "skipped": skipped,
            "failed": failed,
            "compression_ratio": compression_ratio if total_original_size > 0 else None,
            "processing_version": "2.0_enhanced"
        }

    # -----------------------------
    # DAG Flow
    # -----------------------------
    raw_files = list_raw_files()
    step1_results = process_all_files_step1(raw_files)
    step2_results = process_all_files_step2(step1_results)
    summary = summarize_results(step2_results)
