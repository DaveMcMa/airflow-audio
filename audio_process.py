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
    
    logger.info("Installing required audio packages...")
    
    # Install webrtcvad-wheels first with specific flags and longer timeout
    logger.info("Installing webrtcvad-wheels...")
    webrtc_result = subprocess.run(
        [
            "pip", "install", "--no-cache-dir", 
            "--only-binary=:all:",  # Force binary wheel
            "--prefer-binary",      # Prefer binary over source
            "--verbose",            # Add verbose output
            "webrtcvad-wheels==2.0.14"
        ],
        capture_output=True,
        text=True,
        timeout=600  # Increase to 10 minutes
    )
    
    if webrtc_result.returncode != 0:
        logger.error(f"WebRTC installation failed: {webrtc_result.stderr}")
        logger.error(f"WebRTC stdout: {webrtc_result.stdout}")
        raise RuntimeError(f"WebRTC installation failed: {webrtc_result.stderr}")
    
    logger.info(f"WebRTC installation successful: {webrtc_result.stdout}")
    
    # Install other packages one by one to see which one might be causing issues
    packages = [
        "torch==2.0.1", 
        "torchaudio==2.0.2", 
        "librosa==0.10.1", 
        "noisereduce==3.0.0", 
        "soundfile==0.12.1", 
        "numpy==1.24.3",
        "scipy==1.10.1"
    ]
    
    for package in packages:
        logger.info(f"Installing {package}...")
        result = subprocess.run(
            ["pip", "install", "--no-cache-dir", package],
            capture_output=True,
            text=True,
            timeout=900
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install {package}: {result.stderr}")
            raise RuntimeError(f"Failed to install {package}: {result.stderr}")
        
        logger.info(f"Successfully installed {package}")
    
    logger.info("All audio packages installed successfully")
    
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
# STEP 1: Quality Improvement Functions (from Document 14)
# -----------------------------
def voice_activity_detection(audio, sr, frame_duration=30):
    """WebRTC VAD exactly as in Document 14"""
    try:
        import webrtcvad
        import numpy as np
        
        vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Frame size in samples
        frame_size = int(sr * frame_duration / 1000)
        
        # Pad audio to be divisible by frame_size
        padding = frame_size - (len(audio_16bit) % frame_size)
        if padding != frame_size:
            audio_16bit = np.pad(audio_16bit, (0, padding), mode='constant')
        
        voice_frames = []
        for i in range(0, len(audio_16bit), frame_size):
            frame = audio_16bit[i:i+frame_size].tobytes()
            try:
                is_speech = vad.is_speech(frame, sr)
            except:
                is_speech = False  # Default to non-speech for problematic frames
            voice_frames.append(is_speech)
        
        # Convert back to sample-level mask
        voice_mask = np.repeat(voice_frames, frame_size)[:len(audio)]
        return voice_mask
        
    except ImportError:
        logger.warning("webrtcvad not available, using fallback")
        import librosa
        import numpy as np
        
        # Simple energy-based fallback
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        energy_threshold = np.percentile(rms, 25)
        voice_frames = rms > energy_threshold
        voice_mask = np.repeat(voice_frames, hop_length)[:len(audio)]
        return voice_mask

def bandpass_filter(audio, sr, low_freq=300, high_freq=3400):
    """Apply bandpass filter optimized for speech frequencies"""
    from scipy.signal import butter, filtfilt
    import numpy as np
    
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, audio)

def spectral_subtraction(audio, sr, alpha=2.0, beta=0.01):
    """Advanced spectral subtraction exactly as in Document 14"""
    import librosa
    import numpy as np
    
    # STFT parameters
    n_fft = 1024
    hop_length = 256
    
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from first 0.5 seconds
    noise_frames = int(0.5 * sr / hop_length)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction
    clean_magnitude = magnitude - alpha * noise_spectrum
    
    # Prevent over-subtraction
    clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
    
    # Reconstruct signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
    
    return clean_audio

def automatic_gain_control(audio, target_rms=0.1, attack_time=0.01, release_time=0.1, sr=16000):
    """AGC exactly as in Document 14"""
    import numpy as np
    from scipy.signal import medfilt
    
    # Convert time constants to samples
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    # Calculate RMS in overlapping windows
    window_size = 1024
    hop_size = 512
    
    rms_values = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    # Smooth RMS values
    rms_values = np.array(rms_values)
    rms_values = medfilt(rms_values, kernel_size=5)
    
    # Calculate gain adjustments
    gains = []
    current_gain = 1.0
    
    for rms in rms_values:
        if rms > 0:
            target_gain = target_rms / rms
            
            # Apply attack/release
            if target_gain < current_gain:
                # Attack (reduce gain quickly)
                current_gain = current_gain + (target_gain - current_gain) / attack_samples
            else:
                # Release (increase gain slowly)
                current_gain = current_gain + (target_gain - current_gain) / release_samples
        
        gains.append(current_gain)
    
    # Interpolate gains to match audio length
    gain_indices = np.arange(0, len(audio), hop_size)[:len(gains)]
    audio_indices = np.arange(len(audio))
    interpolated_gains = np.interp(audio_indices, gain_indices, gains)
    
    return audio * interpolated_gains

def polish_speech_normalization(audio, sr):
    """Polish-specific audio normalization exactly as in Document 14"""
    import numpy as np
    
    # Dynamic range compression
    threshold = 0.3
    ratio = 3.0
    
    # Find samples above threshold
    above_threshold = np.abs(audio) > threshold
    
    # Apply compression
    compressed = audio.copy()
    compressed[above_threshold] = np.sign(audio[above_threshold]) * (
        threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
    )
    
    # Polish-specific frequency emphasis (slight boost around 1-3kHz for consonants)
    # Create emphasis filter
    frequencies = np.fft.fftfreq(len(audio), 1/sr)
    emphasis_mask = np.ones_like(frequencies)
    
    # Boost 1-3kHz range (Polish consonants)
    boost_range = (np.abs(frequencies) >= 1000) & (np.abs(frequencies) <= 3000)
    emphasis_mask[boost_range] *= 1.2
    
    # Apply frequency domain emphasis
    fft_audio = np.fft.fft(compressed)
    emphasized_fft = fft_audio * emphasis_mask
    emphasized_audio = np.real(np.fft.ifft(emphasized_fft))
    
    return emphasized_audio

def remove_silence_with_padding(audio, voice_mask, padding_ms=150, sr=16000):
    """Remove silence with padding exactly as in Document 14"""
    import numpy as np
    
    padding_samples = int(padding_ms * sr / 1000)
    
    # Extend voice regions with padding
    extended_mask = voice_mask.copy()
    
    # Find voice region boundaries
    voice_changes = np.diff(voice_mask.astype(int))
    voice_starts = np.where(voice_changes == 1)[0] + 1
    voice_ends = np.where(voice_changes == -1)[0] + 1
    
    # Add padding around voice regions
    for start in voice_starts:
        pad_start = max(0, start - padding_samples)
        extended_mask[pad_start:start] = True
    
    for end in voice_ends:
        pad_end = min(len(extended_mask), end + padding_samples)
        extended_mask[end:pad_end] = True
    
    return audio[extended_mask]

# -----------------------------
# STEP 2: Advanced Diarization Functions (from Document 15)
# -----------------------------
def enhanced_voice_activity_detection(audio, sr, frame_duration=20):
    """Enhanced VAD with multiple aggressiveness levels exactly as in Document 15"""
    try:
        import webrtcvad
        import numpy as np
        from scipy.ndimage import binary_dilation, binary_erosion
        
        vad_levels = [1, 2, 3]  # Test multiple aggressiveness levels
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
        
    except ImportError:
        logger.warning("webrtcvad not available for enhanced VAD, using fallback")
        return voice_activity_detection(audio, sr, 30)  # Fallback to step1 VAD

def stereo_separation_enhancement(audio, sr):
    """Stereo separation exactly as in Document 15"""
    import numpy as np
    
    # Apply complementary bandpass filters to create pseudo-stereo effect
    # Lower frequencies (male voices typically)
    low_band = bandpass_filter(audio, sr, 150, 800)
    # Higher frequencies (female voices typically) 
    high_band = bandpass_filter(audio, sr, 800, 3400)
    
    # Create stereo-like separation
    left_channel = 0.7 * audio + 0.3 * low_band
    right_channel = 0.7 * audio + 0.3 * high_band
    
    # Return enhanced mono (average) but with better spectral separation
    return (left_channel + right_channel) / 2

def multi_band_dynamic_range_compression(audio, sr):
    """Multi-band compression exactly as in Document 15"""
    import numpy as np
    
    # Split into frequency bands
    low_band = bandpass_filter(audio, sr, 150, 500)    # Fundamental frequencies
    mid_band = bandpass_filter(audio, sr, 500, 2000)   # Speech formants
    high_band = bandpass_filter(audio, sr, 2000, 3400) # Consonants
    
    def compress_band(band, threshold=0.2, ratio=3.0):
        """Apply compression to a frequency band"""
        compressed = band.copy()
        above_threshold = np.abs(band) > threshold
        compressed[above_threshold] = np.sign(band[above_threshold]) * (
            threshold + (np.abs(band[above_threshold]) - threshold) / ratio
        )
        return compressed
    
    # Apply different compression ratios to each band
    low_compressed = compress_band(low_band, threshold=0.15, ratio=2.5)   # Gentle for low freq
    mid_compressed = compress_band(mid_band, threshold=0.25, ratio=3.0)   # Moderate for mid
    high_compressed = compress_band(high_band, threshold=0.3, ratio=4.0)  # More aggressive for high
    
    # Recombine with emphasis on mid frequencies (speech)
    return 0.3 * low_compressed + 0.5 * mid_compressed + 0.2 * high_compressed

def adaptive_spectral_subtraction(audio, sr, alpha=1.5, beta=0.02):
    """Memory-efficient adaptive spectral subtraction exactly as in Document 15"""
    import librosa
    import numpy as np
    
    chunk_size = sr * 10  # Process 10 seconds at a time
    n_fft = 512  # Smaller FFT size
    hop_length = 256
    
    if len(audio) <= chunk_size:
        # Process small files normally
        return _process_spectral_chunk(audio, sr, alpha, beta, n_fft, hop_length)
    
    # Process large files in chunks
    processed_chunks = []
    overlap = n_fft  # Small overlap between chunks
    
    for start in range(0, len(audio), chunk_size - overlap):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]
        
        processed_chunk = _process_spectral_chunk(chunk, sr, alpha, beta, n_fft, hop_length)
        
        # Remove overlap from all but first chunk
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
        # Fallback to first few frames
        noise_frames = min(5, magnitude.shape[1])
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Simple spectral subtraction (less memory intensive)
    clean_magnitude = magnitude - alpha * noise_spectrum
    clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
    
    # Reconstruct
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft, hop_length=hop_length)
    
    return clean_audio

def speaker_transition_enhancement(audio, sr):
    """Speaker transition enhancement exactly as in Document 15"""
    import librosa
    import numpy as np
    
    # Compute spectral centroid over time
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
    
    # Find rapid changes (potential speaker transitions)
    centroid_diff = np.abs(np.diff(centroids))
    transition_threshold = np.percentile(centroid_diff, 80)  # Top 20% of changes
    
    # Create enhancement mask
    enhancement_mask = np.ones_like(audio)
    
    # For each significant transition, apply local enhancement
    for i, diff in enumerate(centroid_diff):
        if diff > transition_threshold:
            # Convert frame index to sample index
            start_sample = i * 512
            end_sample = min(start_sample + 1024, len(audio))
            
            # Apply mild emphasis around transition
            enhancement_mask[start_sample:end_sample] *= 1.1
    
    return audio * enhancement_mask

def fast_automatic_gain_control(audio, target_rms=0.1, attack_time=0.005, release_time=0.05, sr=16000):
    """Faster AGC exactly as in Document 15"""
    import numpy as np
    from scipy.signal import medfilt
    
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    window_size = 512  # Smaller window for faster response
    hop_size = 256
    
    rms_values = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    # Smooth RMS values with smaller kernel
    rms_values = np.array(rms_values)
    rms_values = medfilt(rms_values, kernel_size=3)
    
    gains = []
    current_gain = 1.0
    
    for rms in rms_values:
        if rms > 0:
            target_gain = target_rms / rms
            target_gain = np.clip(target_gain, 0.1, 10.0)  # Limit gain range
            
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

def speaker_aware_silence_removal(audio, voice_mask, min_gap_ms=200, padding_ms=100, sr=16000):
    """Speaker-aware silence removal exactly as in Document 15"""
    import numpy as np
    
    min_gap_samples = int(min_gap_ms * sr / 1000)
    padding_samples = int(padding_ms * sr / 1000)
    
    # Find voice regions
    voice_changes = np.diff(voice_mask.astype(int))
    voice_starts = np.where(voice_changes == 1)[0] + 1
    voice_ends = np.where(voice_changes == -1)[0] + 1
    
    # Handle edge cases
    if len(voice_starts) == 0 and len(voice_ends) == 0:
        return audio  # No voice detected, return original
    
    if len(voice_starts) > 0 and (len(voice_ends) == 0 or voice_starts[0] < voice_ends[0]):
        voice_ends = np.append(voice_ends, len(voice_mask))
    
    if len(voice_ends) > 0 and (len(voice_starts) == 0 or voice_ends[0] < voice_starts[0]):
        voice_starts = np.insert(voice_starts, 0, 0)
    
    # Create enhanced mask that preserves important silences
    enhanced_mask = np.zeros_like(voice_mask, dtype=bool)
    
    for i in range(len(voice_starts)):
        start = voice_starts[i]
        end = voice_ends[i] if i < len(voice_ends) else len(voice_mask)
        
        # Add padding around voice regions
        pad_start = max(0, start - padding_samples)
        pad_end = min(len(voice_mask), end + padding_samples)
        enhanced_mask[pad_start:pad_end] = True
        
        # If gap to next voice region is small, include it (likely speaker transition)
        if i < len(voice_starts) - 1:
            next_start = voice_starts[i + 1]
            gap_size = next_start - end
            if gap_size < min_gap_samples:
                enhanced_mask[end:next_start] = True
    
    return audio[enhanced_mask]

# -----------------------------
# Processing Pipeline Functions
# -----------------------------
def enhanced_audio_processing_step1(audio, sr):
    """Step 1: Quality improvement processing exactly as Document 14"""
    import librosa
    import numpy as np
    
    logger.info("Step 1: Applying quality improvements (Document 14 pipeline)...")
    
    # Step 1: Voice Activity Detection
    logger.info("  - Applying Voice Activity Detection...")
    voice_mask = voice_activity_detection(audio, sr)
    voice_ratio = np.sum(voice_mask) / len(voice_mask)
    logger.info(f"  - Voice activity: {voice_ratio:.1%} of audio")
    
    # Step 2: Bandpass filtering for speech
    logger.info("  - Applying bandpass filter...")
    audio_filtered = bandpass_filter(audio, sr)
    
    # Step 3: Spectral subtraction for noise reduction
    logger.info("  - Applying spectral subtraction...")
    audio_denoised = spectral_subtraction(audio_filtered, sr)
    
    # Step 4: Automatic Gain Control
    logger.info("  - Applying automatic gain control...")
    audio_agc = automatic_gain_control(audio_denoised, sr=sr)
    
    # Step 5: Polish-specific normalization
    logger.info("  - Applying Polish speech normalization...")
    audio_polish = polish_speech_normalization(audio_agc, sr)
    
    # Step 6: Remove silence with speaker-aware padding
    logger.info("  - Removing silence with padding...")
    audio_final = remove_silence_with_padding(audio_polish, voice_mask, padding_ms=150, sr=sr)
    
    # Final normalization
    audio_final = librosa.util.normalize(audio_final)
    
    # Validate output
    if np.any(np.isnan(audio_final)) or np.any(np.isinf(audio_final)):
        raise ValueError("Step 1 processing produced invalid values")
    
    logger.info("Step 1 complete: Enhanced quality improvement finished")
    return audio_final

def advanced_audio_processing_step2(audio, sr):
    """Step 2: Advanced diarization processing exactly as Document 15"""
    import librosa
    import numpy as np
    
    logger.info("Step 2: Applying advanced diarization enhancements (Document 15 pipeline)...")
    
    # Step 1: Enhanced Voice Activity Detection (on improved audio)
    logger.info("  - Performing enhanced voice activity detection...")
    voice_mask = enhanced_voice_activity_detection(audio, sr)
    voice_ratio = np.sum(voice_mask) / len(voice_mask)
    logger.info(f"  - Voice activity detected: {voice_ratio:.1%} of audio")
    
    # Step 2: Stereo separation enhancement
    logger.info("  - Applying stereo separation enhancement...")
    audio_stereo = stereo_separation_enhancement(audio, sr)
    
    # Step 3: Multi-band compression
    logger.info("  - Applying multi-band dynamic range compression...")
    audio_compressed = multi_band_dynamic_range_compression(audio_stereo, sr)
    
    # Step 4: Adaptive spectral subtraction
    logger.info("  - Applying adaptive spectral subtraction...")
    audio_denoised = adaptive_spectral_subtraction(audio_compressed, sr)
    
    # Step 5: Speaker transition enhancement
    logger.info("  - Enhancing speaker transitions...")
    audio_transitions = speaker_transition_enhancement(audio_denoised, sr)
    
    # Step 6: Fast automatic gain control
    logger.info("  - Applying fast automatic gain control...")
    audio_agc = fast_automatic_gain_control(audio_transitions, sr=sr)
    
    # Step 7: Speaker-aware silence removal
    logger.info("  - Applying speaker-aware silence removal...")
    # Adjust voice mask if audio length changed
    if len(audio_agc) != len(voice_mask):
        logger.info(f"  - Adjusting voice mask: original {len(voice_mask)}, current audio {len(audio_agc)}")
        if len(audio_agc) < len(voice_mask):
            voice_mask = voice_mask[:len(audio_agc)]
        else:
            # Pad with False (no voice)
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
    dag_id='full_advanced_audio_processing',
    schedule_interval='0 */12 * * *',
    start_date=days_ago(1),
    tags=['audio', 'processing', 'full-advanced', 'diarization'],
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
        """Step 1: Full advanced quality improvement processing"""
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
                logger.info(f"Step 1: Processing {file_key} with full Document 14 pipeline")
                obj = s3.get_object(Bucket=input_bucket, Key=file_key)
                audio_bytes = BytesIO(obj['Body'].read())
                original_size = len(audio_bytes.getvalue())
                
                logger.info(f"  - Loading audio file...")
                audio, sr = librosa.load(audio_bytes, sr=16000)
                validate_audio_file(audio, sr)
                
                # Apply Full Step 1 processing (Document 14 pipeline)
                audio_improved = enhanced_audio_processing_step1(audio, sr)
                
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
                        'processing_step': 'step1_full_document14_pipeline',
                        'processing_version': '3.0_full_advanced',
                        'vad_type': 'webrtc_30ms',
                        'spectral_subtraction': 'alpha2.0_beta0.01',
                        'polish_normalization': 'enabled',
                        'frequency_emphasis': '1-3kHz_boost'
                    }
                )
                
                logger.info(f"  - Successfully improved: {output_key} ({improved_size} bytes)")
                
                results.append({
                    "status": "success",
                    "original_size": original_size,
                    "improved_size": improved_size,
                    "sample_rate": sr,
                    "output_key": output_key,
                    "file_key": file_key,
                    "processing_pipeline": "document14_full"
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
        """Step 2: Full advanced diarization optimization processing"""
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
                logger.info(f"Step 2: Processing {file_key} with full Document 15 pipeline")
                obj = s3.get_object(Bucket=input_bucket, Key=improved_key)
                audio_bytes = BytesIO(obj['Body'].read())
                improved_size = len(audio_bytes.getvalue())
                
                logger.info(f"  - Loading improved audio file...")
                audio, sr = librosa.load(audio_bytes, sr=16000)
                validate_audio_file(audio, sr)
                
                # Apply Full Step 2 processing (Document 15 pipeline)
                audio_enhanced = advanced_audio_processing_step2(audio, sr)
                
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
                        'processing_step': 'step2_full_document15_pipeline',
                        'processing_version': '3.0_full_advanced',
                        'vad_type': 'webrtc_multilevel_123_voting',
                        'stereo_separation': 'enabled',
                        'multiband_compression': 'enabled',
                        'speaker_transitions': 'enhanced',
                        'fast_agc': 'enabled'
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
                    "step1_output": improved_key,
                    "processing_pipeline": "document15_full"
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
        """Summarize full advanced processing results"""
        total_files = len(process_results)
        step1_successful = sum(1 for r in process_results if r.get('status') == 'success' and 'improved_size' in r)
        step2_successful = sum(1 for r in process_results if r.get('status') == 'success' and 'enhanced_size' in r)
        skipped = sum(1 for r in process_results if r.get('status') == 'skipped')
        failed = sum(1 for r in process_results if r.get('status') == 'failed')
        
        logger.info(f"Full Advanced Two-Step Processing Summary:")
        logger.info(f"  Total files: {total_files}")
        logger.info(f"  Step 1 (Document 14 Full Pipeline) successful: {step1_successful}")
        logger.info(f"  Step 2 (Document 15 Full Pipeline) successful: {step2_successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")
        
        # Calculate processing improvements
        total_original_size = sum(r.get('original_size', 0) for r in process_results if r.get('original_size'))
        total_enhanced_size = sum(r.get('enhanced_size', 0) for r in process_results if r.get('enhanced_size'))
        
        compression_ratio = None
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
            logger.info(f"Full Advanced processing optimized {len(successful_results)} files with:")
            logger.info(f"  - Complete Document 14 pipeline: WebRTC VAD + Polish normalization + 1-3kHz emphasis")
            logger.info(f"  - Complete Document 15 pipeline: Multi-level VAD + Stereo separation + Speaker transitions")
            logger.info(f"  - Optimized for maximum speaker diarization accuracy")
            logger.info(f"  - Advanced spectral processing and multi-band compression")
            logger.info(f"  - Speaker-aware silence removal with transition preservation")
        
        return {
            "total": total_files,
            "step1_successful": step1_successful,
            "step2_successful": step2_successful,
            "skipped": skipped,
            "failed": failed,
            "compression_ratio": compression_ratio,
            "processing_version": "3.0_full_advanced_identical_to_standalone",
            "step1_pipeline": "document14_complete",
            "step2_pipeline": "document15_complete"
        }

    # -----------------------------
    # DAG Flow
    # -----------------------------
    raw_files = list_raw_files()
    step1_results = process_all_files_step1(raw_files)
    step2_results = process_all_files_step2(step1_results)
    summary = summarize_results(step2_results)
