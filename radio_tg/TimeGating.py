# TimeGating.py
import numpy as np
from scipy.signal import tukey, savgol_filter

def impulse_response(freq_resp: np.ndarray, N_fft: int) -> np.ndarray:
    """
    Convert frequency response (angles x freqs) to impulse response (angles x time).
    """
    return np.fft.ifft(freq_resp, n=N_fft, axis=1)

def apply_time_gate(h_t: np.ndarray, fs: float, gate_ns: float = 10.0, alpha: float = 0.5) -> np.ndarray:
    """
    Apply a Tukey window time-gate around the strongest peak across all angles.

    h_t: impulse response [angles, time]
    fs: sample rate (Hz)
    gate_ns: gate width in ns
    alpha: Tukey shape parameter (0=rect, 1=Hann)
    """
    N_angles, N_time = h_t.shape
    t = np.arange(N_time) / fs

    # Find global strongest early peak (direct path)
    power = np.abs(h_t) ** 2
    peak_idx = np.argmax(power[:, :N_time // 4])  # search only early quarter
    gate_center = peak_idx % N_time

    # Gate indices
    gate_len = int(np.ceil(gate_ns * 1e-9 * fs))
    start = max(0, gate_center - gate_len // 2)
    end = min(N_time, start + gate_len)

    # Tukey window
    win = tukey(end - start, alpha=alpha)

    # Apply gate
    g = np.zeros(N_time)
    g[start:end] = win
    h_t_gated = h_t * g[np.newaxis, :]

    return h_t_gated

def gated_frequency_response(h_t_gated: np.ndarray, N_fft: int) -> np.ndarray:
    """
    FFT back to frequency domain after time gating.
    """
    return np.fft.fft(h_t_gated, n=N_fft, axis=1)

def extract_pattern(H_gated: np.ndarray) -> np.ndarray:
    """
    Extract polar pattern in dB from gated frequency response.
    Uses DC bin magnitude (or low freq avg).
    """
    mags = np.abs(H_gated[:, 0])  # take bin 0 magnitude
    mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
    return 20 * np.log10(np.clip(mags, 1e-12, None))

def denoise_pattern(pattern_db: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Smooth jagged gated pattern using Savitzky-Golay filter.
    """
    if len(pattern_db) < window:
        return pattern_db
    return savgol_filter(pattern_db, window, poly)
