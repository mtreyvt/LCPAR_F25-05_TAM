# --- TimeGating.py additions/overhaul ---

from __future__ import annotations
import numpy as np
from typing import Literal, Optional

# optional wavelet denoise
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False


def next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def build_time_gate(fs: float,
                    Nfft: int,
                    t_start_s: float,
                    t_end_s: float,
                    window: Literal['tukey', 'hann', 'hamming', 'rect'] = 'tukey',
                    tukey_alpha: float = 0.5) -> np.ndarray:
    """Create a real time-domain gate (length Nfft) starting at t_start_s and ending at t_end_s."""
    fs = float(fs); Nfft = int(Nfft)
    t_start_s = float(t_start_s); t_end_s = float(t_end_s)
    if t_end_s <= t_start_s:
        raise ValueError("t_end_s must be greater than t_start_s.")
    start_idx = int(np.floor(t_start_s * fs))
    end_idx   = int(np.ceil (t_end_s * fs))
    start_idx = max(0, min(start_idx, Nfft - 1))
    end_idx   = max(0, min(end_idx,   Nfft))
    gate = np.zeros(Nfft, dtype=float)
    length = max(0, end_idx - start_idx)
    if length == 0:
        return gate

    if window == 'tukey':
        # local tukey without SciPy if needed
        a = float(tukey_alpha)
        if a <= 0:
            w = np.ones(length, float)
        elif a >= 1:
            n = np.arange(length)
            w = 0.5 * (1 - np.cos(2 * np.pi * n / (length - 1)))
        else:
            n = np.arange(length)
            edge = int(np.floor(a * (length - 1) / 2.0))
            w = np.ones(length, float)
            if edge > 0:
                n1 = np.arange(edge, dtype=float)
                w[:edge]      = 0.5 * (1 + np.cos(np.pi * (2*n1/(a*(length-1)) - 1)))
                n2 = np.arange(edge, dtype=float)
                w[-edge:]     = 0.5 * (1 + np.cos(np.pi * (2*n2/(a*(length-1)) + 1)))
    elif window == 'hann':
        n = np.arange(length)
        w = 0.5 * (1 - np.cos(2*np.pi*n/(length-1)))
    elif window == 'hamming':
        n = np.arange(length)
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(length-1))
    elif window == 'rect':
        w = np.ones(length, float)
    else:
        raise ValueError(f"Unknown window: {window}")

    gate[start_idx:end_idx] = w
    return gate


def pulse_weights_and_fftlen(freq_list: np.ndarray,
                             gate_width_s: float,
                             *,
                             fs: Optional[float] = None,
                             min_fft: int = 1024) -> tuple[np.ndarray, int, float]:
    """
    Build frequency-domain weights = |FFT{time_gate}| sampled at your freq_list.
    Returns (weights[Nf], Nfft, fs).
    """
    f = np.asarray(freq_list, dtype=float).ravel()
    if f.size < 2:
        # trivial
        Nfft = max(min_fft, 1024)
        fs_eff = float(max(2.0*np.max([1.0, f.max(initial=1.0)]), 10.0))
        gate = build_time_gate(fs_eff, Nfft, 0.0, float(gate_width_s))
        Wf = np.fft.rfft(gate, n=Nfft)
        return np.ones_like(f), Nfft, fs_eff

    f_sorted = np.sort(f)
    df_est = float(np.median(np.diff(f_sorted)))
    # choose a synthetic sampling rate and FFT that comfortably cover your band
    if fs is None:
        fs_eff = max(2.0 * f_sorted.max(), 8.0 * df_est * f.size)  # roomy
    else:
        fs_eff = float(fs)

    Nfft = next_pow2(max(min_fft, 4 * f.size))

    gate = build_time_gate(fs_eff, Nfft, 0.0, float(gate_width_s))
    Wf = np.fft.rfft(gate, n=Nfft)
    # map requested freqs to rfft bins
    df = fs_eff / float(Nfft)
    k = np.rint(np.clip(f / df, 0, Nfft//2)).astype(int)
    weights = np.abs(Wf[k])
    m = weights.max() if weights.size else 1.0
    if m > 0:
        weights = weights / m
    return weights, Nfft, fs_eff


def apply_time_gating_matrix(H_af: np.ndarray,
                             freq_list: np.ndarray,
                             gate_width_s: float,
                             *,
                             pick: Literal['dc', 'max'] = 'dc',
                             denoise_wavelet: bool = True) -> np.ndarray:
    """
    H_af: complex array shape (Nangles, Nf)  — complex response vs freq per angle
    Returns: gated pattern in dB (length Nangles), normalized so peak = 0 dB.
    """
    H = np.asarray(H_af, dtype=np.complex128)
    Nangles, Nf = H.shape
    weights, Nfft, _fs = pulse_weights_and_fftlen(freq_list, gate_width_s)

    if weights.size != Nf:
        # simple linear resample weights to match Nf
        x_old = np.linspace(0.0, 1.0, weights.size)
        x_new = np.linspace(0.0, 1.0, Nf)
        weights = np.interp(x_new, x_old, weights)

    # apply weights across frequency dimension
    Hw = H * weights.reshape(1, -1)

    # zero-pad in IFFT to Nfft for better time resolution
    td = np.fft.ifft(Hw, n=Nfft, axis=1)   # complex time-domain per angle

    if pick == 'dc':
        # equivalent to integrating gated time impulse (after freq weighting)
        y = np.abs(td[:, 0])
    else:
        # use the max magnitude sample in time (robust if alignment is off)
        y = np.abs(td).max(axis=1)

    # normalize and convert to dB
    y = y / (y.max() if y.size else 1.0)
    out_db = 20.0 * np.log10(np.clip(y, 1e-12, None))

    if denoise_wavelet and _HAS_PYWT and Nangles >= 16:
        out_db = wavelet_denoise_circ(out_db, domain='dB', name='coif5',
                                      method='universal', rule='soft',
                                      cyclespin=8)
    return out_db


def wavelet_denoise_circ(x_db: np.ndarray,
                         domain: Literal['dB', 'linear'] = 'dB',
                         pad: int = 16,
                         name: str = 'coif5',
                         method: Literal['bayes', 'universal', 'sure'] = 'universal',
                         rule: Literal['soft', 'hard'] = 'soft',
                         levels: Optional[int] = None,
                         cyclespin: int = 8) -> np.ndarray:
    """Circular-padding + wavelet denoise; returns a trace normalized to 0 dB peak."""
    if not _HAS_PYWT:
        x = np.asarray(x_db, float)
        return x - x.max(initial=0.0)

    x = np.asarray(x_db, float).ravel()
    xpad = np.concatenate([x[-pad:], x, x[:pad]]) if pad > 0 else x.copy()

    if domain == 'linear':
        xwork = np.power(10.0, xpad/20.0)
    else:
        xwork = xpad

    if levels is None:
        maxlev = pywt.dwt_max_level(len(xwork), pywt.Wavelet(name).dec_len)
        levels = max(3, min(6, maxlev - 1))

    def _denoise(arr: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(arr, name, mode='periodization', level=levels)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 + 1e-12
        if method == 'bayes':
            thr = [sigma**2 / np.sqrt(np.var(c) + 1e-12) for c in coeffs[1:]]
            coeffs_d = [coeffs[0]] + [pywt.threshold(c, t, mode=rule) for c, t in zip(coeffs[1:], thr)]
        else:  # universal or sure -> use universal here
            t = sigma * np.sqrt(2.0 * np.log(arr.size))
            coeffs_d = [coeffs[0]] + [pywt.threshold(c, t, mode=rule) for c in coeffs[1:]]
        return pywt.waverec(coeffs_d, name, mode='periodization')

    if cyclespin <= 0:
        xden = _denoise(xwork)
    else:
        acc = np.zeros_like(xwork)
        for s in range(cyclespin):
            xs = np.roll(xwork, s)
            xd = _denoise(xs)
            acc += np.roll(xd, -s)
        xden = acc / float(cyclespin)

    if domain == 'linear':
        xpad_d = 20.0 * np.log10(np.maximum(xden, np.finfo(float).eps))
    else:
        xpad_d = xden

    out = xpad_d[pad:-pad] if pad > 0 else xpad_d
    return out - out.max(initial=0.0)
