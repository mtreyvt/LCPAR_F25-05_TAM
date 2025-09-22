#------------------------------------------------------------------------------
#'TimeGating.py'                                LCPAR F25-05 VT-ECE
#                                                      20/09/2025
# Desc. of file here... 
#The time-gating function will be utilized to take in the complex data received
#add a gate of at each point that filters out the noise from reflections and then
#gives us cleaner data, This done by setting a window performing some fft/ifft
#and re-sending the data 
#------------------------------------------------------------------------------
from __future__ import annotations
from gnuradio import analog
from gnuradio import blocks
from gnuradio import network
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.filter import window
import matplotlib.pyplot as plt
import numpy as np


import os
import argparse
from typing import Literal, Optional, Tuple

from scipy.signal import windows

#letting wavelet denoise be optional
try: 
    import pywt 
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

#-----setting up time gate

#defining math function for powers of 2 
def next_pow2(n: int) -> int:
    """Return the next power of 2 >= n."""
    return 1 << (int(n - 1).bit_length())

#defining different window functions we could use
def _window_vec(length: int,
                window: Literal['tukey', 'hann', 'hamming', 'rect'] = 'tukey',
                tukey_alpha: float = 0.5) -> np.ndarray:
    if length <= 0:
        raise ValueError("window length must be > 0.")
    if window == 'tukey':
        return windows.tukey(length, alpha=tukey_alpha, sym=False)
    if window == 'hann':
        return windows.hann(length, sym=False)
    if window == 'hamming':
        return windows.hamming(length, sym=False)
    if window == 'rect':
        return np.ones(length, dtype=float)
    raise ValueError(f"Unknown window: {window}")
# --------------------- Time gate core --------------------- #
def build_time_gate(fs: float,
                    Nfft: int,
                    t_start_s: float,
                    t_end_s: float,
                    window: Literal['tukey', 'hann', 'hamming', 'rect'] = 'tukey',
                    tukey_alpha: float = 0.5) -> np.ndarray:
    """
    Build a full-length time-gating vector (length Nfft) with a window placed
    between [t_start_s, t_end_s] (seconds). Python indexing throughout.
    """
    if t_end_s <= t_start_s:
        raise ValueError("t_end_s must be greater than t_start_s.")

    start_idx = int(np.floor(t_start_s * fs))
    end_idx   = int(np.ceil(t_end_s * fs))

    start_idx = max(0, min(start_idx, Nfft - 1))
    end_idx   = max(0, min(end_idx,   Nfft - 1))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, Nfft - 1)

    length = end_idx - start_idx + 1
    w = _window_vec(length, window=window, tukey_alpha=tukey_alpha)

    gate = np.zeros(Nfft, dtype=float)
    gate[start_idx:end_idx + 1] = w
    return gate


def apply_time_gate(H_f: np.ndarray,
                    gate_t: np.ndarray,
                    out_bin: Literal['dc', 'maxmag', 'carrier'] = 'dc',
                    fs: Optional[float] = None,
                    f_c: Optional[float] = None) -> float:
    """
    IFFT -> time gate -> FFT -> return magnitude of selected bin (linear).
    """
    Nfft = H_f.shape[0]
    if gate_t.shape[0] != Nfft:
        raise ValueError("gate_t and H_f must have same length")

    h_t = np.fft.ifft(H_f, n=Nfft)
    h_t_g = h_t * gate_t
    H_clean = np.fft.fft(h_t_g, n=Nfft)
    mag = np.abs(H_clean)

    if out_bin == 'dc':
        return float(mag[0])
    elif out_bin == 'maxmag':
        return float(mag.max())
    elif out_bin == 'carrier':
        if fs is None or f_c is None:
            raise ValueError("fs & f_c required for out_bin='carrier'")
        dk = fs / Nfft
        k = int(round(f_c / dk))
        k = max(0, min(k, Nfft - 1))
        return float(mag[k])
    else:
        raise ValueError(f"unknown out_bin '{out_bin}'.")


def apply_time_gate_sweep(tot_freq_resp: np.ndarray,
                          gate_t: np.ndarray,
                          out_bin: Literal['dc', 'maxmag', 'carrier'] = 'dc',
                          fs: Optional[float] = None,
                          f_c: Optional[float] = None,
                          normalize: bool = True) -> np.ndarray:
    """
    Sweep across angles (rows). Returns dB (optionally normalized to 0 dB peak).
    """
    if tot_freq_resp.ndim != 2:
        raise ValueError("tot_freq_resp must be 2D (angles, fft)")
    n_angles, Nfft = tot_freq_resp.shape
    if gate_t.shape[0] != Nfft:
        raise ValueError("gate_t's length has to match the Nfft")

    mags = np.empty(n_angles, dtype=float)
    for i in range(n_angles):
        mags[i] = apply_time_gate(
            tot_freq_resp[i, :],
            gate_t,
            out_bin=out_bin,
            fs=fs,
            f_c=f_c
        )
    dB = 20.0 * np.log10(np.maximum(mags, np.finfo(float).eps))
    if normalize:
        dB = dB - np.max(dB)
    return dB


# --------------------- Wavelet denoise --------------------- #

def _auto_levels(n: int, name: str) -> int:
    """Heuristic 3..6 levels depending on length."""
    if not _HAS_PYWT:
        return 5
    maxlev = pywt.dwt_max_level(data_len=n, filter_len=pywt.Wavelet(name).dec_len)
    return max(3, min(6, maxlev - 1))


def wavelet_denoise_circ(x_db: np.ndarray,
                         domain: Literal['dB', 'linear'] = 'dB',
                         pad: int = 16,
                         name: str = 'coif5',
                         method: Literal['bayes', 'universal', 'sure'] = 'universal',
                         rule: Literal['soft', 'hard'] = 'soft',
                         levels: Optional[int] = None,
                         cyclespin: int = 8) -> np.ndarray:
    """
    Circular-padding + (optional) cycle-spinning wavelet denoise.
    """
    if not _HAS_PYWT:
        return np.asarray(x_db, dtype=np.complex128).copy()

    x = np.asarray(x_db,dtype=np.complex128).ravel()
    xpad = np.concatenate([x[-pad:], x, x[:pad]]) if pad > 0 else x.copy()

    if domain.lower() == 'linear':
        xwork = np.power(10.0, xpad / 20.0)
    else:
        xwork = xpad

    if levels is None:
        levels = _auto_levels(len(xwork), name)

    def _denoise_1d(arr: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(arr, name, mode='periodization', level=levels)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 + 1e-12

        if method == 'bayes':
            thr = []
            for c in coeffs[1:]:
                var = np.var(c) + 1e-12
                t = sigma**2 / np.sqrt(var)
                thr.append(t)
            coeffs_d = [coeffs[0]]
            for c, t in zip(coeffs[1:], thr):
                coeffs_d.append(pywt.threshold(c, t, mode=rule))
        else:
            if method in ('universal', 'sure'):
                t = sigma * np.sqrt(2.0 * np.log(arr.size))
            else:
                raise ValueError(f"Unknown method '{method}'")
            coeffs_d = [coeffs[0]] + [pywt.threshold(c, t, mode=rule) for c in coeffs[1:]]

        return pywt.waverec(coeffs_d, name, mode='periodization')

    if cyclespin <= 0:
        xden = _denoise_1d(xwork)
    else:
        acc = np.zeros_like(xwork)
        for s in range(cyclespin):
            xs = np.roll(xwork, s)
            xd = _denoise_1d(xs)
            acc += np.roll(xd, -s)
        xden = acc / float(cyclespin)

    if domain.lower() == 'linear':
        xpad_d = 20.0 * np.log10(np.maximum(xden, np.finfo(float).eps))
    else:
        xpad_d = xden

    out = xpad_d[pad:len(xpad_d)-pad] if pad > 0 else xpad_d
    out = out - np.max(out)
    return out


#integration of new time-gating functions

def print_and_return_data(data):
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    plt.plot(arr)
    plt.show()
    return data


def format_data(data, num_freqs):
    a = _np.asarray(data)
    row_len = len(a) // int(num_freqs)
    return _np.array([a[i:i+row_len] for i in range(0, len(a), row_len)])


def synthetic_pulse(frequencies, duration):
    #Old code returned per-frequency qeights
    #instead we FFT time-window of a duration of time at the specific frequency 
    freqs = _np.asarray(frequencies, dtype=np.complex128)
    Nf = len(freqs)
    Nfft = next_pow2(max(1024, 4*Nf))
    # pick a workable fs from the grid (robust fallback if spacing is uneven)
    df_est = _np.median(_np.diff(_np.sort(freqs))) if Nf > 1 else 1.0
    fs = max(2.0*_np.max(freqs), Nfft*max(df_est, 1.0))

    gate_t = build_time_gate(fs=fs, Nfft=Nfft, t_start_s=0.0, t_end_s=float(duration),
                             window='tukey', tukey_alpha=0.5)
    Wf = _np.fft.rfft(gate_t, n=Nfft)
    mag = _np.abs(Wf)

    # map requested freqs to nearest rFFT bins
    df = fs / float(Nfft)
    k = _np.rint(_np.clip(freqs/df, 0, (Nfft//2))).astype(int)
    weights = mag[k]
    m = weights.max() if weights.size else 1.0
    return weights/m if m > 0 else weights
def synthetic_output(pulse, data, num_freqs):
    """Multiply angle×freq matrix by per-frequency weights."""
    A = _np.asarray(data)
    if A.ndim == 1:
        A = format_data(A, num_freqs)
    A = A.astype(_np.complex128, copy=False)
    w = _np.asarray(pulse,dtype=np.complex128).reshape(1, -1)
    if A.shape[1] != w.shape[1]:
        raise ValueError(f"Frequency dimension mismatch: data has {A.shape[1]}, pulse has {w.shape[1]}")
    return A * w

def to_time_domain(data, num_freqs):
    """
    Legacy: IFFT along frequency axis to make time rows per angle.
    Returns angles × time matrix.
    """
    A = _np.asarray(data)
    if A.ndim == 1:
        A = format_data(A, num_freqs)
    A = A.astype(_np.complex128, copy=False)
    td = _np.fft.ifft(A, axis=1)
    try:
        _plt.figure(); _plt.plot(_np.real(td[0,:])); _plt.title("Time-domain (angle 0)"); _plt.grid(True); _plt.show()
    except Exception:
        pass
    return td

