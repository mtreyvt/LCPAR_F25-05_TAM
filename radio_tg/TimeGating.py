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

import numpy as np
import typing as _t
import matplotlib.pyplot as plt

from typing import Literal, Optional
from scipy.signal import windows

# Optional wavelet denoise
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False


# ---------- utils ----------
def next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


# ---------- core gate ----------
def build_time_gate(
    fs: _t.SupportsFloat,
    Nfft: _t.SupportsInt,
    t_start_s: _t.SupportsFloat,
    t_end_s: _t.SupportsFloat,
    window: Literal["tukey", "hann", "hamming", "rect"] = "tukey",
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """Return a real time window (length Nfft) with a gate in [t_start_s, t_end_s)."""
    try:
        fs = float(fs)
        Nfft = int(Nfft)
        t_start_s = float(t_start_s)
        t_end_s = float(t_end_s)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"build_time_gate expects numeric scalars: fs={type(fs)}, "
            f"Nfft={type(Nfft)}, t_start_s={type(t_start_s)}, t_end_s={type(t_end_s)}"
        ) from e

    if t_end_s <= t_start_s:
        raise ValueError("t_end_s must be greater than t_start_s.")

    start_idx = int(np.floor(t_start_s * fs))
    end_idx   = int(np.ceil(t_end_s * fs))
    start_idx = max(0, min(start_idx, Nfft))
    end_idx   = max(0, min(end_idx, Nfft))

    gate = np.zeros(Nfft, dtype=float)
    if end_idx > start_idx:
        win_len = end_idx - start_idx
        if window == "tukey":
            w = windows.tukey(win_len, alpha=tukey_alpha, sym=False)
        elif window == "hann":
            w = windows.hann(win_len, sym=False)
        elif window == "hamming":
            w = windows.hamming(win_len, sym=False)
        elif window == "rect":
            w = np.ones(win_len, dtype=float)
        else:
            raise ValueError(f"Unknown window '{window}'")
        gate[start_idx:end_idx] = w
    return gate


def apply_time_gate(
    H_f: np.ndarray,
    gate_t: np.ndarray,
    *,
    normalize: bool = True,
    out_bin: Literal["dc", "carrier", "full"] = "dc",
    fs: float | None = None,
    f_c: float | None = None,
    return_complex: bool = False,
) -> _t.Union[complex, float, np.ndarray]:
    """
    H_f: complex spectrum for one angle (shape [Nf] or [Nfft])
    gate_t: real time window (length Nfft)
    """
    H_f = np.asarray(H_f, dtype=np.complex128)
    Nf = H_f.shape[0]
    Nfft = int(gate_t.shape[0])

    # simple front-embed; replace with a freq→bin mapper if needed
    H_embed = np.zeros(Nfft, dtype=np.complex128)
    m = min(Nf, Nfft)
    H_embed[:m] = H_f[:m]

    h_t = np.fft.ifft(H_embed, n=Nfft)
    h_g = h_t * gate_t
    H_clean = np.fft.fft(h_g, n=Nfft)

    if normalize and H_clean.size:
        peak = np.max(np.abs(H_clean))
        if peak > 0:
            H_clean = H_clean / peak

    if out_bin == "dc":
        val = H_clean[0]
        return val if return_complex else float(np.abs(val))

    if out_bin == "carrier":
        if fs is None or f_c is None:
            raise ValueError("carrier bin requested, but fs or f_c not provided")
        df = fs / float(Nfft)
        k = int(np.rint(f_c / df))
        k = int(np.clip(k, 0, Nfft - 1))
        val = H_clean[k]
        return val if return_complex else float(np.abs(val))

    # "full": return the whole spectrum
    return H_clean if return_complex else np.abs(H_clean)


def apply_time_gate_sweep(
    H_angles_freq: np.ndarray,
    gate_t: np.ndarray,
    *,
    out_bin: Literal["dc", "carrier", "full"] = "dc",
    fs: float | None = None,
    f_c: float | None = None,
    normalize: bool = True,
    return_complex: bool = False,
) -> np.ndarray:
    """Apply the gate for each angle; returns vector (angles) or spectra stack if full."""
    A = np.asarray(H_angles_freq, dtype=np.complex128)
    Nangles = A.shape[0]

    if out_bin == "full":
        # return a stack of spectra (Nangles x Nfft)
        out = np.zeros((Nangles, gate_t.shape[0]),
                       dtype=np.complex128 if return_complex else float)
        for i in range(Nangles):
            out[i] = apply_time_gate(
                A[i, :], gate_t,
                normalize=normalize, out_bin="full",
                fs=fs, f_c=f_c, return_complex=return_complex
            )
        return out

    out = np.zeros(Nangles, dtype=np.complex128 if return_complex else float)
    for i in range(Nangles):
        out[i] = apply_time_gate(
            A[i, :], gate_t,
            normalize=normalize, out_bin=out_bin,
            fs=fs, f_c=f_c, return_complex=return_complex
        )
    return out


# ---------- wavelet denoise (optional) ----------
def _auto_levels(n: int, name: str) -> int:
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
    if not _HAS_PYWT:
        return np.asarray(x_db, dtype=np.complex128).copy()

    x = np.asarray(x_db, dtype=np.complex128).ravel()
    xpad = np.concatenate([x[-pad:], x, x[:pad]]) if pad > 0 else x.copy()

    xwork = 10.0**(xpad / 20.0) if domain.lower() == 'linear' else xpad
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
            coeffs_d = [coeffs[0]] + [pywt.threshold(c, t, mode=rule) for c in coeffs[1:]]
        else:
            if method not in ('universal', 'sure'):
                raise ValueError(f"Unknown method '{method}'")
            t = sigma * np.sqrt(2.0 * np.log(arr.size))
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

    xpad_d = 20.0 * np.log10(np.maximum(xden, np.finfo(float).eps)) if domain.lower() == 'linear' else xden
    out = xpad_d[pad:-pad] if pad > 0 else xpad_d
    out = out - np.max(out)
    return out


# ---------- legacy compatibility shims ----------
def print_and_return_data(data):
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    try:
        plt.figure()
        plt.plot(arr)
        plt.title("Raw data")
        plt.grid(True)
        plt.show()
    except Exception:
        pass
    return data


def format_data(data, num_freqs):
    a = np.asarray(data)
    row_len = len(a) // int(num_freqs)
    return np.array([a[i:i+row_len] for i in range(0, len(a), row_len)])


def synthetic_pulse(frequencies, duration):
    """
    Legacy API: return per-frequency weights.
    Implemented as magnitude of FFT{time-window of 'duration'} sampled at freqs.
    """
    freqs = np.asarray(frequencies, dtype=float)  # must be real
    Nf = len(freqs)
    Nfft = next_pow2(max(1024, 4 * Nf))

    # pick a workable fs from the grid (robust for non-uniform spacing)
    df_est = (np.median(np.diff(np.sort(freqs))) if Nf > 1 else 1.0)
    fs = max(2.0 * float(np.max(freqs) if Nf else 1.0), Nfft * max(df_est, 1.0))

    gate_t = build_time_gate(fs=fs, Nfft=Nfft, t_start_s=0.0, t_end_s=float(duration),
                             window='tukey', tukey_alpha=0.5)
    Wf = np.fft.rfft(gate_t, n=Nfft)
    mag = np.abs(Wf)

    # map requested freqs to nearest rFFT bins
    df = fs / float(Nfft)
    k = np.rint(np.clip(freqs / df, 0, (Nfft // 2))).astype(int)
    weights = mag[k]
    m = weights.max() if weights.size else 1.0
    return weights / m if m > 0 else weights


def synthetic_output(pulse, data, num_freqs):
    A = np.asarray(data)
    if A.ndim == 1:
        A = format_data(A, num_freqs)
    A = A.astype(np.complex128, copy=False)   # keep complex
    w = np.asarray(pulse, dtype=float).reshape(1, -1)
    if A.shape[1] != w.shape[1]:
        raise ValueError(f"Frequency dimension mismatch: data has {A.shape[1]}, pulse has {w.shape[1]}")
    return A * w


def to_time_domain(data, num_freqs):
    A = np.asarray(data)
    if A.ndim == 1:
        A = format_data(A, num_freqs)
    A = A.astype(np.complex128, copy=False)
    td = np.fft.ifft(A, axis=1)  # complex time domain
    return td
