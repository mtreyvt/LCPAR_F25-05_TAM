"""NewTimeGating
=================

This module implements a simplified time‑gating algorithm inspired by the
automatic calibration procedure described in Bekasiewicz et al. (2018)
for far‑field measurements in non‑anechoic environments.  The classic
approach uses a Tukey window centred on the direct‑path peak of the
impulse response.  The new method implemented here instead derives a
rectangular gate from the envelope of the impulse response itself.  A
threshold is set as a fraction of the peak power and the gate spans
the contiguous region where the average power remains above this
threshold.  The rectangular gate naturally adapts its width to the
dominant arrival and can better reject late reflections when the
measurement environment is highly reflective【905336941599853†L420-L505】.

The functions in this file mirror those provided in ``TimeGating.py``
but use a rectangular window determined by a threshold rather than a
Tukey taper.  They are designed to be lightweight and have no
external dependencies beyond NumPy.  See ``RadioFunctions.do_time_sync_tg_new``
for an example of how to use them in a measurement pipeline.
"""

from __future__ import annotations

import numpy as np


def impulse_response(freq_resp: np.ndarray, N_fft: int) -> np.ndarray:
    """Convert frequency response to impulse response.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape (N_angles, N_freqs).
    N_fft : int
        FFT size to use when zero‑padding before the IFFT.

    Returns
    -------
    np.ndarray
        Real/complex impulse response of shape (N_angles, N_fft).
    """
    return np.fft.ifft(freq_resp, n=N_fft, axis=1)


def _next_pow2(n: int) -> int:
    """Return the next power of two greater than or equal to ``n``."""
    p = 1
    while p < n:
        p <<= 1
        
    return p


def rectangular_gate(h_t: np.ndarray, fs: float, *, gate_ns: float = 10.0,
                     thr_factor: float = 0.5) -> np.ndarray:
    """Apply a rectangular time gate to the impulse response.

    The gate is determined by first computing the average power envelope
    across all angles.  A threshold is defined as ``thr_factor`` times
    the maximum average power.  The gate spans the contiguous region
    where the envelope exceeds this threshold.  If no samples exceed
    the threshold (e.g. due to noise), a fixed window of width
    ``gate_ns`` nanoseconds centred on the strongest peak is used.

    Parameters
    ----------
    h_t : np.ndarray
        Impulse response, shape (N_angles, N_time).
    fs : float
        Sampling rate in Hz used to compute the impulse response.
    gate_ns : float, optional
        Default width of the gate in nanoseconds when no thresholded
        region is found.  This value is converted to a sample count.
    thr_factor : float, optional
        Fraction of the peak average power used as the threshold.  A
        value between 0 and 1.  A typical value of 0.5 captures the
        main lobe around the direct‑path arrival【905336941599853†L420-L505】.

    Returns
    -------
    np.ndarray
        Time‑gated impulse response with the same shape as ``h_t``.
    """
    N_angles, N_time = h_t.shape
    # Compute the average power envelope across angles
    env = np.mean(np.abs(h_t)**2, axis=0)
    max_env = float(np.max(env))
    if not np.isfinite(max_env) or max_env <= 0.0:
        max_env = 1e-12
    threshold = thr_factor * max_env
    # Find contiguous region above threshold
    above = np.where(env >= threshold)[0]
    if above.size > 0:
        start = int(above[0])
        end = int(above[-1]) + 1  # include last sample
    else:
        # Fall back to a fixed window around the strongest peak
        center = int(np.argmax(env))
        gate_len = max(1, int(np.ceil((gate_ns * 1e-9) * fs)))
        start = max(0, center - gate_len // 2)
        end = min(N_time, start + gate_len)
    # Create rectangular window
    gate = np.zeros(N_time, dtype=float)
    gate[start:end] = 1.0
    return h_t * gate[np.newaxis, :]


def gated_frequency_response(h_t_gated: np.ndarray, N_fft: int) -> np.ndarray:
    """FFT back to the frequency domain after time gating."""
    return np.fft.fft(h_t_gated, n=N_fft, axis=1)


def extract_pattern(H_gated: np.ndarray) -> np.ndarray:
    """Extract polar pattern in dB from the gated frequency response.

    Uses the magnitude of the DC bin (bin 0) as the per‑angle value and
    normalises the resulting pattern such that its peak is at 0 dB.
    """
    mags = np.abs(H_gated[:, 0])
    mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
    pattern_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
    if pattern_db.size:
        pattern_db = pattern_db - np.max(pattern_db)
    return pattern_db


def apply_time_gating_matrix_rect(
    freq_resp: np.ndarray,
    freq_list: np.ndarray,
    *,
    gate_width_s: float = 25e-9,
    fs: float | None = None,
    thr_factor: float = 0.5,
    N_fft: int | None = None,
) -> np.ndarray:
    """Convenience wrapper to time‑gate a frequency response using a
    rectangular window.

    The pipeline consists of:

    1. Zero‑pad the frequency response to ``N_fft`` and take the IFFT
       to obtain the impulse response.
    2. Apply a rectangular gate determined either by the threshold
       ``thr_factor`` or by a default width ``gate_width_s``.
    3. FFT back to the frequency domain and extract the DC bin
       magnitude in decibels.
    4. Normalise so that the peak of the pattern is 0 dB.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape (N_angles, N_freqs) containing the
        measured frequency response for each angle.
    freq_list : np.ndarray
        1‑D array of frequency points (Hz) used to construct the
        columns of ``freq_resp``.
    gate_width_s : float, optional
        Default gate width in seconds to use when no thresholded
        region is detected.
    fs : float, optional
        Sampling rate (Hz).  If ``None``, the rate is inferred from the
        frequency spacing.
    thr_factor : float, optional
        Fraction of the peak average power used as the threshold.
    N_fft : int, optional
        FFT size.  If ``None``, the next power of two of twice the
        number of frequency points is chosen.

    Returns
    -------
    np.ndarray
        1‑D array of length ``N_angles`` containing the gated pattern in
        dB, normalised so the peak is at 0 dB.
    """
    freq_resp = np.asarray(freq_resp, dtype=np.complex128)
    if freq_resp.ndim != 2:
        raise ValueError("freq_resp must be a 2D array [N_angles, N_freqs].")
    N_angles, N_freqs = freq_resp.shape
    if N_fft is None:
        # choose a reasonable zero‑padding length; ensure at least twice
        # the number of frequency points to avoid circular overlap
        N_fft = _next_pow2(max(256, 2 * N_freqs))
    # Infer sampling rate from frequency spacing if not provided
    if fs is None:
        f = np.asarray(freq_list, dtype=float).ravel()
        if f.size < 2:
            raise ValueError("Need at least 2 frequency points or provide fs explicitly.")
        df = float(np.median(np.diff(np.sort(f))))
        fs = df * float(N_fft)
    # Time domain response
    h_t = impulse_response(freq_resp, N_fft)
    # Gate using the rectangular window
    h_t_g = rectangular_gate(h_t, fs, gate_ns=gate_width_s * 1e9, thr_factor=thr_factor)
    # Back to frequency domain
    H_g = gated_frequency_response(h_t_g, N_fft)
    # Extract and normalise pattern
    pat_db = extract_pattern(H_g)
    return pat_db