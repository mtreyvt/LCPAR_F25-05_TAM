"""
RadioFunctions
==============

This module implements helper functions for the Low‑Cost Portable
Antenna Range (LCPAR) radio system.  It is based on the open source
implementation from the original project but has been extended to
support a new time‑gating algorithm and time‑synchronised single
frequency scans.  Functions in this file load parameters from JSON,
control the motor via the ``MotorController``, open data files for
logging, perform various measurement routines and plot antenna
patterns.  See the provided documentation for details.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from PlotGraph import PlotGraph
import RxRadio
import TxRadio
import TimeGating
import NewTimeGating
from MotorController import MotorController
from PolarPlot import plot_polar_patterns, plot_patterns


# -----------------------------------------------------------------------------
# Parameter loading and utility functions
# -----------------------------------------------------------------------------

def LoadParams(filename: str | None = None) -> Dict[str, Any]:
    """Load a JSON parameter file.

    Parameters are stored as a dictionary in JSON format.  If a file name
    is not given then the default file ``params_default.json`` is used.
    Any parameters missing from the provided file fall back to the defaults.
    If the file cannot be found or parsing fails an exception is raised.

    Returns
    -------
    dict
        Dictionary of parameters with defaults filled in.
    """
    try:
        defaults = json.load(open("params_default.json"))
    except Exception as e:
        print("params_default.json file is missing")
        raise e
    # No file provided – return the defaults
    if filename is None:
        return defaults
    try:
        params = json.load(open(filename))
    except Exception as e:
        print(f"Failed to load parameter file {filename}")
        raise e
    # Overwrite defaults with values from the provided file
    print(params)
    for p in defaults:
        if p in params:
            defaults[p] = params[p]
        else:
            print(f"Parameter {p} not specified in {filename} using default of ",
                  defaults[p])
    # Check frequency range is within HackRF limits
    if defaults["frequency"] < 30e6 or defaults["frequency"] > 6e9:
        raise Exception(f"Frequency {defaults['frequency']:e} out of range")
    return defaults


def InitMotor(params: Dict[str, Any]) -> MotorController:
    """Initialise and connect the motor controller."""
    motor_controller = MotorController(params["usb_port"], params["baudrate"])
    try:
        motor_controller.connect()
        print("Success: Motor controller fully connected.")
    except Exception as e:
        print("Error: Motor controller not responding, verify connections.")
        raise e
    # Ensure the orientation starts at zero
    motor_controller.reset_orientation()
    return motor_controller


def OpenDatafile(params: Dict[str, Any]):
    """Open a new data file for writing measurement results.

    The filename is based on the current time stamp and the ``filename``
    field from the parameters.  A header is written consisting of the
    notes string and column headings.
    """
    filename = time.strftime("%d-%b-%Y_%H-%M-%S") + params["filename"]
    datafile_fp = open(filename, 'w')
    datafile_fp.write(params["notes"] + "\n")
    datafile_fp.write("% Mast Angle, Arm Angle, Background RSSI, Real, Imag\n")
    return datafile_fp


def rms(data: Iterable[float]) -> float:
    """Return the RMS of a data vector."""
    data = np.asarray(data, dtype=float)
    return float(np.sqrt(np.square(data).mean()))


def round_sig(x: float, sig: int = 3) -> float:
    """Round a floating point value to a given number of significant digits.

    If the value is not finite or zero it is returned unchanged.
    """
    if not np.isfinite(x) or x == 0.0:
        return 0.0 if x == 0.0 else x
    ndigits = int(sig - 1 - np.floor(np.log10(abs(x))))
    ndigits = max(-12, min(12, ndigits))
    return float(round(x, ndigits))


# -----------------------------------------------------------------------------
# Basic measurement routines
# -----------------------------------------------------------------------------

def do_single(Tx: bool = True) -> float:
    """Capture a single measurement either with the transmitter on or off.

    When ``Tx`` is True both the transmitter and receiver SDR flow graphs are
    started.  The receiver collects a fixed number of samples then stops and
    the RMS of the received signal is returned.  When ``Tx`` is False only
    the receiver is started.
    """
    params = LoadParams()
    # Construct the radio flow graphs
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], params["frequency"], params["tx_freq_offset"]
    )
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], params["frequency"], params["rx_freq_offset"],
        numSamples=10000
    )
    # Start the graphs
    if Tx:
        radio_tx_graph.start()
    radio_rx_graph.start()
    radio_rx_graph.wait()
    if Tx:
        radio_tx_graph.stop()
    # Extract samples and compute RMS
    rxd = np.asarray(radio_rx_graph.vector_sink_0.data(), dtype=float)
    # Plot for debugging
    plt.plot(rxd)
    plt.show()
    return rms(rxd)


def do_singleTG(params: Dict[str, Any]):
    """Simple bench test: sweep frequencies at a fixed pointing and time gate.

    This function sweeps from ``lower_frequency`` to ``upper_frequency`` in
    ``freq_steps`` steps at a single pointing.  The complex response at each
    frequency is averaged, a time gate is applied using the Tukey method in
    ``TimeGating.apply_time_gating_matrix``, and the resulting pattern in dB
    is printed and returned.
    """
    freq_lin = np.linspace(params["lower_frequency"],
                           params["upper_frequency"],
                           params["freq_steps"])
    freq_list = np.unique(np.array([round_sig(v, 3) for v in freq_lin],
                                   dtype=float))
    Nf = int(freq_list.size)
    if Nf < 2:
        print("Increase freq_steps/span: need ≥2 frequency points for time gating.")
        return None
    # Collect complex mean per frequency (static antenna)
    resp = np.zeros((1, Nf), dtype=np.complex128)
    for i, f in enumerate(freq_list):
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], f, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], f, params["tx_freq_offset"])
        tx.start()
        time.sleep(1.5)
        rx.start()
        rx.wait()
        I = np.array(rx.vector_sink_0.data(), dtype=float)
        Q = np.array(rx.vector_sink_1.data(), dtype=float)
        L = min(I.size, Q.size)
        resp[0, i] = (I[:L] + 1j * Q[:L]).mean() if L else 0.0 + 0.0j
        try:
            rx.stop()
            rx.wait()
        except Exception:
            pass
        try:
            tx.stop()
            tx.wait()
        except Exception:
            pass
    # Apply time gating to the 1×Nf sweep (fs inferred from df)
    gate_width = float(params.get("tg_duration_s", 25e-9))
    tg_db = TimeGating.apply_time_gating_matrix(resp, freq_list,
                                                gate_width_s=gate_width)
    print("Time‑gated (dB, peak=0):", tg_db.tolist())
    return tg_db


# -----------------------------------------------------------------------------
# Fast scan and stepwise AM measurement routines
# -----------------------------------------------------------------------------

def do_AMscan(params: Dict[str, Any]):
    """Perform a coherent AM fast scan over the mast angle range.

    This function rotates the mast from ``mast_start_angle`` to
    ``mast_end_angle`` while the transmitter is on and the receiver is
    streaming.  The raw samples are binned into equal angular bins and
    optionally time gated if ``params['time_gate']`` is True.
    """
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    time_gating_enabled = params.get("time_gate", False)
    print("Time gating enabled: " + str(time_gating_enabled))
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], params["frequency"], params["tx_freq_offset"]
    )
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], params["frequency"], params["rx_freq_offset"]
    )
    AMantenna_data: List[Tuple[float, float, float, complex]] = []
    # Start TX and let it settle
    radio_tx_graph.start()
    time.sleep(3)
    # Move to start angle
    print("Moving to start angle")
    motor_controller.rotate_mast(params["mast_start_angle"])
    print("Collecting data while moving to end angle")
    radio_rx_graph.start()
    motor_controller.rotate_mast(params["mast_end_angle"])
    radio_rx_graph.stop()
    radio_tx_graph.stop()
    # Reset mast orientation
    print("Finished collection, return to 0")
    motor_controller.rotate_mast(0)
    antenna_data = np.array(radio_rx_graph.vector_sink_0.data(), dtype=float)
    # Apply optional time gating
    if time_gating_enabled:
        antenna_data = TimeGating.print_and_return_data(antenna_data)
    n = len(antenna_data)
    print(f"read {n:d} data points")
    antenna_pow = np.square(antenna_data)
    numangles = int(params["mast_end_angle"] - params["mast_start_angle"])
    binsize = int(n / numangles) if numangles > 0 else 1
    print(f"binsize= {binsize:d}")
    avg = np.zeros(numangles)
    for i in range(numangles):
        avg[i] = np.sqrt(np.square(
            antenna_data[i * binsize:(i + 1) * binsize]).sum() / binsize)
    angles = range(int(params["mast_start_angle"]),
                   int(params["mast_end_angle"]), 1)
    arm_angle = np.zeros(len(avg))
    background_rssi = np.zeros(len(avg))
    # Plot for debugging
    plt.plot(antenna_pow)
    plt.show()
    plt.plot(avg)
    plt.show()
    print(f"avg {len(avg):d}", binsize)
    for i, a in enumerate(avg):
        datafile.write(f"{angles[i]},0.0,0.0,{a:.8e},0.0\n")
    datafile.close()
    return list(zip(angles, arm_angle.tolist(),
                    background_rssi.tolist(), avg.tolist()))


def do_AMmeas(params: Dict[str, Any]):
    """Perform a coherent AM measurement with stepwise rotation.

    At each mast angle in ``mast_steps`` the receiver samples while the
    transmitter is on.  The background and transmitted RSSI values are
    recorded and returned.
    """
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], params["frequency"], params["tx_freq_offset"]
    )
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], params["frequency"], params["rx_freq_offset"]
    )
    antenna_data: List[Tuple[float, float, float, float]] = []
    mast_angles = np.linspace(
        params["mast_start_angle"], params["mast_end_angle"], params["mast_steps"]
    )
    arm_angles = np.linspace(
        params["arm_start_angle"], params["arm_end_angle"], params["arm_steps"]
    )
    for mast_angle in mast_angles:
        # azimuth loop
        for arm_angle in arm_angles:
            # elevation loop
            background_rssi = 0.0
            transmission_rssi = 0.0
            print(f"Target Mast Angle: {mast_angle}")
            print(f"Target Arm Angle: {arm_angle}")
            print("Moving antenna...")
            motor_controller.rotate_mast(mast_angle)
            motor_controller.rotate_arm(arm_angle)
            print("Movement complete")
            print("Taking background noise sample...")
            # Background RSSI
            radio_rx_graph.start()
            radio_rx_graph.wait()
            data = np.asarray(radio_rx_graph.vector_sink_0.data(), dtype=float)
            print(f"received {len(data)} background samples")
            radio_rx_graph.vector_sink_0.reset()
            radio_rx_graph.blocks_head_0.reset()
            background_rssi = rms(data)
            print("Taking transmitted signal sample...")
            radio_tx_graph.start()
            time.sleep(1.3)
            radio_rx_graph.start()
            radio_rx_graph.wait()
            radio_tx_graph.stop()
            radio_tx_graph.wait()
            data = np.asarray(radio_rx_graph.vector_sink_0.data(), dtype=float)
            print(f"received {len(data)} transmitted samples")
            radio_rx_graph.vector_sink_0.reset()
            radio_rx_graph.blocks_head_0.reset()
            transmission_rssi = rms(data)
            # Write RSSI readings to file
            print("Saving samples")
            datafile.write(
                f"{mast_angle},{arm_angle},{background_rssi},{transmission_rssi}\n"
            )
            print(f"Sample angle={mast_angle:f} bkgnd={background_rssi:e} "
                  f"received={transmission_rssi:e}")
            antenna_data.append(
                (mast_angle, arm_angle, background_rssi, transmission_rssi)
            )
            # Return to home position
            print("Returning mast and arm to home position...")
            motor_controller.rotate_mast(0)
            motor_controller.rotate_arm(0)
            print("Mast and arm should now be in home position")
    datafile.close()
    return antenna_data


# -----------------------------------------------------------------------------
# Time‑gated AM routines
# -----------------------------------------------------------------------------

def do_AMTGmeas(params: Dict[str, Any]):
    """Stepwise AM measurement with time gating applied after the sweep.

    This routine performs a frequency sweep at each mast angle and applies
    a Tukey window time gate in the post processing.  The time gated
    pattern is plotted and returned.
    """
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    # Build frequency list (rounded and deduplicated)
    freq_lin = np.linspace(
        params["lower_frequency"], params["upper_frequency"], params["freq_steps"]
    )
    freq_list = np.unique([round_sig(f, 3) for f in freq_lin])
    Nf = int(freq_list.size)
    if Nf < 2:
        print("WARNING: time gating needs multiple frequencies. Increase freq_steps or span.")
        print(f"Using {Nf} frequency points.")
    # Angle grid
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    # Storage: complex response per angle per freq
    responses = np.zeros((mast_steps, Nf), dtype=np.complex128)
    # Sweep
    for fi, freq in enumerate(freq_list):
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        tx.start()
        time.sleep(3)
        print(f"[{fi+1}/{Nf}] freq={freq/1e6:.3f} MHz: rotating & collecting …")
        per_angle = _collect_complex_per_angle(rx, mast_start, mast_end, mast_steps, motor_controller)
        responses[:, fi] = per_angle
        tx.stop()
        tx.wait()
        motor_controller.rotate_mast(0)
        # Log raw complex mean to the open file
        for ang, cval in zip(mast_angles, per_angle):
            datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    # Time gating
    gate_width = float(params.get("tg_duration_s", 25e-9))
    print(f"Applying time gating with width {gate_width*1e9:.1f} ns …")
    gated_db = TimeGating.apply_time_gating_matrix(
        responses, freq_list, gate_width_s=gate_width, denoise_wavelet=True
    )
    # Plot polar + cartesian
    plot_polar_patterns(
        mast_angles,
        traces=[("Time‑Gated", gated_db)],
        rmin=-60.0,
        rmax=0.0,
        rticks=(-60, -40, -20, 0),
        title="Radiation Pattern (Time‑Gated, Polar)"
    )
    try:
        plot_patterns(
            mast_angles,
            traces=[("Time‑Gated", gated_db)],
            title="Radiation Pattern (Time‑Gated)"
        )
    except Exception:
        pass
    # Write gated results (linear magnitude, normalised)
    y_lin = 10.0 ** (gated_db / 20.0)
    outname = time.strftime("%d-%b-%Y_%H-%M-%S") + "_TG.csv"
    with open(outname, "w") as fp:
        fp.write("Angle,Arm,Background,GatedMag\n")
        for a, v in zip(mast_angles, y_lin):
            fp.write(f"{a:.1f},0.0,0.0,{v:.8e}\n")
    print(f"wrote gated results → {outname}")
    return list(zip(mast_angles.tolist(), [0.0] * len(mast_angles), [0.0] * len(mast_angles), y_lin.tolist()))


def do_AMTGscan(params: Dict[str, Any]):
    """Perform a fast rotating scan over frequency with time gating.

    This routine rotates the mast continuously while sweeping over a set of
    frequencies.  After collecting the complex responses a Tukey window
    time gate is applied.  The gated pattern is plotted and returned.
    """
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    # Build frequency list
    freq_lin = np.linspace(
        params["lower_frequency"], params["upper_frequency"], params["freq_steps"]
    )
    freq_list = np.unique([round_sig(v, 3) for v in freq_lin])
    Nf = int(freq_list.size)
    if Nf < 2:
        print("WARNING: time gating needs multiple frequencies. Increase freq_steps or span.")
        print(f"Using {Nf} frequency points.")
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    responses = np.zeros((mast_steps, Nf), dtype=np.complex128)
    # Sweep
    for fi, freq in enumerate(freq_list):
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        tx.start()
        time.sleep(3)
        print(f"[{fi+1}/{Nf}] freq={freq/1e6:.3f} MHz: rotating & collecting …")
        per_angle = _collect_complex_per_angle(rx, mast_start, mast_end, mast_steps, motor_controller)
        responses[:, fi] = per_angle
        tx.stop()
        tx.wait()
        motor_controller.rotate_mast(0)
        # Log raw complex mean to the open file
        for ang, cval in zip(mast_angles, per_angle):
            datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    gate_width = float(params.get("tg_duration_s", 25e-9))
    print(f"Applying time gating with width {gate_width*1e9:.1f} ns …")
    gated_db = TimeGating.apply_time_gating_matrix(
        responses, freq_list, gate_width_s=gate_width, denoise_wavelet=True
    )
    # Plot
    plot_polar_patterns(
        mast_angles,
        traces=[("Time‑Gated", gated_db)],
        rmin=-60.0,
        rmax=0.0,
        rticks=(-60, -40, -20, 0),
        title="Radiation Pattern (Time‑Gated, Polar)"
    )
    try:
        plot_patterns(
            mast_angles,
            traces=[("Time‑Gated", gated_db)],
            title="Radiation Pattern (Time‑Gated)"
        )
    except Exception:
        pass
    # Write gated results
    y_lin = 10.0 ** (gated_db / 20.0)
    outname = time.strftime("%d-%b-%Y_%H-%M-%S") + "_TG.csv"
    with open(outname, "w") as fp:
        fp.write("Angle,Arm,Background,GatedMag\n")
        for a, v in zip(mast_angles, y_lin):
            fp.write(f"{a:.1f},0.0,0.0,{v:.8e}\n")
    print(f"wrote gated results → {outname}")
    return list(zip(mast_angles.tolist(), [0.0] * len(mast_angles), [0.0] * len(mast_angles), y_lin.tolist()))


def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Compute a moving RMS envelope of the input complex array."""
    if win <= 1:
        return np.abs(x)
    mag2 = np.abs(x) ** 2
    ker = np.ones(win, dtype=float) / float(win)
    return np.sqrt(np.convolve(mag2, ker, mode="same"))


def _find_tx_on_edge(x: np.ndarray, win: int = 256, k_sigma: float = 6.0) -> int:
    """Detect the transmitter turn‑on edge in a complex sample stream.

    The RMS envelope is computed with a window of length ``win`` and a
    threshold of ``k_sigma`` standard deviations above the mean of the
    first 10% of the envelope.  The first index exceeding the threshold
    is returned.  If no such point exists then the index of the
    maximum envelope value is returned instead.
    """
    env = _moving_rms(x, win)
    n0 = max(win, int(0.10 * len(env)))
    base = env[:n0]
    mu, sigma = float(base.mean()), float(base.std() + 1e-12)
    thr = mu + k_sigma * sigma
    idx = np.where(env > thr)[0]
    return int(idx[0]) if idx.size else int(np.argmax(env))


def _collect_complex_per_angle(
    rx_graph, mast_start: float, mast_end: float, mast_steps: int, motor_controller
) -> np.ndarray:
    """Start RX, rotate mast from start->end, stop RX and return per‑angle means.

    RX is started and then the mast is rotated from ``mast_start`` to
    ``mast_end``.  Samples are collected continuously.  After stopping
    the rotation and RX, the complex samples are uniformly binned into
    ``mast_steps`` bins and the mean of each bin is returned.
    """
    mast_start_f = float(mast_start)
    mast_end_f = float(mast_end)
    mast_steps_i = int(mast_steps)
    # Move to start angle and start RX
    motor_controller.rotate_mast(mast_start_f)
    rx_graph.start()
    # Rotate to end angle
    motor_controller.rotate_mast(mast_end_f)
    # Stop RX
    rx_graph.stop()
    rx_graph.wait()
    # Fetch raw complex samples
    I = np.array(rx_graph.vector_sink_0.data(), dtype=float)
    Q = np.array(rx_graph.vector_sink_1.data(), dtype=float)
    L = int(min(I.size, Q.size))
    samps = I[:L] + 1j * Q[:L] if L else np.zeros(0, dtype=np.complex128)
    if L == 0:
        return np.zeros(mast_steps_i, dtype=np.complex128)
    # Uniformly bin the samples into mast_steps bins
    idx = np.linspace(0, len(samps), mast_steps_i + 1, dtype=int)
    per_angle = np.zeros(mast_steps_i, dtype=np.complex128)
    for i in range(mast_steps_i):
        s, e = idx[i], idx[i + 1]
        seg = samps[s:e] if e > s else samps[s:s + 1]
        per_angle[i] = seg.mean()
    return per_angle


def do_AMTGscan_single_freq(
    params: Dict[str, Any],
    freq_hz: float = 5.6e9,
    *,
    show_plots: bool = True,
    pre_tx_delay_s: float = 0.25,
    edge_win: int = 256,
    edge_k_sigma: float = 6.0,
) -> Dict[str, Any]:
    """Perform a time‑synchronised single‑frequency scan with Tukey gating.

    This function captures a complex response for each mast angle at a
    single frequency.  A simple software time‑synchronisation sequence
    is used: RX starts first to capture pre‑roll, then after a short
    delay TX starts.  The transmitter turn‑on edge is detected in the
    complex stream so the samples can be aligned.  The aligned samples
    are binned into ``mast_steps`` bins.  A Tukey window time gate is
    applied to the impulse response and the resulting pattern is
    returned along with the raw noisy magnitude.
    """
    import numpy as _np
    # Motor and data file setup
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    freq = float(round_sig(freq_hz, 3))
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = _np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    rx = tx = None
    per_angle_complex = None
    try:
        # Prepare graphs and move to start angle
        motor_controller.rotate_mast(mast_start)
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        # 1) Start RX first to get pre‑roll
        rx.start()
        # 2) After short delay start TX so we can detect the TX‑on edge
        time.sleep(max(0.05, float(pre_tx_delay_s)))
        tx.start()
        # 3) Begin rotation while RX is running
        print(f"Single‑frequency scan @ {freq/1e9:.3f} GHz: rotating & collecting …")
        motor_controller.rotate_mast(mast_end)
        # 4) Stop RX and TX once rotation completes
        rx.stop()
        rx.wait()
        tx.stop()
        tx.wait()
        # Fetch raw complex stream
        I = _np.array(rx.vector_sink_0.data(), dtype=float)
        Q = _np.array(rx.vector_sink_1.data(), dtype=float)
        L = int(min(I.size, Q.size))
        if L == 0:
            print("No samples received.")
            return {
                "angles_deg": mast_angles.tolist(),
                "complex_per_angle": [],
                "noisy_db": [],
                "tg_db": [],
            }
        samps = I[:L] + 1j * Q[:L]
        # Detect TX‑on edge in the stream and trim
        i0 = _find_tx_on_edge(samps, win=int(edge_win), k_sigma=float(edge_k_sigma))
        if (L - i0) >= max(1024, mast_steps * 8):
            samps = samps[i0:]
        else:
            print("WARNING: short post‑edge segment; using full buffer.")
        # Bin the aligned samples uniformly into mast_steps means
        idx = _np.linspace(0, len(samps), mast_steps + 1, dtype=int)
        per_angle = _np.zeros(mast_steps, dtype=np.complex128)
        for i in range(mast_steps):
            s, e = idx[i], idx[i + 1]
            seg = samps[s:e] if e > s else samps[s:s + 1]
            per_angle[i] = seg.mean()
        per_angle_complex = per_angle
    finally:
        try:
            if rx:
                rx.stop()
                rx.wait()
        except Exception:
            pass
        try:
            if tx:
                tx.stop()
                tx.wait()
        except Exception:
            pass
        try:
            motor_controller.rotate_mast(0)
        except Exception:
            pass
    # Write raw complex per‑angle data
    for ang, cval in zip(mast_angles, per_angle_complex):
        datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    # Build patterns
    noisy_mag = _np.abs(per_angle_complex)
    noisy_mag /= (_np.max(noisy_mag) if _np.max(noisy_mag) > 0 else 1.0)
    noisy_db = 20 * _np.log10(_np.clip(noisy_mag, 1e-12, None))
    # With a single frequency column, the time‑gating block is largely a no‑op,
    # but we keep the call structure for consistency/plotting.
    freq_resp = per_angle_complex[:, _np.newaxis]
    fs = float(params.get("fs", 200e6))
    N_fft = 2048
    h_t = TimeGating.impulse_response(freq_resp, N_fft)
    h_t_gated = TimeGating.apply_time_gate(h_t, fs, gate_ns=3.75, alpha=0.4)
    H_gated = TimeGating.gated_frequency_response(h_t_gated, N_fft)
    gated_pattern = TimeGating.extract_pattern(H_gated)
    gated_pattern = TimeGating.denoise_pattern(gated_pattern)
    gated_pattern -= _np.max(gated_pattern)
    if show_plots:
        plot_polar_patterns(
            mast_angles,
            traces=[
                ("Raw (noisy)", noisy_db),
                ("Time‑Gated", gated_pattern)
            ],
            rmin=-60,
            rmax=0,
            rticks=(-60, -40, -20, 0),
            title=f"Noisy vs Time‑Gated Pattern @ {freq/1e9:.2f} GHz"
        )
    return {
        "angles_deg": mast_angles.tolist(),
        "complex_per_angle": per_angle_complex.tolist(),
        "noisy_db": noisy_db.tolist(),
        "tg_db": gated_pattern.tolist(),
    }


# -----------------------------------------------------------------------------
# New time‑synchronised routines without and with different gating methods
# -----------------------------------------------------------------------------

def do_time_sync_no_tg(
    params: Dict[str, Any],
    freq_hz: float = 5.6e9,
    *,
    show_plots: bool = True,
    pre_tx_delay_s: float = 0.25,
    edge_win: int = 256,
    edge_k_sigma: float = 6.0,
) -> Dict[str, Any]:
    """Perform a time‑synchronised single‑frequency scan without any time gating.

    This routine is similar to ``do_AMTGscan_single_freq`` but the post‑processing
    does not apply a time gate.  The raw complex per‑angle samples are
    binned, written to file and the resulting magnitude pattern is
    normalised and returned.
    """
    import numpy as _np
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    freq = float(round_sig(freq_hz, 3))
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = _np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    rx = tx = None
    per_angle_complex = None
    try:
        motor_controller.rotate_mast(mast_start)
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        rx.start()
        time.sleep(max(0.05, float(pre_tx_delay_s)))
        tx.start()
        print(f"Time‑synchronised scan (no gating) @ {freq/1e9:.3f} GHz: rotating & collecting …")
        motor_controller.rotate_mast(mast_end)
        rx.stop()
        rx.wait()
        tx.stop()
        tx.wait()
        I = _np.array(rx.vector_sink_0.data(), dtype=float)
        Q = _np.array(rx.vector_sink_1.data(), dtype=float)
        L = int(min(I.size, Q.size))
        if L == 0:
            print("No samples received.")
            return {
                "angles_deg": mast_angles.tolist(),
                "complex_per_angle": [],
                "noisy_db": [],
            }
        samps = I[:L] + 1j * Q[:L]
        i0 = _find_tx_on_edge(samps, win=int(edge_win), k_sigma=float(edge_k_sigma))
        if (L - i0) >= max(1024, mast_steps * 8):
            samps = samps[i0:]
        else:
            print("WARNING: short post‑edge segment; using full buffer.")
        idx = _np.linspace(0, len(samps), mast_steps + 1, dtype=int)
        per_angle = _np.zeros(mast_steps, dtype=np.complex128)
        for i in range(mast_steps):
            s, e = idx[i], idx[i + 1]
            seg = samps[s:e] if e > s else samps[s:s + 1]
            per_angle[i] = seg.mean()
        per_angle_complex = per_angle
    finally:
        try:
            if rx:
                rx.stop()
                rx.wait()
        except Exception:
            pass
        try:
            if tx:
                tx.stop()
                tx.wait()
        except Exception:
            pass
        try:
            motor_controller.rotate_mast(0)
        except Exception:
            pass
    for ang, cval in zip(mast_angles, per_angle_complex):
        datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    noisy_mag = _np.abs(per_angle_complex)
    noisy_mag /= (_np.max(noisy_mag) if _np.max(noisy_mag) > 0 else 1.0)
    noisy_db = 20 * _np.log10(_np.clip(noisy_mag, 1e-12, None))
    if show_plots:
        plot_polar_patterns(
            mast_angles,
            traces=[("No gating (raw)", noisy_db)],
            rmin=-60, rmax=0,
            rticks=(-60, -40, -20, 0),
            title=f"No gating pattern @ {freq/1e9:.2f} GHz"
        )
    return {
        "angles_deg": mast_angles.tolist(),
        "complex_per_angle": per_angle_complex.tolist(),
        "noisy_db": noisy_db.tolist(),
    }


def do_time_sync_tg_old(
    params: Dict[str, Any],
    freq_hz: float = 5.6e9,
    *,
    show_plots: bool = True,
    pre_tx_delay_s: float = 0.25,
    edge_win: int = 256,
    edge_k_sigma: float = 6.0,
    gate_ns: float = 3.75,
    alpha: float = 0.4,
) -> Dict[str, Any]:
    """Perform a time‑synchronised single‑frequency scan with Tukey window gating.

    This function follows the same acquisition procedure as
    ``do_time_sync_no_tg`` but applies the legacy Tukey time gate from
    ``TimeGating.py``.  The gated pattern is normalised and returned.
    """
    import numpy as _np
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    freq = float(round_sig(freq_hz, 3))
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = _np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    rx = tx = None
    per_angle_complex = None
    try:
        motor_controller.rotate_mast(mast_start)
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        rx.start()
        time.sleep(max(0.05, float(pre_tx_delay_s)))
        tx.start()
        print(f"Time‑synchronised scan (old TG) @ {freq/1e9:.3f} GHz: rotating & collecting …")
        motor_controller.rotate_mast(mast_end)
        rx.stop()
        rx.wait()
        tx.stop()
        tx.wait()
        I = _np.array(rx.vector_sink_0.data(), dtype=float)
        Q = _np.array(rx.vector_sink_1.data(), dtype=float)
        L = int(min(I.size, Q.size))
        if L == 0:
            print("No samples received.")
            return {
                "angles_deg": mast_angles.tolist(),
                "complex_per_angle": [],
                "noisy_db": [],
                "tg_db_old": [],
            }
        samps = I[:L] + 1j * Q[:L]
        i0 = _find_tx_on_edge(samps, win=int(edge_win), k_sigma=float(edge_k_sigma))
        if (L - i0) >= max(1024, mast_steps * 8):
            samps = samps[i0:]
        else:
            print("WARNING: short post‑edge segment; using full buffer.")
        idx = _np.linspace(0, len(samps), mast_steps + 1, dtype=int)
        per_angle = _np.zeros(mast_steps, dtype=np.complex128)
        for i in range(mast_steps):
            s, e = idx[i], idx[i + 1]
            seg = samps[s:e] if e > s else samps[s:s + 1]
            per_angle[i] = seg.mean()
        per_angle_complex = per_angle
    finally:
        try:
            if rx:
                rx.stop()
                rx.wait()
        except Exception:
            pass
        try:
            if tx:
                tx.stop()
                tx.wait()
        except Exception:
            pass
        try:
            motor_controller.rotate_mast(0)
        except Exception:
            pass
    for ang, cval in zip(mast_angles, per_angle_complex):
        datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    noisy_mag = _np.abs(per_angle_complex)
    noisy_mag /= (_np.max(noisy_mag) if _np.max(noisy_mag) > 0 else 1.0)
    noisy_db = 20 * _np.log10(_np.clip(noisy_mag, 1e-12, None))
    # Apply legacy Tukey window gating
    freq_resp = per_angle_complex[:, _np.newaxis]
    fs = float(params.get("fs", 200e6))
    N_fft = 2048
    h_t = TimeGating.impulse_response(freq_resp, N_fft)
    h_t_gated = TimeGating.apply_time_gate(h_t, fs, gate_ns=float(gate_ns), alpha=float(alpha))
    H_gated = TimeGating.gated_frequency_response(h_t_gated, N_fft)
    gated_pattern = TimeGating.extract_pattern(H_gated)
    gated_pattern = TimeGating.denoise_pattern(gated_pattern)
    gated_pattern -= _np.max(gated_pattern)
    if show_plots:
        plot_polar_patterns(
            mast_angles,
            traces=[
                ("Raw (noisy)", noisy_db),
                ("Tukey‑Gated", gated_pattern)
            ],
            rmin=-60,
            rmax=0,
            rticks=(-60, -40, -20, 0),
            title=f"Old Tukey Gating Pattern @ {freq/1e9:.2f} GHz"
        )
    return {
        "angles_deg": mast_angles.tolist(),
        "complex_per_angle": per_angle_complex.tolist(),
        "noisy_db": noisy_db.tolist(),
        "tg_db_old": gated_pattern.tolist(),
    }


def do_time_sync_tg_new(
    params: Dict[str, Any],
    freq_hz: float = 5.6e9,
    *,
    show_plots: bool = True,
    pre_tx_delay_s: float = 0.25,
    edge_win: int = 256,
    edge_k_sigma: float = 6.0,
    thr_factor: float = 0.5,
    gate_ns: float = 3.75,
) -> Dict[str, Any]:
    """Perform a time‑synchronised single‑frequency scan with rectangular gating.

    This routine is identical to ``do_time_sync_no_tg`` except that a
    rectangular time gate determined by the power envelope of the impulse
    response is applied.  The gate width is automatically derived using
    ``thr_factor`` of the peak; if no thresholded region is found a
    fallback window of ``gate_ns`` nanoseconds is used.  The resulting
    pattern is normalised and returned.
    """
    import numpy as _np
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params)
    freq = float(round_sig(freq_hz, 3))
    mast_start = float(params["mast_start_angle"])
    mast_end = float(params["mast_end_angle"])
    mast_steps = int(params["mast_steps"])
    mast_angles = _np.linspace(mast_start, mast_end, mast_steps, endpoint=False, dtype=float)
    rx = tx = None
    per_angle_complex = None
    try:
        motor_controller.rotate_mast(mast_start)
        rx = RxRadio.RadioFlowGraph(params["rx_radio_id"], freq, params["rx_freq_offset"])
        tx = TxRadio.RadioFlowGraph(params["tx_radio_id"], freq, params["tx_freq_offset"])
        rx.start()
        time.sleep(max(0.05, float(pre_tx_delay_s)))
        tx.start()
        print(f"Time‑synchronised scan (rectangular TG) @ {freq/1e9:.3f} GHz: rotating & collecting …")
        motor_controller.rotate_mast(mast_end)
        rx.stop()
        rx.wait()
        tx.stop()
        tx.wait()
        I = _np.array(rx.vector_sink_0.data(), dtype=float)
        Q = _np.array(rx.vector_sink_1.data(), dtype=float)
        L = int(min(I.size, Q.size))
        if L == 0:
            print("No samples received.")
            return {
                "angles_deg": mast_angles.tolist(),
                "complex_per_angle": [],
                "noisy_db": [],
                "tg_db_new": [],
            }
        samps = I[:L] + 1j * Q[:L]
        i0 = _find_tx_on_edge(samps, win=int(edge_win), k_sigma=float(edge_k_sigma))
        if (L - i0) >= max(1024, mast_steps * 8):
            samps = samps[i0:]
        else:
            print("WARNING: short post‑edge segment; using full buffer.")
        idx = _np.linspace(0, len(samps), mast_steps + 1, dtype=int)
        per_angle = _np.zeros(mast_steps, dtype=np.complex128)
        for i in range(mast_steps):
            s, e = idx[i], idx[i + 1]
            seg = samps[s:e] if e > s else samps[s:s + 1]
            per_angle[i] = seg.mean()
        per_angle_complex = per_angle
    finally:
        try:
            if rx:
                rx.stop()
                rx.wait()
        except Exception:
            pass
        try:
            if tx:
                tx.stop()
                tx.wait()
        except Exception:
            pass
        try:
            motor_controller.rotate_mast(0)
        except Exception:
            pass
    for ang, cval in zip(mast_angles, per_angle_complex):
        datafile.write(f"{ang:.1f},0.0,0.0,{cval.real:.8e},{cval.imag:.8e}\n")
    datafile.close()
    print("raw datafile closed")
    noisy_mag = _np.abs(per_angle_complex)
    noisy_mag /= (_np.max(noisy_mag) if _np.max(noisy_mag) > 0 else 1.0)
    noisy_db = 20 * _np.log10(_np.clip(noisy_mag, 1e-12, None))
    # Rectangular gating
    freq_resp = per_angle_complex[:, _np.newaxis]
    freq_list = _np.array([freq], dtype=float)
    # Convert fallback width to seconds
    gate_width_s = float(gate_ns) * 1e-9
    gated_pattern = NewTimeGating.apply_time_gating_matrix_rect(
        freq_resp, freq_list, gate_width_s=gate_width_s, thr_factor=float(thr_factor)
    )
    # apply_time_gating_matrix_rect returns dB already normalised
    tg_db = gated_pattern
    if show_plots:
        plot_polar_patterns(
            mast_angles,
            traces=[
                ("Raw (noisy)", noisy_db),
                ("Rect‑Gated", tg_db)
            ],
            rmin=-60,
            rmax=0,
            rticks=(-60, -40, -20, 0),
            title=f"New Rectangular Gating Pattern @ {freq/1e9:.2f} GHz"
        )
    return {
        "angles_deg": mast_angles.tolist(),
        "complex_per_angle": per_angle_complex.tolist(),
        "noisy_db": noisy_db.tolist(),
        "tg_db_new": tg_db.tolist(),
    }

# -----------------------------------------------------------------------------
# Simple plot helper functions
#
# The original project provided additional helpers for parsing CSV files
# produced by this software and plotting them from the CLI.  To maintain
# compatibility with the command‑line and GUI interfaces these functions are
# reproduced here in simplified form.  Each routine delegates to the
# ``PlotGraph`` class defined elsewhere in the project to generate the plots.

def get_plot_data(text: Iterable[str]) -> List[Tuple[float, float, float, float]]:
    """
    Parse lines of comma separated measurement data into a list of tuples.

    Each line in ``text`` should contain four numeric fields separated by
    commas: mast angle (degrees), arm angle (degrees), background RSSI and
    a value (either transmission RSSI or gated magnitude).  The returned
    list consists of 4‑tuples of floats.  Non‑numeric lines are ignored.

    Parameters
    ----------
    text : iterable of str
        Lines of text from a measurement file, excluding the header lines.

    Returns
    -------
    list of tuple(float, float, float, float)
        Parsed data tuples.
    """
    file_data: List[Tuple[float, float, float, float]] = []
    for data_string in text:
        parts = data_string.strip().split(',')
        if len(parts) < 4:
            continue
        try:
            angle = float(parts[0])
            arm = float(parts[1])
            bkg = float(parts[2])
            val = float(parts[3])
            file_data.append((angle, arm, bkg, val))
        except ValueError:
            # skip lines with non‑numeric data
            continue
    return file_data


def PlotFile() -> None:
    """
    Prompt for a data file name from the user and plot it.

    This helper replicates the behaviour of the original ``PlotFile`` function
    by asking the user for a filename, reading the file, parsing the data
    via ``get_plot_data`` and launching a ``PlotGraph`` to display the
    result.  If the file cannot be read an error message is printed.
    """
    file_name = input("Enter the name of the data to plot\n")
    try:
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print(f"Failed to open {file_name}: {e}")
        return
    # Remove notes and header lines if present
    lines = lines[2:] if len(lines) >= 2 else lines
    file_data = get_plot_data(lines)
    pg = PlotGraph(file_data, file_name)
    pg.show()


def PlotFiles() -> None:
    """
    Prompt for two data file names from the user and plot them.

    This helper allows a quick comparison of two measurement files.  It
    prompts the user for two filenames, parses each file into data tuples
    and displays two separate plots.  If either file fails to open an
    error message is printed and the operation is aborted.
    """
    file_name1 = input("Enter the name of the first file to plot\n")
    try:
        with open(file_name1, 'r') as fr:
            lines1 = fr.readlines()
    except Exception as e:
        print(f"Failed to open {file_name1}: {e}")
        return
    file_name2 = input("Enter the name of the second file to plot\n")
    try:
        with open(file_name2, 'r') as fr:
            lines2 = fr.readlines()
    except Exception as e:
        print(f"Failed to open {file_name2}: {e}")
        return
    # Strip headers
    lines1 = lines1[2:] if len(lines1) >= 2 else lines1
    lines2 = lines2[2:] if len(lines2) >= 2 else lines2
    data1 = get_plot_data(lines1)
    data2 = get_plot_data(lines2)
    pg1 = PlotGraph(data1, file_name1)
    pg2 = PlotGraph(data2, file_name2)
    # Display each graph separately.  Users can close one to view the other.
    pg1.show()
    pg2.show()


def do_NSmeas(params: Dict[str, Any]):
    """
    Noise subtraction measurement (stub).

    The original project implemented a noise subtraction measurement.  A
    full implementation requires additional hardware control and is beyond
    the scope of this revision.  This stub is provided for interface
    compatibility and simply reports that noise subtraction is not yet
    implemented.

    Parameters
    ----------
    params : dict
        Measurement parameters loaded from a JSON file.  Currently unused.

    Returns
    -------
    None
    """
    print("Noise subtraction measurement is not implemented in this revision.")
    return None
