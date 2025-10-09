#!/usr/bin/python3
"""
Command line interface for the Low‑Cost Portable Antenna Range radio system.

This script presents a simple textual menu that maps user selections to
functions provided by ``RadioFunctions``.  The original implementation
has been extended with additional options to perform time‑synchronised
single frequency scans without gating, with legacy Tukey window gating
and with the new rectangular gating algorithm described in the supplied
paper.  When run, the menu prompts the user to choose an operation,
loads parameters from the default ``params.json`` file and calls the
corresponding function.  Errors are caught and printed along with a
traceback to aid debugging.
"""

from __future__ import annotations

import traceback

from PlotGraph import PlotGraph
import RadioFunctions


def main() -> None:
    quit = False
    data = None
    # user should preload measurement parameters in "params.json"
    param_filename = "params.json"
    # -------- Menu text (aligned with function calls below) --------
    # Menu choices with clearer descriptions.  Each entry
    # indicates the measurement type and whether time gating is applied.
    menu_choices: list[str] = []
    # 1. Continuous scan of AM amplitude (FastScan) with no time gating.
    menu_choices.append(
        "Continuous AM Scan (FastScan, rotating – no gating)"
    )
    # 2. Stepwise AM amplitude measurement with no time gating.
    menu_choices.append(
        "Stepwise AM Measurement (no gating)"
    )
    # 3. Noise subtraction measurement (not yet implemented in GUI).
    menu_choices.append("Noise Subtraction Measurement (stub)")
    # 4. Plot the data from the last run.
    menu_choices.append("Plot last run data")
    # 5. Plot data from a file.
    menu_choices.append("Plot data from file")
    # 6. Plot data from two files for comparison.
    menu_choices.append("Plot data from two files")
    # 7. Capture a single measurement with the transmitter on.
    menu_choices.append("Capture single measurement (Tx ON)")
    # 8. Capture a single background measurement with the transmitter off.
    menu_choices.append("Capture single background measurement (Tx OFF)")
    # 9. Legacy time‑gated AM measurement (stepwise) using Tukey window (supervised gating).
    menu_choices.append(
        "Legacy Time‑Gated AM Measurement – Stepwise (Tukey gating – supervised)"
    )
    # 10. Legacy time‑gated AM FastScan (continuous sweep) using Tukey window (supervised gating).
    menu_choices.append(
        "Legacy Time‑Gated AM FastScan – Continuous (Tukey gating – supervised)"
    )
    # 11. Legacy time‑gated single‑point frequency sweep (no rotation) using Tukey window (supervised).
    menu_choices.append(
        "Legacy Time‑Gated Single‑Point Sweep – (Tukey gating – supervised)"
    )
    # 12. Legacy time‑gated single frequency measurement at 2.5 GHz using Tukey window (supervised).
    menu_choices.append(
        "Legacy Time‑Gated Single Frequency @ 2.5 GHz (Tukey gating – supervised)"
    )
    # 13. Time‑synchronised single‑frequency scan with no time gating.
    menu_choices.append(
        "Time‑Sync Single Frequency – no gating"
    )
    # 14. Time‑synchronised single‑frequency scan with legacy Tukey gating (supervised).
    menu_choices.append(
        "Time‑Sync Single Frequency – Legacy Tukey gating (supervised)"
    )
    # 15. Time‑synchronised single‑frequency scan with the new unsupervised rectangular gating.
    menu_choices.append(
        "Time‑Sync Single Frequency – Unsupervised Rectangular gating (new)"
    )
    # 16. Quit the program.
    menu_choices.append("Quit")
    while not quit:
        try:
            selection = show_menu(menu_choices)
            if selection == 1:
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_AMscan(params)
                print(data)
            elif selection == 2:
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_AMmeas(params)
                print(data)
            elif selection == 3:
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_NSmeas(params)
                print(data)
            elif selection == 4:
                if data is None:
                    print("Run a scan before plotting data\n")
                    continue
                title = input("Enter a title for the graph: ")
                plot_graph = PlotGraph(data, title)
                plot_graph.show()
            elif selection == 5:
                RadioFunctions.PlotFile()
            elif selection == 6:
                RadioFunctions.PlotFiles()
            elif selection == 7:
                # Single measurement with transmitter on.
                print("Capture single measurement (Tx ON)")
                data_val = RadioFunctions.do_single(Tx=True)
                print("RMS = {:.3e}".format(data_val))
            elif selection == 8:
                # Single background measurement with transmitter off.
                print("Capture single background measurement (Tx OFF)")
                data_val = RadioFunctions.do_single(Tx=False)
                print("RMS = {:.3e}".format(data_val))
            elif selection == 9:
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_AMTGmeas(params)
                print(data)
            elif selection == 10:
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_AMTGscan(params)  # (returns after plots & file writes)
            elif selection == 11:
                # Stationary, frequency‑sweep time‑gate sanity test
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_singleTG(params)  # prints TG dB values; no rotation
            elif selection == 12:
                # Rotating scan at one frequency (defaults to 2.5 GHz)
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_AMTGscan_single_freq(params, freq_hz=2.5e9, show_plots=True)
            elif selection == 13:
                # New: time‑synchronised single frequency scan with no gating
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_time_sync_no_tg(params, freq_hz=2.5e9, show_plots=True)
                print(data)
            elif selection == 14:
                # New: time‑synchronised single frequency scan with Tukey gating
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_time_sync_tg_old(params, freq_hz=2.5e9, show_plots=True)
                print(data)
            elif selection == 15:
                # New: time‑synchronised single frequency scan with rectangular gating
                params = RadioFunctions.LoadParams(param_filename)
                data = RadioFunctions.do_time_sync_tg_new(params, freq_hz=2.5e9, show_plots=True)
                print(data)
            elif selection == 16:
                print("Exiting...\n")
                quit = True
        except Exception as e:
            print("Operation failed")
            print(e)
            print(traceback.format_exc())
            input("Press enter to continue")
    return  # exit‑user quit


def show_menu(choices: list[str]) -> int:
    if not choices:
        return 0
    print("\n\n\n")
    print("Please select from the following options:\n")
    for i, choice in enumerate(choices, start=1):
        print(f"{i}: {choice}\n")
    print("\n")
    selection = 0
    while selection < 1 or selection > len(choices):
        try:
            selection = int(input("Please enter selection: "))
        except ValueError:
            selection = 0
    return selection


def compute_far_field(freq: float, diameter: float) -> float:
    c = 2.99792458e8
    wavelength = c / freq
    r = (2 * diameter ** 2) / wavelength
    return r


if __name__ == "__main__":
    main()