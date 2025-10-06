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
    menu_choices: list[str] = []
    menu_choices.append("FastScan (coherent AM, rotating)")            # 1 -> do_AMscan
    menu_choices.append("Measure (coherent AM, stepwise)")            # 2 -> do_AMmeas
    menu_choices.append("Measure (Noise Subtraction)")                 # 3 -> do_NSmeas
    menu_choices.append("Plot last run data")                          # 4
    menu_choices.append("Plot data from file")                        # 5
    menu_choices.append("Plot data from two files")                   # 6
    menu_choices.append("Capture single measurement (Tx ON)")         # 7 -> do_single(Tx=True)
    menu_choices.append("Capture single background (Tx OFF)")        # 8 -> do_single(Tx=False)
    menu_choices.append("Time-Gated Measure (coherent AM, stepwise)") # 9 -> do_AMTGmeas
    menu_choices.append("Time-Gated FastScan (rotating, freq SWEEP)") # 10 -> do_AMTGscan
    menu_choices.append("Time-Gated single pointing (freq SWEEP, no rotation)") # 11 -> do_singleTG
    menu_choices.append("Time-Gated single frequency @ 2.5 GHz (rotating)")    # 12 -> do_AMTGscan_single_freq
    menu_choices.append("Time-sync single freq (no gating)")          # 13 -> do_time_sync_no_tg
    menu_choices.append("Time-sync single freq (Tukey gating)")       # 14 -> do_time_sync_tg_old
    menu_choices.append("Time-sync single freq (Rect gating)")        # 15 -> do_time_sync_tg_new
    menu_choices.append("Quit")                                       # 16
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
                print("Single measurement (Tx ON)")
                data_val = RadioFunctions.do_single(Tx=True)
                print("RMS = {:.3e}".format(data_val))
            elif selection == 8:
                print("Single background (Tx OFF)")
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