import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from itertools import product


@dataclass
class XPS_Measurement:
    path: str

    number_of_channels: int = 5

    def __post_init__(self):
        self.path = Path(self.path)
        self.import_measurements()

    def import_measurements(self):
        # Collect metadata
        self.metadata = {}
        with open(self.path, "r") as file:
            line_idx = 0
            for line in file:
                line_idx += 1
                line = line.strip()

                # Check if Data section begins
                if line.startswith("# Cycle"):
                    break

                # Extract info from preamble
                if line.startswith("#"):
                    info = line[2:].split(":", maxsplit=1)
                    if len(info) == 2:
                        self.metadata[info[0].strip()] = info[1].strip()

        self.number_of_scans = int(self.metadata["Number of Scans"])
        self.values_per_scan = int(self.metadata["Values/Curve"])
        self.excitation_energy = float(self.metadata["Excitation Energy"])

        # Get all data from the file
        self.rawdata = pd.read_csv(self.path,
                                   names=["e_kin", "cps", "cps_err"],
                                   delim_whitespace=True, comment="#")

        self.rawdata.set_index("e_kin", inplace=True)

        tempdata = self.rawdata.copy()

        if "cps_err" in tempdata:
            tempdata.drop(columns=["cps_err"], inplace=True)

        # Check if number of lines matches expectations
        assert len(self.rawdata) == self.number_of_channels * \
            self.number_of_scans * self.values_per_scan, \
            "Number of values does not match expectations"

        # Split data into scans & channels
        Data = []
        Keys = []
        for scan, channel in product(
                range(self.number_of_scans), range(self.number_of_channels)):
            idx_start = self.values_per_scan * \
                (self.number_of_channels * scan + channel)
            idx_stop = self.values_per_scan * \
                (self.number_of_channels * scan + channel + 1)

            Data.append(tempdata.iloc[idx_start:idx_stop])
            Keys.append((f"S{scan:02d}", f"C{channel:02d}"))

        self.data = pd.concat(Data, axis=1, keys=Keys,
                              names=["scan", "channel"])
        self.data.columns = self.data.columns.droplevel(2)

        # Combine channels into single scan
        scan_sum = self.data.sum(level=0, axis=1)
        scan_sum = pd.concat([scan_sum], keys=["cps"], axis=1)
        scan_sum = scan_sum.swaplevel(axis=1)
        self.data = self.data.join(scan_sum)
        self.data.sort_values(["scan", "channel"], axis=1, inplace=True)

        # Average over all scans
        channel_avg = self.data.mean(level=1, axis=1)
        channel_avg = pd.concat([channel_avg], keys=["Total"], axis=1)
        self.data = self.data.join(channel_avg)
        self.data.sort_values(["scan", "channel"], axis=1, inplace=True)

        # Restore indices and energy columns
        self.data.reset_index(inplace=True)
        self.data["e_bin"] = self.excitation_energy - self.data["e_kin"]


def shirley_calculate(x, y, tol=1e-5, maxit=10, debug=False):
    """ S = specs.shirley_calculate(x,y, tol=1e-5, maxit=10)
    Calculate the best auto-Shirley background S for a dataset (x,y). Finds the biggest peak
    and then uses the minimum value either side of this peak as the terminal points of the
    Shirley background.
    The tolerance sets the convergence criterion, maxit sets the maximum number
    of iterations.
    """

    # Make sure we've been passed arrays and not lists.
    x = np.array(x)
    y = np.array(y)

    # Sanity check: Do we actually have data to process here?
    if not (x.any() and y.any()):
        print("specs.shirley_calculate: One of the arrays x or y is empty. Returning zero background.")
        return np.zeros(x.shape)

    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if x[0] < x[-1]:
        is_reversed = True
        x = x[::-1]
        y = y[::-1]
    else:
        is_reversed = False

    # Locate the biggest peak.
    maxidx = np.abs(y - np.amax(y)).argmin()

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    if maxidx == 0 or maxidx >= len(y) - 1:
        print("specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background.")
        return np.zeros(x.shape)

    # Locate the minima either side of maxidx.
    lmidx = np.abs(y[0:maxidx] - np.amin(y[0:maxidx])).argmin()
    rmidx = np.abs(y[maxidx:] - np.amin(y[maxidx:])).argmin() + maxidx
    xl = x[lmidx]
    yl = y[lmidx]
    xr = x[rmidx]
    yr = y[rmidx]

    # Max integration index
    imax = rmidx - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.
    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()

    it = 0
    while it < maxit:
        if debug:
            print("Shirley iteration: ", it)
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = 0.0
        for i in range(lmidx, imax):
            ksum += (x[i] - x[i + 1]) * 0.5 * (y[i] + y[i + 1]
                                               - 2 * yr - B[i] - B[i + 1])
        k = (yl - yr) / ksum
        # Calculate new B
        for i in range(lmidx, rmidx):
            ysum = 0.0
            for j in range(i, imax):
                ysum += (x[j] - x[j + 1]) * 0.5 * (y[j] +
                                                   y[j + 1] - 2 * yr - B[j] - B[j + 1])
            Bnew[i] = k * ysum
        # If Bnew is close to B, exit.
        if np.linalg.norm(Bnew - B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B
