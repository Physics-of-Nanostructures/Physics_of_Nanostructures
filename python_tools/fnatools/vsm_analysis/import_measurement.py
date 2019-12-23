#
# This file is part of the vsm_analysis package
#

import pandas
import numpy
import re

## DO NOT UPDATE, OBSOLETE
def mpms3(filename, keep_columns=False, no_print=False):
    """
    Function to import measurements from a Quantum Design
    MPMS3 VSM SQUID.

    Parameters
    ----------
    filename : string
        filename (relative path) of the MPMS3 VSM SQUID .dat file.
    keep_columns : bool, optional
        Indicate whether or not all columns are to be kept. If False
        (default), all empty columns will be removed from the dataframe.
    no_print : bool, optional
        Print comments and units of the returned dataframe.

    Returns
    -------
    pandas.DataFrame
        The data from the datafile.
    dict
        contains other information obtained from the datafile
    """

    metadata = {}
    with open(filename, "r") as file:
        line_idx = 0
        for line in file:
            line_idx += 1
            line = line.strip()

            # Check if Data section begins
            if line == "[Data]":
                break

            # Extract info from preamble
            if line.startswith("INFO"):
                info = line[5:].rsplit(",", maxsplit=1)
                metadata[info[1]] = info[0]

    data = pandas.read_csv(filename, skiprows=line_idx)

    # remove number of columns and add info to metadata
    metadata["ALL_COLUMNS"] = list(data.columns)

    drop_columns = ["Transport Action", "Field Status (code)",
                    "Chamber Status (code)", "Temp. Status (code)",
                    "Motor Status (code)", "Measure Status (code)",
                    "Center Position (mm)", "Redirection State",
                    "SQUID Status (code)", "Measure Count",
                    "Measurement Number", "Averaging Time (sec)",
                    "Mass (grams)", "Motor Lag (deg)",
                    "Pressure (Torr)", "Motor Current (amps)",
                    "Motor Temp. (C)", "Chamber Temp (K)",
                    "Average Temp (K)", "Peak Amplitude (mm)",
                    "Lockin Signal (V)", "Lockin Signal' (V)",
                    "Min. Temperature (K)", "Max. Temperature (K)",
                    "Min. Field (Oe)", "Max. Field (Oe)",
                    "Mass (grams)", "Range", "M. Quad. Signal (Am2)",
                    "M. Quad. Signal (emu)", "Frequency (Hz)"]

    metadata["Mass (grams)"] = numpy.mean(data["Mass (grams)"])
    metadata["Mass (grams) std.err."] = numpy.std(data["Mass (grams)"])

    for column in drop_columns:
        try:
            temp = set(data[column])
            if len(temp) == 1:
                metadata[column] = list(temp)[0]

            if not keep_columns:
                data.drop(labels=column, axis=1, inplace=True)
        except KeyError:
            pass

    # Timestamp fixing
    start_time = data["Time Stamp (sec)"].min()
    metadata["Time Stamp start (sec)"] = start_time
    data["Time Stamp (sec)"] -= start_time

    if not keep_columns:
        # Drop other columns that contain only nan values
        data.dropna(axis=1, how="all", inplace=True)

    # Extract comments and remove lines with only comments
    comments = {}
    if "Comment" in data:
        for idx, row in data[
                numpy.logical_not(data["Comment"].isna())].iterrows():
            comments[row["Time Stamp (sec)"]] = row["Comment"]

        metadata["Comments"] = comments
        data.drop(labels=["Comment"], axis=1, inplace=True)
        columns = list(data.columns)
        data.dropna(axis=0, inplace=True, thresh=len(columns) - 1)

    # Convert units to SI
    columns = list(data.columns)
    rename_columns = dict()
    for column in columns:
        if "(sec)" in column:
            new_name = column.replace("(sec)", "(s)").strip()
            rename_columns[column] = new_name
        elif "(Oe)" in column:
            data[column] *= 1e3 / (4 * numpy.pi)
            new_name = column.replace("(Oe)", "(A/m)").strip()
            rename_columns[column] = new_name
        elif "(emu)" in column:
            data[column] *= 1e-3
            new_name = column.replace("(emu)", "(Am2)").strip()
            rename_columns[column] = new_name
        elif "(emu/Oe)" in column:
            data[column] *= 1e-6 * (4 * numpy.pi)
            new_name = column.replace("(emu/Oe)", "(Am)").strip()
            rename_columns[column] = new_name

    data.rename(columns=rename_columns, inplace=True)

    if "Magnetic Field (A/m)" in data:
        data["Induction (T)"] = data["Magnetic Field (A/m)"] * \
            4 * numpy.pi * 1e-7
        data.rename(columns={"Magnetic Field (A/m)": "Field (A/m)"},
                    inplace=True)

    # Rename columns to valid variable names and remove units
    columns = list(data.columns)
    rename_columns = dict()
    units = dict()
    for column in columns:
        if column == "Comment":
            continue

        try:
            name, unit = column.split(" (", maxsplit=1)
        except ValueError:
            name = column
        else:
            unit = unit.strip()
            unit = unit.replace(")", "")
            units[name] = unit

        name = name.strip()
        name = re.sub(r"\W|^(?=\d)", "_", name)
        name = name.replace("__", "_")
        name = name.strip("_")
        rename_columns[column] = name

    data.rename(columns=rename_columns, inplace=True)
    metadata["Units"] = units

    # Reset the indices of the dataframe
    data.sort_values("Time_Stamp", inplace=True)
    data.reset_index(drop=True, inplace=True)

    if not no_print:
        print()
        print("Columns, Unit")
        for key, value in metadata["Units"].items():
            print("{}, {}".format(key, value))

        print()

        if "Comments" in metadata:
            print(" Time, comment")
            for key, value in metadata["Comments"].items():
                print("{: 5.0f}, {}".format(key, value))
            print()

    return data, metadata
