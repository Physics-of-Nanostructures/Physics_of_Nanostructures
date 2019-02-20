#
# This file is part of the mr_analysis package
#

import pandas
import numpy
import re


def pyMeasurement(filename):
    """
    Function to import MR measurements done with a pyMeasure measuring suite.

    Parameters
    ----------
    filename : string
        filename (relative path) of the data file.

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
            if line == "#Data:":
                break
            elif line == "#Parameters:":
                continue

            # Extract info from preamble
            if line.startswith("#Procedure:"):
                info = line.rsplit(":", maxsplit=1)
                metadata["Procedure"] = info[1].strip("\t <>")

            if line.startswith("#\t"):
                info = line[2:].rsplit(":", maxsplit=1)
                metadata[info[0].strip()] = info[1].strip()

    data = pandas.read_csv(filename, delimiter=",", comment="#")

    # Convert units if necessary
    columns = list(data.columns)
    rename_columns = dict()
    for column in columns:
        if column == "Magnetic Field (mT)":
            data[column] *= 1e-3
            rename_columns[column] = "Magnetic Field (T)"

    data.rename(columns=rename_columns, inplace=True)

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
            unit = 'None'
        name = name.strip()
        name = re.sub("\W|^(?=\d)", "_", name)
        name = name.replace("__", "_")
        name = name.strip("_")
        unit = unit.strip()
        unit = unit.replace(")", "")
        rename_columns[column] = name
        units[name] = unit

    data.rename(columns=rename_columns, inplace=True)
    metadata["Units"] = units

    return data, metadata
