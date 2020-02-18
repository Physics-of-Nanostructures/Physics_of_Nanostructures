#
# This file is part of the mr_analysis package
#

import pandas
import numpy
import re


def pyMeasurement(filename, min_length=None, MD_in_DF=False, lower_case=False):
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

            elif line.startswith("#\t"):
                info = line[2:].rsplit(":", maxsplit=1)
                metadata[info[0].strip()] = info[1].strip()

    data = pandas.read_csv(filename, delimiter=",", comment="#")

    if min_length is not None:
        if len(data) < min_length:
            return None, None

    # Convert units if necessary
    columns = list(data.columns)
    rename_columns = dict()
    for column in columns:
        if column == "Magnetic Field (mT)":
            data[column] *= 1e-3
            rename_columns[column] = "Magnetic Field (T)"

    data.rename(columns=rename_columns, inplace=True)

    if MD_in_DF:
        str_value_w_unit = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? .*"
        re_value_w_unit = re.compile(str_value_w_unit)
        for key, value in metadata.items():
            if not isinstance(MD_in_DF, bool) and key not in MD_in_DF:
                continue

            unit = ""
            match = re_value_w_unit.match(value)
            if match:
                split = match.group().split()
                value = float(split[0])
                unit = " (" + split[1] + ")"
            else:
                if value in ["true", "True"]:
                    value = True
                elif value in ["false", "False"]:
                    value = False
                else:
                    try:
                        v_int = int(value)
                        v_float = float(value)
                    except ValueError:
                        pass
                    else:
                        if v_int == v_float:
                            value = v_int
                        else:
                            value = v_float

            if isinstance(MD_in_DF, dict):
                key = MD_in_DF[key]

            data[key + unit] = value

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

        if lower_case:
            name = name.lower()

        rename_columns[column] = name
        units[name] = unit

    data.rename(columns=rename_columns, inplace=True)
    metadata["Units"] = units

    return data, metadata
