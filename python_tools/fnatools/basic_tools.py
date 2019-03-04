from re import sub
from pandas import DataFrame


def strip_column_names(columns: DataFrame.columns):
    """
    Function that strips the column names from clutter and splits in names and
    units.

    Parameters
    ----------
    columns : DataFrame.columns
        Columns to be stripped.

    Returns
    -------
    dict
        Contains replacement names.
    dict
        Contains units for each (replaced) column
    """
    column_list = list(columns)

    column_names = dict()
    column_units = dict()

    for column in column_list:
        name, unit = strip_unit(column)
        column_names[column] = name
        column_units[name] = unit

    return column_names, column_units


def strip_unit(header):
    try:
        name, unit = header.split(" (", maxsplit=1)
    except ValueError:
        name = header
        unit = None

    name = name.strip()
    name = sub("\W|^(?=\d)", "_", name)
    name = name.replace("__", "_")
    name = name.strip("_")
    unit = unit.strip()
    unit = unit.replace(")", "")
    return name, unit


def print_units(units):
    print("Columns, Unit")
    for key, value in units.items():
        print("{}, {}".format(key, value))
