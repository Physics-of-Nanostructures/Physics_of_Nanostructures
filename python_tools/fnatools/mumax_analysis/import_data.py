from pandas import read_csv
from ..basic_tools import strip_column_names, print_units


def import_table(filename, no_print=False):
    """
    Function to import data from a MuMax3 simulation table.

    Parameters
    ----------
    filename : string or Path
        filename (relative path) of the MuMax3 table file.
    no_print : bool, optional
        Print comments and units of the returned dataframe.

    Returns
    -------
    pandas.DataFrame
        The data from the datafile.
    dict
        contains other information obtained from the datafile
    """

    data = read_csv(filename,
                    sep='\t')
    metadata = {}

    rename_columns, units = strip_column_names(data.columns)
    data.rename(columns=rename_columns, inplace=True)
    metadata["Units"] = units

    if not no_print:
        print_units(metadata["Units"])

    return data, metadata
