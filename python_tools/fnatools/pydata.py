#
# This file is part of the fnatools package
#
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import Callable
from functools import partial
from itertools import repeat
from pprint import pprint

import tqdm
import pandas as pd
import numpy as np
import re


@dataclass
class pyData:
    """
    Class that reads and stores data from a file or set of files. The file requires
    a pyMeasure-compatible format (i.e. # for comments and metadata in comments)

    Parameters
    ----------
    filename : string
        filename (relative path) of the data file.
    min_length : None or int
        Minimum numbers of lines of data that is required for the file to be
        imported.
    MD_in_DF : bool, list, or dict
        determined whether data from the metadata should be put in the dataframe.
        If True, all metadata will be copied to the dataframe; if a list of keys
        is given, those keys will be copied from the metadata to the dataframe; if
        a dict is given it will copy the keys of the dict to columns in the dataframe
        with names given by the values in the dict.
    col_to_variable : bool
        Convert all columns to lower case and convert to valid variable names (e.g. no spaces).
    map_fn: Callable
        Determines the map function that is used, can e.g. be map (default), pool.map
        or pool.imap
    """

    filename: str = "./*txt"
    min_length: {None, int} = None
    MD_in_DF: {bool, list} = False
    col_to_variable: bool = True

    map_fn: Callable = map
    use_tqdm_gui: bool = False

    data: {pd.DataFrame, None} = None
    metadata: {dict, None} = None
    units: {dict, None} = None

    def __post_init__(self):
        # switch between tqdm and tqdm gui
        if self.use_tqdm_gui:
            self.tqdm_fn = tqdm.tqdm_gui
        else:
            self.tqdm_fn = tqdm.tqdm

        # convert filename to Path object
        self.file = Path(self.filename)

        # Import the file(s)
        self.data, self.metadata = self.import_files()

    def import_files(self):
        """
        TODO: docstring
        """

        # Search for the files
        files = self.file.parent.glob(self.file.name)

        # Generate a mapping to efficiently read all the files
        import_fn = partial(self.import_single_file,
                            min_length=self.min_length,
                            MD_in_DF=self.MD_in_DF)
        output = self.map_fn(import_fn, files)

        # Import data
        results = list(output)
        results = np.array(results, dtype=object).T.tolist()

        # Concatenate data into single dataframe
        data = pd.concat(results[0]) \
            .dropna(axis=1, how='all') \
            .reset_index(drop=True)

        # Merge metadata
        metadata_list = list(filter(None, results[1]))
        metadata, units = self.merge_metadata_dicts(metadata_list)

        self.units = units

        # Remove units and (if required) rename columns
        data, units = self.remove_units_and_rename_columns(data, self.col_to_variable)
        self.units.update(units)

        metadata["units"] = self.units

        return data, metadata

    @classmethod
    def import_single_file(cls, filename, min_length=None, MD_in_DF=False):
        """
        TODO: docstring
        """

        try:
            data = cls.import_single_file_data(filename)
            metadata = cls.import_single_file_metadata(filename)
        except FileNotFoundError:
            return None, None

        # Check if the dataframe has a sufficient length
        if min_length is not None and len(data) < min_length:
            return None, None

        # Add keys of the metadata to the dataframe (if required)
        if MD_in_DF:
            data = cls.add_metadata_to_dataframe(
                data, metadata, MD_in_DF
            )

        return data, metadata

    @staticmethod
    def import_single_file_data(filename):
        """
        TODO: docstring
        """
        data = pd.read_csv(
            filename,
            delimiter=",",
            comment='#',
            skipinitialspace=True,
            engine='c',
        )

        return data

    @staticmethod
    def import_single_file_metadata(filename):
        """
        TODO: docstring
        """
        metadata = dict()

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
                elif line.startswith("#Procedure:"):
                    info = line.rsplit(": ", maxsplit=1)
                    metadata["Procedure"] = info[1] \
                        .strip("\t <>") \
                        .replace("__main__.", "")

                elif line.startswith("#\t"):
                    info = line[2:].rsplit(": ", maxsplit=1)
                    metadata[info[0].strip()] = info[1].strip()

        return metadata

    @staticmethod
    def add_metadata_to_dataframe(data, metadata, key_to_colums):
        """
        TODO: docstring
        """

        if key_to_colums:
            str_value_w_unit = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? .*"
            re_value_w_unit = re.compile(str_value_w_unit)
            for key, value in metadata.items():
                if not isinstance(key_to_colums, bool) and key not in key_to_colums:
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

                if isinstance(key_to_colums, dict):
                    key = key_to_colums[key]

                data[key + unit] = value

        return data

    @staticmethod
    def merge_metadata_dicts(metadata_list):
        """
        TODO: docstring
        """

        # Get all keys
        keys = [k for md in metadata_list for k in md.keys()]
        keys = sorted(set(keys))

        # Prepare matching of value-unit pairs
        str_value_w_unit = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? .*"
        re_value_w_unit = re.compile(str_value_w_unit)

        metadata = dict()
        units = dict()

        for key in keys:
            # Get all values for a given key
            data = [md[key] for md in metadata_list]

            # Get all unique values
            data = sorted(set(data))

            # Try to separate values and units
            try:
                matches = [re_value_w_unit
                           .match(d)
                           .group()
                           .split()
                           for d in data]
            except AttributeError:
                # No match found
                values = np.array(data)
                # Try float
                try:
                    data = values.astype(np.float)
                except ValueError:
                    pass
                    # Not float
                    # Try boolean
                    options = {"false": False, "False": False,
                               "true": True, "True": True}
                    bools = np.vectorize(options.get)(values)
                    if bools.dtype == bool:
                        data = bools

            else:
                # Found a match to value-unit pattern
                matches = np.array(matches).T
                values = np.unique(matches[0])
                unit = np.unique(matches[1])

                if len(unit) == 1:
                    units[key] = unit[0]
                    data = values.astype(np.float)
                    data.sort()

            if len(data) == 1:
                data = data[0]

            metadata[key] = data

        return metadata, units

    @staticmethod
    def remove_units_and_rename_columns(data, col_to_variable):
        """
        TODO: docstring
        """

        rename_columns = dict()
        units = dict()

        for column in data.columns:
            if column == "Comment":
                continue
            try:
                name, unit = column.split(" (", maxsplit=1)
            except ValueError:
                name = column
                unit = 'None'

            name = name.strip()

            if col_to_variable:
                name = re.sub("\W|^(?=\d)", "_", name) \
                    .replace("__", "_") \
                    .strip("_") \
                    .lower()

            unit = unit.strip()
            unit = unit.replace(")", "")

            rename_columns[column] = name
            units[name] = unit

        data.rename(columns=rename_columns, inplace=True)

        return data, units
