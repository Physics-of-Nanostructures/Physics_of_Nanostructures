import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import collections
from pathlib import Path
from lmfit import Model
from functools import partial
import re
from scipy import integrate
from uncertainties import ufloat, nominal_value, std_dev
from uncertainties.core import Variable
from dataclasses import dataclass, InitVar
import typing
import copy
from .import_measurements import pyMeasurement
from .data_fitting import *


def sort_out_harmonics(dataset, metadata,):
    # Convert lock-in columns to harmonic-columns
    li1_harmonics = dataset["lock_in_1_harmonic"].unique()
    li2_harmonics = dataset["lock_in_2_harmonic"].unique()

    li1_harmonic = int(li1_harmonics[0])
    li2_harmonic = int(li2_harmonics[0])

    if len(li1_harmonics) != 1 or len(li2_harmonics) != 1:
        raise ValueError("Harmonics are not uniform for entire dataset")
    elif li1_harmonic == li2_harmonic:
        raise ValueError("Both lock-ins measured the same harmonic")

    replacements = {
        "lock_in_1_x": f"harmonic_{li1_harmonic:d}_x",
        "lock_in_1_y": f"harmonic_{li1_harmonic:d}_y",
        "lock_in_2_x": f"harmonic_{li2_harmonic:d}_x",
        "lock_in_2_y": f"harmonic_{li2_harmonic:d}_y",
    }

    dataset.rename(columns=replacements, inplace=True)
    dataset.drop([
        "lock_in_1_harmonic",
        "lock_in_2_harmonic",
    ], axis=1, inplace=True)

    metadata.update({
        "lock-in 1 harmonic": li1_harmonic,
        "lock-in 2 harmonic": li2_harmonic,
    })

    for old_key, new_key in replacements.items():
        metadata["Units"][new_key] = metadata["Units"].pop(old_key)

    dataset["harmonic_1_r"] = np.sqrt(dataset["harmonic_1_x"]**2 +
                                      dataset["harmonic_1_y"]**2)
    dataset["harmonic_2_r"] = np.sqrt(dataset["harmonic_2_x"]**2 +
                                      dataset["harmonic_2_y"]**2)

    metadata["manipulations"].append("Labels_converted")

    return dataset, metadata


def sort_out_ac_source(dataset, metadata,):
    li1_voltages = dataset["lock_in_1_v"].unique()
    li2_voltages = dataset["lock_in_2_v"].unique()

    li_voltages = {1: li1_voltages[0], 2: li2_voltages[0]}

    if len(li1_voltages) != 1 or len(li2_voltages) != 1:
        raise ValueError("Voltages are not uniform for entire dataset")
    elif not ((li_voltages[1] == 0.) ^ (li_voltages[2] == 0.)):
        raise ValueError("Both of the lock-ins set non-zero voltage")

    source = max(li_voltages, key=li_voltages.get)

    metadata.update({
        "AC Voltage source": source,
        "AC Voltage": li_voltages.pop(source),
    })

    replacements = {
        f"lock_in_{source:d}_v": "voltage",
        f"lock_in_{source:d}_f": "frequency",
    }

    dataset.rename(columns=replacements, inplace=True)

    for old_key, new_key in replacements.items():
        metadata["Units"][new_key] = metadata["Units"].pop(old_key)

    for no_source in li_voltages:
        drop = [
            f"lock_in_{no_source:d}_v",
            f"lock_in_{no_source:d}_f",
        ]
        dataset.drop(columns=drop, inplace=True)
        for drop_key in drop:
            metadata["Units"].pop(drop_key, None)

    metadata["manipulations"].append("AC_Source_interpretation")

    return dataset, metadata


def include_current(dataset, metadata, I0=None, R=None, U=None):
    """
    :params:
    I0 : optional float representing the current in A
    R : optional float representing the resistance in Ohm
    U : optional float representing the voltage in V

    """

    if I0 is not None:
        dataset["current"] = I0
        metadata["current"] = list(set(I0))
        metadata["Units"]["current"] = "A"
    else:
        if U is None:
            U = dataset["voltage"]

        if R is None:
            R = dataset["resistance"]

        dataset["current"] = U / R

    metadata["manipulations"].append("AC_Current_calculated")

    return dataset, metadata


def remove_average_per_group(dataset, metadata, group_keys, columns):

    def subtract_mean(df):
        df[columns] -= df[columns].mean()
        return df

    dataset = dataset.groupby(group_keys, as_index=False).apply(subtract_mean)

    metadata["manipulations"].append("Background_removed")

    return dataset, metadata


def average_per_group(dataset, metadata, group_keys):
    dataset = dataset.groupby(group_keys, as_index=False).mean()
    dataset.reset_index(drop=True, inplace=True)

    metadata["manipulations"].append("Binned_and_averaged")

    return dataset, metadata
