import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import datetime
from lmfit import Model
from functools import partial
import re
from time import sleep, time
from scipy import integrate
from .import_measurements import pyMeasurement
from .data_fitting import *


def import_multiple_data(folder, filenames, map_fn=map, min_length=None):
    files = Path(folder).glob(filenames)

    fn = partial(pyMeasurement,
                 min_length=min_length,
                 MD_in_DF={"Temperature Set-point": "Temperature SP"},
                 lower_case=True,
                 )

    results = list(map_fn(fn, files))
    results = np.array(results).T.tolist()

    Data = pd.concat(results[0])

    MetaData = list(filter(None, results[1]))
    MetaData = {k: [d[k] for d in MetaData] for k in MetaData[0]}

    str_value_w_unit = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)? .*"
    re_value_w_unit = re.compile(str_value_w_unit)

    md_units = {}
    for key, value in MetaData.items():
        try:
            value = list(sorted(set(value)))
        except TypeError:
            if type(value[0]) == dict:
                value = [dict(t) for t in {tuple(d.items()) for d in value}]
            else:
                assert False, type(value[0])

        temp_value = []
        temp_unit = []
        for val in value:
            if isinstance(val, str):
                match = re_value_w_unit.match(val)
                if match:
                    split = match.group().split()
                    temp_value.append(float(split[0]))
                    temp_unit.append(split[1])
                else:
                    if val in ["true", "True"]:
                        temp_value.append(True)
                    elif val in ["false", "False"]:
                        temp_value.append(False)
                    else:
                        try:
                            v_int = int(val)
                            v_float = float(val)
                        except ValueError:
                            pass
                        else:
                            if v_int == v_float:
                                temp_value.append(v_int)
                            else:
                                temp_value.append(v_float)

        temp_unit = list(set(temp_unit))
        temp_value = list(sorted(set(temp_value)))

        if len(temp_value) == len(value):
            value = temp_value
            if len(temp_unit) == 1:
                md_units[key] = temp_unit[0]

        if len(value) == 1:
            value = value[0]

        MetaData[key] = value

    MetaData["Units"].update(md_units)

    Data.reset_index(drop=True, inplace=True)

    MetaData["manipulations"] = []
    MetaData["manipulations"].append("Imported")

    return Data, MetaData


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


def correct_angle_per_group(dataset, metadata, group_keys, ):

    # def sine(phi, phi_0=0.5, A=1):
    #     return A * np.sin(np.radians(2 * phi - phi_0))

    # model = Model(sine, independent_vars=['phi'])

    def subtract_angle(df):
        angle = np.radians(df.angle)

        Is = integrate.simps(df.harmonic_1_x * np.sin(angle * 2), angle)
        Ic = integrate.simps(df.harmonic_1_x * np.cos(angle * 2), angle)

        angle_0 = np.degrees(np.arctan(Is / Ic) / 2)

        df["angle"] -= angle_0
        df["angle"] = np.mod(df["angle"], 360)
        df.sort_values("angle", inplace=True)

        return df

    dataset = dataset.groupby(group_keys, as_index=False).apply(subtract_angle)

    metadata["manipulations"].append("Angle_corrected")

    return dataset, metadata


def dataset_analysis(dataset, metadata, group_keys, map_fn=map):

    data_groups = dataset.groupby(group_keys)

    data_analysis_fn = partial(
        data_analysis,
        group_keys=group_keys,
    )

    results = list(map_fn(data_analysis_fn, data_groups))

    if not isinstance(group_keys, list) or len(group_keys) == 1:
        results = {key: result for key, result in
                   zip(data_groups.groups.keys(), results)}

    return results


def data_analysis(group_item, group_keys=None):
    group_values, data = group_item
    if not isinstance(group_values, (list, )):
        group_values = [group_values]

    group = {key: value for key, value in zip(group_keys, group_values)}

    model_h1 = Model(hall_voltage_function,
                     independent_vars=["phi", "I_0", "H"])

    params_h1 = model_h1.make_params()
    params_h1["theta_M"].vary = False
    params_h1["theta_M"].value = 90
    params_h1["theta_M0"].vary = False
    params_h1["theta_M0"].value = 0
    params_h1["R_AHE"].vary = False
    params_h1["R_AHE"].value = 1
    params_h1["V_0"].vary = False
    params_h1["V_0"].value = 0

    fit_result_h1 = model_h1.fit(
        data["harmonic_1_x"],
        params_h1,
        phi=data.angle,
        I_0=data.current,
        H=data.magnetic_field,
    )

    model_h2 = Model(hall_voltage_2nd_harmonic_function,
                     independent_vars=["phi", "I_0", "H"])

    params_h2 = model_h2.make_params()
    params_h2["Ms"].value = 0.800
    params_h2["Ms"].vary = False
    params_h2["gamma"].value = 1
    params_h2["gamma"].vary = False
    params_h2["R_AHE"].value = 1
    params_h2["R_AHE"].vary = False
    params_h2["V_0"].value = 0
    params_h2["V_0"].vary = False
    params_h2["phi_0"].value = fit_result_h1.params['phi_0']
    # params_h2["phi_0"].vary = False
    params_h2["phi_E"].value = fit_result_h1.params['phi_E']
    # params_h2["phi_E"].vary = False
    params_h2["H_A"].value = fit_result_h1.params['H_A']
    # params_h2["H_A"].vary = False
    params_h2["R_PHE"].value = fit_result_h1.params['R_PHE']
    # params_h2["R_PHE"].vary = False

    fit_result_h2 = model_h2.fit(
        data["harmonic_2_y"],
        params_h2,
        phi=data.angle,
        I_0=data.current,
        H=data.magnetic_field,
    )

    group.update({
        "data": data,
        "h1_fit": fit_result_h1,
        "h2_fit": fit_result_h2,
    })

    return group


def normalize(x, x_range):
    x_min = x_range[0]
    x_max = x_range[1]
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def plot_measurements(data, subgroup_key=None, key_range=None):
    h1_fit = None
    h2_fit = None
    # print(data)

    if isinstance(data, dict):
        group = data
        if "data" in group:
            data = group["data"]
        if "h1_fit" in group:
            h1_fit = group["h1_fit"]
            # print(h1_fit.fit_report())
        if "h2_fit" in group:
            h2_fit = group["h2_fit"]
            # print(h2_fit.fit_report())
    else:
        raise NotImplementedError("Other situations not yet implemented")

    if key_range is None:
        key_range = (data[subgroup_key].min(), data[subgroup_key].max())

    fig, axes = plt.subplots(2, 1, sharex=True)

    offset = np.array([0, 0])

    for key, subdata in data.groupby(subgroup_key):
        key_norm = normalize(key, key_range)
        color = (key_norm, 0, 1 - key_norm)
        color_fit = (key_norm, 1, 1 - key_norm)

        axes[0].plot(subdata.angle, subdata["harmonic_1_x"] + offset[0],
                     '.', color=color)
        axes[1].plot(subdata.angle, subdata["harmonic_2_y"] + offset[1],
                     '.', color=color)

        if h1_fit is not None:
            h1 = h1_fit.eval(phi=subdata.angle,
                             H=subdata.magnetic_field,
                             I_0=subdata.current)
            axes[0].plot(subdata.angle, h1 + offset[0],
                         color=color_fit)

        if h2_fit is not None:
            h2 = h2_fit.eval(phi=subdata.angle,
                             H=subdata.magnetic_field,
                             I_0=subdata.current)
            axes[1].plot(subdata.angle, h2 + offset[1],
                         color=color_fit)

        offset += 2 * np.std(data[["harmonic_1_x", "harmonic_2_y"]])


def params_to_dict(params, prefix=""):
    param_dict = {}

    for name, param in params.items():
        param_dict.update({
            prefix + name: param.value,
            prefix + name + "_std": param.stderr,
        })

    return param_dict


def analysis_results_to_df(results, group_keys):
    fit_params = []

    for key, result in results.items():
        if len(group_keys) == 1:
            param_dict = {group_keys[0]: key}
        param_dict.update({
            **params_to_dict(result["h1_fit"].params, "h1_"),
            **params_to_dict(result["h2_fit"].params, "h2_"),
        })

        fit_params.append(param_dict)

    return pd.DataFrame(fit_params)
