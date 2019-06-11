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
from .import_measurements import pyMeasurement
from .data_fitting import *


@dataclass
class hallMeasurement:
    path: str = "./"
    file_name_format: str = "*.txt"
    map_fn: map = map
    min_length: int = 401
    angle_offset: float = 0.
    series_resistance: float = 1e3
    device_resistance: float = 0
    angle_auto_correct: bool = True
    analyse_and_plot: InitVar[bool] = False
    orientation: str = "PHE"

    preprocessed = False
    standardized = False
    Data: pd.DataFrame = None
    results_f1: pd.DataFrame = None
    results_f2: pd.DataFrame = None

    def __post_init__(self, analyse_and_plot):
        # Validate input
        if self.orientation not in ["PHE", "AHE"]:
            raise ValueError("Orientation should be either AHE of PHE.")

        self.DataOriginal, self.MetaDataOriginal = import_multiple_data(
            self.path, self.file_name_format,
            map_fn=self.map_fn,
            min_length=self.min_length,
        )
        self.reset_to_original()
        if analyse_and_plot:
            self.full_analysis()
            self.plot_data()
            self.plot_results()

    # Wrapper functions
    def reset_to_original(self):
        self.Data = self.DataOriginal.copy(deep=True)
        self.MData = self.MetaDataOriginal.copy()
        self.preprocessed = False
        self.complement_data()

    def complement_data(self):
        self.Data, self.MData = sort_out_harmonics(self.Data, self.MData)
        self.Data, self.MData = sort_out_ac_source(self.Data, self.MData)
        self.Data, self.MData = include_current(
            self.Data, self.MData,
            R=self.series_resistance + self.device_resistance
        )

    def preprocess_data(self):
        self.Data, self.MData = remove_average_per_group(
            self.Data, self.MData,
            ["temperature_sp", "magnetic_field"],
            ["harmonic_1_x", "harmonic_1_y", "harmonic_1_r",
             "harmonic_2_x", "harmonic_2_y", "harmonic_2_r", ]
        )
        self.Data, self.MData = average_per_group(
            self.Data, self.MData,
            ["temperature_sp", "magnetic_field", "angle"]
        )
        self.Data, self.MData = correct_angle_per_group(
            self.Data, self.MData,
            ["temperature_sp", "magnetic_field"],
            angle_offset=self.angle_offset,
            auto_correct=self.angle_auto_correct,
        )
        self.Data, self.MData = average_per_group(
            self.Data, self.MData,
            ["temperature_sp", "magnetic_field", "angle"]
        )
        self.preprocessed = True

    def full_analysis(self, unified=False, preprocess=True, replacement_AHE=None):
        if preprocess and not self.preprocessed:
            self.preprocess_data()
        if self.orientation == "PHE":
            if unified:
                self.unified_analysis()
            else:
                self.individual_analysis_f1()
                if replacement_AHE is not None:
                    self.substitute_columns_f1(
                        replacement_AHE, ["h1_R_AHE", "h1_R_AHE_std"])
                self.normalize_f1_results()
                self.individual_analysis_f2()
        elif self.orientation == "AHE":
            self.AHE_analysis()
        else:
            raise ValueError("Orientation should be either AHE of PHE.")

    def unified_analysis(self):
        self.Data, self.results_f2 = dataset_analysis(
            self.Data, self.MData,
            group_keys=["temperature_sp"],
            analysis_fn=data_analysis_unified,
            map_fn=self.map_fn,
        )

    def individual_analysis_f1(self):
        self.Data, self.results_f1 = dataset_analysis(
            self.Data, self.MData,
            group_keys=["temperature_sp", "magnetic_field"],
            analysis_fn=data_analysis_individual,
            map_fn=self.map_fn,
        )

    def individual_analysis_f2(self):
        self.results_f1, self.results_f2 = results_f1_analysis(
            self.results_f1,
            min_field=0,
        )

    def AHE_analysis(self):
        self.Data, self.results_f1 = dataset_analysis(
            self.Data, self.MData,
            group_keys=["temperature_sp", "magnetic_field"],
            analysis_fn=data_analysis_AHE,
            map_fn=self.map_fn,
        )

    def plot_data(self, T=None, label=""):
        self.Data = sort_dataset(self.Data)
        plotData = self.Data.copy()

        if T is not None:
            if not isinstance(T, (list, tuple, )):
                T = [T]

            plotData = plotData[plotData["temperature_sp"].isin(T)]

        group = plotData.groupby("temperature_sp")

        for key, result in group:
            plot_measurements(result, "magnetic_field")
            plt.gcf().suptitle(f"{label} Temperature = {key:3.0f} K")

    def plot_results(self, keys_h1: dict=None, keys_h2: dict=None,
                     save=True, label=""):

        out = []
        if self.results_f1 is not None:
            if keys_h1 is not None:
                plot_keys = keys_h1
            else:
                unit = " Ω"
                if self.standardized:
                    unit = ""
                plot_keys = {
                    "h1_R_PHE": "R_PHE (Ω)",
                    "h1_R_AHE": "R_AHE (Ω)",
                    "h2_R_PHE_FL": "R_PHE_FL" + unit,
                    "h2_R_PHE_DL": "R_PHE_DL" + unit,
                    "h2_R_AHE_DL": "R_AHE_DL" + unit,
                }
            if isinstance(plot_keys, list):
                plot_keys = {k: k for k in plot_keys}

            plot_keys = {k: v for k, v in plot_keys.items()
                         if k in self.results_f1}

            fig, ax = plot_fit_results(
                self.results_f1, "temperature_sp",
                x_key="magnetic_field",
                x_label="Magnetic field (T)",
                plot_keys=plot_keys)
            fig.suptitle(label)

            if save:
                fig.savefig(self.path + "fit_h1_results.png",
                            transparent=True)

            out.append(fig)
            out.append(ax)

        if self.results_f2 is not None:
            if keys_h2 is not None:
                plot_keys = keys_h2
            else:
                plot_keys = {
                    "tau_a_tau": "τA",
                    "tau_b_tau": "τB",
                    "tau_s_tau": "τS",
                }

            if isinstance(plot_keys, list):
                plot_keys = {k: k for k in plot_keys}

            plot_keys = {k: v for k, v in plot_keys.items()
                         if k in self.results_f2}

            fig, ax = plot_fit_results(
                self.results_f2, None,
                x_key="temperature_sp",
                x_label="Temperature (K)",
                y_label="Torques (A/m)",
                single_axis=True,
                plot_keys=plot_keys)
            ax.legend()
            fig.suptitle(label)
            if save:
                fig.savefig(self.path + "fit_h2_results.png",
                            transparent=True)

            out.append(fig)
            out.append(ax)

        return out

    # Functions
    def mask_data(self, Data=None, mask_angles=[90, 270], mask_width=20):
        if Data is not None:
            self.Data = Data

        # for angle in mask_angles:
        #     diff = self.Data["angle"] - angle
        #     diff = np.mod(diff + 180, 360) - 180
        #     diff = np.abs(diff)
        #     print(diff <= mask_width)

        query_strs = [
            "abs(((angle - {} + 180) % 360) - 180) > {}".format(
                angle, mask_width
            )
            for angle in mask_angles]
        query_str = " and ".join(query_strs)

        self.Data.eval("mask = " + query_str, inplace=True)

    def normalize_f1_results(self, results_f1=None):
        if results_f1 is not None:
            self.results_f1 = results_f1

        self.results_f1.rename(columns={
            "h2_R_PHE_DL": "h2_R_PHE_DL_u",
            "h2_R_PHE_DL_std": "h2_R_PHE_DL_u_std",
            "h2_R_PHE_FL": "h2_R_PHE_FL_u",
            "h2_R_PHE_FL_std": "h2_R_PHE_FL_u_std",
            "h2_R_AHE_DL": "h2_R_AHE_DL_u",
            "h2_R_AHE_DL_std": "h2_R_AHE_DL_u_std",
        }, inplace=True)

        self.results_f1.eval(
            "h2_R_PHE_DL = h2_R_PHE_DL_u / h1_R_PHE",
            inplace=True)
        self.results_f1.eval(
            "h2_R_PHE_FL = h2_R_PHE_FL_u / h1_R_PHE",
            inplace=True)
        self.results_f1.eval(
            "h2_R_AHE_DL = h2_R_AHE_DL_u / h1_R_AHE",
            inplace=True)

        self.results_f1.eval(
            "h2_R_PHE_DL_std = h2_R_PHE_DL * sqrt(" +
            "(h2_R_PHE_DL_u_std / h2_R_PHE_DL_u)**2 + " +
            "(h1_R_PHE_std / h1_R_PHE)**2)",
            inplace=True)
        self.results_f1.eval(
            "h2_R_PHE_FL_std = h2_R_PHE_FL * sqrt(" +
            "(h2_R_PHE_FL_u_std / h2_R_PHE_FL_u)**2 + " +
            "(h1_R_PHE_std / h1_R_PHE)**2)",
            inplace=True)
        self.results_f1.eval(
            "h2_R_AHE_DL_std = h2_R_AHE_DL * sqrt(" +
            "(h2_R_AHE_DL_u_std / h2_R_AHE_DL_u)**2 + " +
            "(h1_R_AHE_std / h1_R_AHE)**2)",
            inplace=True)

        self.standardized = True

        return self.results_f1

    def substitute_columns_f1(self, newData: pd.DataFrame, columns: list):
        self.results_f1.set_index(
            ["temperature_sp", "magnetic_field"],
            inplace=True
        )
        newData = newData.set_index(
            ["temperature_sp", "magnetic_field"],
        )

        for column in columns:
            self.results_f1[column] = newData[column]

        self.results_f1.reset_index(drop=False, inplace=True)
        self.results_f1.dropna(subset=columns, inplace=True)

        return self.results_f1


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


def correct_angle_per_group(dataset, metadata, group_keys,
                            angle_offset=0, auto_correct=True):
    dataset["angle"] += angle_offset

    if auto_correct:
        def sine(phi, phi_0=0, A=1):
            return A * np.sin(2 * (phi - phi_0))

        sin_model = Model(sine, independent_vars=['phi'])
        sin_param = sin_model.make_params()
        sin_param["phi_0"].set(min=-np.pi / 2, max=+np.pi / 2)
        sin_param["A"].set(min=0)

        def subtract_angle(df):
            fit_res = sin_model.fit(
                df.harmonic_1_x,
                sin_param,
                phi=np.radians(df.angle)
            )

            phi_0 = fit_res.params["phi_0"].value
            phi_0 = np.degrees(phi_0)

            df["angle"] -= phi_0
            return df

        dataset = dataset.groupby(
            group_keys, as_index=False).apply(subtract_angle)

    dataset["angle"] = np.mod(dataset["angle"], 360)
    dataset = sort_dataset(dataset)

    metadata["manipulations"].append("Angle_corrected")

    return dataset, metadata


def sort_dataset(dataset,
                 keys=["temperature_sp", "magnetic_field", "angle"], ):
    dataset.sort_values(keys, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def data_analysis_unified(group_item, group_keys=None):
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

    data["harmonic_1_fit"] = fit_result_h1.best_fit
    data["harmonic_2_fit"] = fit_result_h2.best_fit

    group.update({
        "data": data,
        "h1_fit": fit_result_h1,
        "h2_fit": fit_result_h2,
    })

    return group


def data_analysis_individual_full(group_item, group_keys=None):
    group_values, data = group_item
    if not isinstance(group_values, (list, )):
        group_values = [group_values]

    group = {key: value for key, value in zip(group_keys, group_values)}

    model_h1 = Model(hall_voltage_function,
                     independent_vars=["phi", "I_0"])

    params_h1 = model_h1.make_params()
    params_h1["theta_M"].vary = False
    params_h1["theta_M0"].vary = False
    params_h1["R_AHE"].vary = False
    params_h1["H"].vary = False
    params_h1["H_A"].vary = False
    params_h1["phi_E"].vary = False
    # params_h1["V_0"].vary = False

    fit_result_h1 = model_h1.fit(
        data["harmonic_1_x"],
        params_h1,
        phi=data.angle,
        I_0=data.current,
        # H=data.magnetic_field,
    )

    model_h2 = Model(hall_voltage_2nd_harmonic_function,
                     independent_vars=["phi", "I_0"])

    params_h2 = model_h2.make_params()
    # params_h2["Ms"].value = 0.800
    params_h2["Ms"].vary = False
    # params_h2["gamma"].value = 1
    params_h2["gamma"].vary = False
    # params_h2["R_AHE"].value = 1
    params_h2["R_AHE"].vary = False
    # params_h2["V_0"].value = 0
    # params_h2["V_0"].vary = False
    params_h2["phi_0"].value = fit_result_h1.params['phi_0']
    # params_h2["phi_0"].vary = False
    # params_h2["phi_E"].value = fit_result_h1.params['phi_E']
    params_h2["phi_E"].vary = False
    # params_h2["H_A"].value = fit_result_h1.params['H_A']
    params_h2["H_A"].vary = False
    params_h2["H"].vary = False
    # params_h2["R_PHE"].value = fit_result_h1.params['R_PHE']
    params_h2["R_PHE"].vary = False
    params_h2["tau_iad"].vary = False
    params_h2["V_ANE"].vary = False

    fit_result_h2 = model_h2.fit(
        data["harmonic_2_y"],
        params_h2,
        phi=data.angle,
        I_0=data.current,
        # H=data.magnetic_field,
    )

    data["harmonic_1_fit"] = fit_result_h1.best_fit
    data["harmonic_2_fit"] = fit_result_h2.best_fit

    group.update({
        "data": data,
        "h1_fit": fit_result_h1,
        "h2_fit": fit_result_h2,
    })

    return group


def data_analysis_individual(group_item, group_keys=None):
    group_values, data = group_item
    if not isinstance(group_values, (list, tuple, )):
        group_values = [group_values]

    group = {key: value for key, value in zip(group_keys, group_values)}

    model_h1 = Model(simplified_1st_harmonic,
                     independent_vars=["phi", "I_0"])

    params_h1 = model_h1.make_params()
    params_h1["theta"].set(vary=False)
    params_h1["theta_0"].set(vary=False)
    params_h1["R_AHE"].set(vary=False)

    # model_h1 = Model(simplified_1st_harmonic_SW,
    #                  independent_vars=["phi", "I_0", "B"])

    # params_h1 = model_h1.make_params()
    # params_h1["phi_0"].set(value=0, vary=True)
    # params_h1["R_PHE"].set(value=1, vary=True)
    # params_h1["R_AHE"].set(value=1)
    # params_h1["theta"].set(value=0, vary=False)
    # params_h1["theta_0"].set(value=0, vary=False)
    # params_h1["Ms"].set(value=8.6e5, vary=True)
    # params_h1["Ku"].set(value=2.0e5, vary=True)
    # params_h1["Nx"].set(value=-0.1, vary=True, min=-1, max=1)
    # params_h1["Keb"].set(value=0, vary=False)
    # params_h1["theta_eb"].set(value=0, vary=False)
    # params_h1["phi_eb"].set(value=0, vary=False)

    fit_result_h1 = model_h1.fit(
        data["harmonic_1_x"],
        params_h1,
        phi=np.array(data.angle),
        # B=np.array(data.magnetic_field),
        I_0=np.array(data.current),
        # phi=data.angle,
        # I_0=data.current,
    )

    model_h2 = Model(simplified_2nd_harmonic,
                     independent_vars=["phi", "I_0"])

    params_h2 = model_h2.make_params()

    # params_h2["phi_0"].set(vary=True, value=fit_result_h1.params['phi_0'])

    fit_result_h2 = model_h2.fit(
        data["harmonic_2_y"],
        params_h2,
        phi=data.angle,
        I_0=data.current,
    )

    data["harmonic_1_fit"] = fit_result_h1.best_fit
    data["harmonic_2_fit"] = fit_result_h2.best_fit

    group.update({
        "data": data,
        "h1_fit": fit_result_h1,
        "h2_fit": fit_result_h2,
    })

    return group


def data_analysis_AHE(group_item, group_keys=None):
    # raise NotImplementedError("Fix Masking and parameters")
    group_values, data = group_item
    if not isinstance(group_values, (list, tuple, )):
        group_values = [group_values]

    group = {key: value for key, value in zip(group_keys, group_values)}

    fitdata = data.query("mask")

    model_h1 = Model(simplified_1st_harmonic,
                     independent_vars=["theta", "I_0"])

    params_h1 = model_h1.make_params()
    params_h1["phi"].set(value=60, vary=True, min=0, max=360)
    params_h1["phi_0"].set(value=0, vary=False)
    params_h1["R_PHE"].set(value=1, vary=True)
    params_h1["R_AHE"].set(value=1)
    params_h1["theta_0"].set(value=4, vary=True)

    # model_h1 = Model(simplified_1st_harmonic,
    #                  independent_vars=["theta", "phi", "I_0"])

    # params_h1 = model_h1.make_params()
    # params_h1["phi"].set(value=90)
    # params_h1["phi_0"].set(value=0)
    # params_h1["R_PHE"].set(value=1)
    # params_h1["R_AHE"].set(value=1)
    # params_h1["theta_0"].set(value=0)

    model_h1 = Model(simplified_1st_harmonic_SW,
                     independent_vars=["theta", "I_0", "B"])

    params_h1 = model_h1.make_params()
    params_h1["phi"].set(value=60, vary=True)  # , min=0, max=360
    params_h1["phi_0"].set(value=0, vary=False)
    params_h1["R_PHE"].set(value=1, vary=True)
    params_h1["R_AHE"].set(value=1)
    params_h1["theta_0"].set(value=4, vary=True)
    params_h1["Ms"].set(value=8.6e5, vary=True)
    params_h1["Ku"].set(value=2.0e5, vary=True)
    params_h1["Nx"].set(value=-0.1, vary=True, min=-1, max=1)
    params_h1["Keb"].set(value=0, vary=False)
    params_h1["theta_eb"].set(value=0, vary=False)
    params_h1["phi_eb"].set(value=0, vary=False)
    # params_h1["Keb"].set(value=0, vary=False)
    # params_h1["theta_eb"].set(value=0, vary=False)
    # params_h1["phi_eb"].set(value=0, vary=False)

    fit_result_h1 = model_h1.fit(
        fitdata["harmonic_1_x"],
        params_h1,
        theta=np.array(fitdata.angle),
        # theta=np.degrees(np.arccos(np.cos(np.radians(fitdata.angle))/3)),
        # phi=fitdata.angle,
        B=np.array(fitdata.magnetic_field),
        I_0=np.array(fitdata.current),
    )

    data["harmonic_1_fit"] = fit_result_h1.eval(
        theta=np.array(data.angle),
        # theta=np.degrees(np.arccos(np.cos(np.radians(data.angle))/3)),
        # phi=data.angle,
        B=np.array(data.magnetic_field),
        I_0=np.array(data.current),
    )

    group.update({
        "data": data,
        "h1_fit": fit_result_h1,
    })

    return group


def dataset_analysis(dataset, metadata, group_keys,
                     analysis_fn=data_analysis_unified, map_fn=map):

    data_groups = dataset.groupby(group_keys)

    partial_analysis_fn = partial(
        analysis_fn,
        group_keys=group_keys,
    )

    results_lst = []
    dataset_lst = []

    for group in map_fn(partial_analysis_fn, data_groups):
        dataset_lst.append(group["data"])
        results_lst.append({k: v for k, v in group.items() if not k == "data"})

    dataset = pd.concat(dataset_lst, )
    dataset.sort_values(group_keys, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    columns = [*group_keys]
    if "h1_fit" in results_lst[0]:
        columns.append("h1_fit")
    if "h2_fit" in results_lst[0]:
        columns.append("h2_fit")

    results = pd.DataFrame(
        results_lst,
        columns=columns
    )

    if "h1_fit" in results.columns:
        results = results.apply(add_fit_params_to_df, args=["h1_"],
                                axis=1, result_type="expand")
    if "h2_fit" in results.columns:
        results = results.apply(add_fit_params_to_df, args=["h2_"],
                                axis=1, result_type="expand")

    results.sort_values(group_keys, inplace=True)
    results.reset_index(drop=True, inplace=True)

    return dataset, results


def add_fit_params_to_df(df, prefix="h1_", use_ufloat=False):
    params = df[prefix + "fit"].params

    for name, param in params.items():
        if not use_ufloat:
            df[prefix + name] = param.value
            if param.stderr is not None:
                df[prefix + name + "_std"] = param.stderr
        else:
            df[prefix + name] = ufloat(param.value, param.stderr)

    return df


def normalize(x, x_range):
    x_min = x_range[0]
    x_max = x_range[1]
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def plot_measurements(data, subgroup_key=None, key_range=None):
    h1_fit = None
    h2_fit = None

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
    elif isinstance(data, pd.DataFrame):
        pass
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
        # color_fit = 'k'

        axes[0].plot(subdata.angle, subdata["harmonic_1_x"] + offset[0],
                     '.', color=color)
        axes[1].plot(subdata.angle, subdata["harmonic_2_y"] + offset[1],
                     '.', color=color)

        if "harmonic_1_fit" in subdata:
            axes[0].plot(subdata.angle, subdata["harmonic_1_fit"] + offset[0],
                         color=color_fit)
        elif h1_fit is not None:
            h1 = h1_fit.eval(phi=subdata.angle,
                             H=subdata.magnetic_field,
                             I_0=subdata.current)
            axes[0].plot(subdata.angle, h1 + offset[0],
                         color=color_fit)

        if "harmonic_2_fit" in subdata:
            axes[1].plot(subdata.angle, subdata["harmonic_2_fit"] + offset[1],
                         color=color_fit)
        elif h2_fit is not None:
            h2 = h2_fit.eval(phi=subdata.angle,
                             H=subdata.magnetic_field,
                             I_0=subdata.current)
            axes[1].plot(subdata.angle, h2 + offset[1],
                         color=color_fit)

        offset += 2 * np.std(data[["harmonic_1_x", "harmonic_2_y"]])

    if "mask" in subdata:
        for ax in axes:
            ymin, ymax = ax.get_ylim()
            collection = collections.BrokenBarHCollection.span_where(
                np.array(subdata["angle"]), ymin=ymin, ymax=ymax,
                where=np.logical_not(subdata["mask"]),
                facecolor='gray', alpha=0.5, zorder=10
            )
            ax.add_collection(collection, autolim=False)

    axes[1].set_xlabel("Angle (°)")
    axes[0].set_ylabel(r"V$_\mathregular{1ω}$ (V)")
    axes[1].set_ylabel(r"V$_\mathregular{2ω}$ (V)")
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].xaxis.set_major_locator(MultipleLocator(base=60))


def params_to_dict(params, prefix=""):
    param_dict = {}

    for name, param in params.items():
        param_dict.update({
            prefix + name: param.value,
            prefix + name + "_std": param.stderr,
        })

    return param_dict


def analysis_results_to_df(results, group_keys: list):
    fit_params = []

    if isinstance(results, dict):
        for key, result in results.items():
            if len(group_keys) == 1:
                param_dict = {group_keys[0]: key}
            param_dict.update({
                **params_to_dict(result["h1_fit"].params, "h1_"),
                **params_to_dict(result["h2_fit"].params, "h2_"),
            })

            fit_params.append(param_dict)
    elif isinstance(results, list):
        for result in results:
            print(result)
    else:
        raise NotImplementedError("Results is neither List or Dict")

    return pd.DataFrame(fit_params)


def results_f1_analysis(results_f1: pd.DataFrame,
                        group_key: str = "temperature_sp",
                        min_field: float = 0):
    results_f1["H"] = results_f1["magnetic_field"] / (4 * np.pi * 1e-7)
    f1_grouped = results_f1.groupby(group_key, sort=False)

    results_lst = []
    dataset_lst = []

    model = Model(simplified_torque_field_dependence)
    params_PHE = model.make_params()
    params_PHE["tau"].value = 0.1
    params_PHE["Ms"].vary = False
    params_PHE["Ms"].value = 0
    params_PHE["tau_0"].vary = True
    params_PHE["tau_0"].value = 0

    params_AHE = model.make_params()
    params_AHE["tau"].value = 1
    params_AHE["Ms"].value = 1
    params_AHE["tau_0"].value = 1

    for key, data in f1_grouped:
        data = data.copy(deep=True)
        idx = data["magnetic_field"] >= min_field

        tau_a_fit = model.fit(
            data["h2_R_PHE_FL"][idx], params_PHE, H=data.H[idx]
        )
        tau_b_fit = model.fit(
            data["h2_R_PHE_DL"][idx], params_PHE, H=data.H[idx]
        )
        tau_s_fit = model.fit(
            data["h2_R_AHE_DL"][idx], params_AHE, H=data.H[idx]
        )

        data["h2_R_PHE_FL_fit"] = tau_a_fit.eval(H=data.H)
        data["h2_R_PHE_DL_fit"] = tau_b_fit.eval(H=data.H)
        data["h2_R_AHE_DL_fit"] = tau_s_fit.eval(H=data.H)

        dataset_lst.append(data)
        results_lst.append({
            group_key: key,
            "tau_a_fit": tau_a_fit,
            "tau_b_fit": tau_b_fit,
            "tau_s_fit": tau_s_fit,
        })

    results_f1 = pd.concat(dataset_lst, )
    results_f1.sort_values(group_key, inplace=True)
    results_f1.reset_index(drop=True, inplace=True)

    results_f2 = pd.DataFrame(
        results_lst,
        columns=[group_key, "tau_a_fit", "tau_b_fit", "tau_s_fit"],
    )
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_a_"],
                                  axis=1, result_type="expand")
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_b_"],
                                  axis=1, result_type="expand")
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_s_"],
                                  axis=1, result_type="expand")
    results_f2.sort_values(group_key, inplace=True)
    results_f2.reset_index(drop=True, inplace=True)

    return results_f1, results_f2


def plot_fit_results(results, group_key="temperature_sp",
                     plot_keys={}, x_key="magnetic_field",
                     x_label=None, y_label=None,
                     single_axis=False):
    results.sort_values(x_key, inplace=True)

    if isinstance(plot_keys, list):
        plot_keys = {key: key for key in plot_keys}

    if not single_axis and not len(plot_keys) == 1:
        fig, axs = plt.subplots(len(plot_keys), 1, sharex=True)
        for idx, label in enumerate(plot_keys.values()):
            axs[idx].set_ylabel(label)
            axs[idx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    else:
        fig, ax = plt.subplots(1, 1)
        axs = [ax] * len(plot_keys)
        if y_label is not None:
            ax.set_ylabel(y_label)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    if x_label is not None:
        axs[-1].set_xlabel(x_label)
    elif x_key is not None:
        axs[-1].set_xlabel(x_key)

    if group_key is not None:
        key_range = results[group_key].agg(["min", "max"])

        for key, result in results.groupby(group_key):
            for idx, (plkey, pllabel) in enumerate(plot_keys.items()):
                cnorm = normalize(key, key_range)
                c = (cnorm, 0, 1 - cnorm)

                if x_key in result:
                    x = result[x_key]
                else:
                    x = result.eval(x_key)

                if plkey in result:
                    y = result[plkey]
                else:
                    y = result.eval(plkey)

                if plkey + "_std" in result:
                    y_error = result[plkey + "_std"]
                elif isinstance(y.iloc[0], Variable):
                    y = y.agg(nominal_value)
                    y_error = y.agg(std_dev)
                else:
                    y_error = None

                if plkey + "_fit" in result:
                    y_plot = result[plkey + '_fit']
                    ls_y = ''
                else:
                    y_plot = None
                    ls_y = '-'

                axs[idx].errorbar(
                    x, y, y_error,
                    marker='.', ls=ls_y, c=c,
                    label=f"{pllabel}"
                )

                if y_plot is not None:
                    axs[idx].plot(x, y_plot, color=c)
    else:
        result = results
        for idx, (plkey, pllabel) in enumerate(plot_keys.items()):

            if x_key in result:
                x = result[x_key]
            else:
                x = result.eval(x_key)

            if plkey in result:
                y = result[plkey]
            else:
                y = result.eval(plkey)

    results_f2 = pd.DataFrame(
        results_lst,
        columns=[group_key, "tau_a_fit", "tau_b_fit", "tau_s_fit"],
    )
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_a_"],
                                  axis=1, result_type="expand")
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_b_"],
                                  axis=1, result_type="expand")
    results_f2 = results_f2.apply(add_fit_params_to_df, args=["tau_s_"],
                                  axis=1, result_type="expand")
    results_f2.sort_values(group_key, inplace=True)
    results_f2.reset_index(drop=True, inplace=True)

    return results_f1, results_f2


def plot_fit_results(results, group_key="temperature_sp",
                     plot_keys={}, x_key="magnetic_field",
                     x_label=None, y_label=None,
                     single_axis=False):
    results.sort_values(x_key, inplace=True)

    if isinstance(plot_keys, list):
        plot_keys = {key: key for key in plot_keys}

    if not single_axis and not len(plot_keys) == 1:
        fig, axs = plt.subplots(len(plot_keys), 1, sharex=True)
        for idx, label in enumerate(plot_keys.values()):
            axs[idx].set_ylabel(label)
    else:
        fig, ax = plt.subplots(1, 1)
        axs = [ax] * len(plot_keys)
        if y_label is not None:
            ax.set_ylabel(y_label)

    if x_label is not None:
        axs[-1].set_xlabel(x_label)
    elif x_key is not None:
        axs[-1].set_xlabel(x_key)

    if group_key is not None:
        key_range = results[group_key].agg(["min", "max"])

        for key, result in results.groupby(group_key):
            for idx, (plkey, pllabel) in enumerate(plot_keys.items()):
                cnorm = normalize(key, key_range)
                c = (cnorm, 0, 1 - cnorm)

                if x_key in result:
                    x = result[x_key]
                else:
                    x = result.eval(x_key)

                if plkey in result:
                    y = result[plkey]
                else:
                    y = result.eval(plkey)

                if plkey + "_std" in result:
                    y_error = result[plkey + "_std"]
                elif isinstance(y.iloc[0], Variable):
                    y = y.agg(nominal_value)
                    y_error = y.agg(std_dev)
                else:
                    y_error = None

                if plkey + "_fit" in result:
                    y_plot = result[plkey + '_fit']
                    ls_y = ''
                else:
                    y_plot = None
                    ls_y = '-'

                axs[idx].errorbar(
                    x, y, y_error,
                    marker='.', ls=ls_y, c=c,
                    label=f"{pllabel}"
                )

                if y_plot is not None:
                    axs[idx].plot(x, y_plot, color=c)
    else:
        result = results
        for idx, (plkey, pllabel) in enumerate(plot_keys.items()):

            if x_key in result:
                x = result[x_key]
            else:
                x = result.eval(x_key)

            if plkey in result:
                y = result[plkey]
            else:
                y = result.eval(plkey)

            if plkey + "_std" in result:
                y_error = result[plkey + "_std"]
            elif isinstance(y.iloc[0], Variable):
                y = y.agg(nominal_value)
                y_error = y.agg(std_dev)
            else:
                y_error = None

            if plkey + "_fit" in result:
                y_plot = result[plkey + '_fit']
                ls_y = ''
            else:
                y_plot = None
                ls_y = '-'

            axs[idx].errorbar(
                x, y, y_error,
                marker='.', ls=ls_y,
                label=f"{pllabel}"
            )

            if y_plot is not None:
                axs[idx].plot(x, y_plot)

    if single_axis:
        axs = ax

    return fig, axs
