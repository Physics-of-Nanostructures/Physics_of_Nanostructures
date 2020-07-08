from dataclasses import dataclass, InitVar
import pandas
import numpy
import re


@dataclass
class MagnetometryMeasurement:
    filename: str

    magnetic_volume: float = 0
    shape_factor: float = 1
    measurement_system: str = "MPMS3"

    keep_columns: InitVar[bool] = False
    no_print: InitVar[bool] = False

    def __post_init__(self, keep_columns=False, no_print=False):

        if self.measurement_system.lower() == "mpms3":
            self.import_mpms3(self.filename, keep_columns, no_print)

        self.correct_shape_factor()

        if not self.magnetic_volume == 0:
            self.calculate_magnetization()

        self.calculate_normalized_moment()

    def import_mpms3(self, filename, keep_columns=False, no_print=False):
        """
        Method to import measurements from a Quantum Design
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
        pandas.dataFrame
            The self.data from the datafile.
        dict
            contains other information obtained from the datafile
        """

        self.metadata = {}
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
                    self.metadata[info[1]] = info[0]

        self.data = pandas.read_csv(filename, skiprows=line_idx)

        # remove number of columns and add info to metadata
        self.metadata["ALL_COLUMNS"] = list(self.data.columns)

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

        self.metadata["Mass (grams)"] = numpy.mean(
            self.data["Mass (grams)"])
        self.metadata["Mass (grams) std.err."] = numpy.std(
            self.data["Mass (grams)"])

        for column in drop_columns:
            try:
                temp = set(self.data[column])
                if len(temp) == 1:
                    self.metadata[column] = list(temp)[0]

                if not keep_columns:
                    self.data.drop(labels=column, axis=1, inplace=True)
            except KeyError:
                pass

        # Timestamp fixing
        start_time = self.data["Time Stamp (sec)"].min()
        self.metadata["Time Stamp start (sec)"] = start_time
        self.data["Time Stamp (sec)"] -= start_time

        if not keep_columns:
            # Drop other columns that contain only nan values
            self.data.dropna(axis=1, how="all", inplace=True)

        # Extract comments and remove lines with only comments
        comments = {}
        if "Comment" in self.data:
            for idx, row in self.data[
                    numpy.logical_not(self.data["Comment"].isna())].iterrows():
                comments[row["Time Stamp (sec)"]] = row["Comment"]

            self.metadata["Comments"] = comments
            self.data.drop(labels=["Comment"], axis=1, inplace=True)
            columns = list(self.data.columns)
            self.data.dropna(axis=0, inplace=True, thresh=len(columns) - 1)

        # Convert units to SI
        columns = list(self.data.columns)
        rename_columns = dict()
        for column in columns:
            if "(sec)" in column:
                new_name = column.replace("(sec)", "(s)").strip()
                rename_columns[column] = new_name
            elif "(Oe)" in column:
                self.data[column] *= 1e3 / (4 * numpy.pi)
                new_name = column.replace("(Oe)", "(A/m)").strip()
                rename_columns[column] = new_name
            elif "(emu)" in column:
                self.data[column] *= 1e-3
                new_name = column.replace("(emu)", "(Am2)").strip()
                rename_columns[column] = new_name
            elif "(emu/Oe)" in column:
                self.data[column] *= 1e-6 * (4 * numpy.pi)
                new_name = column.replace("(emu/Oe)", "(Am)").strip()
                rename_columns[column] = new_name

        self.data.rename(columns=rename_columns, inplace=True)

        if "Magnetic Field (A/m)" in self.data:
            self.data["Induction (T)"] = self.data["Magnetic Field (A/m)"] * \
                4 * numpy.pi * 1e-7
            self.data.rename(columns={"Magnetic Field (A/m)": "Field (A/m)"},
                             inplace=True)

        # Rename columns to valid variable names and remove units
        columns = list(self.data.columns)
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

        self.data.rename(columns=rename_columns, inplace=True)
        self.metadata["Units"] = units

        # Reset the indices of the dataframe
        self.data.sort_values("Time_Stamp", inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        if not no_print:
            print()
            print("Columns, Unit")
            for key, value in self.metadata["Units"].items():
                print("{}, {}".format(key, value))

            print()

            if "Comments" in self.metadata:
                print(" Time, comment")
                for key, value in self.metadata["Comments"].items():
                    print("{: 5.0f}, {}".format(key, value))
                print()

        return self.data, self.metadata

    def correct_shape_factor(self, *, data: pandas.DataFrame = None,
                             shape_factor: float = None):
        """
        Correct the measured moment for the shape-artefact of the VSM-SQUID.
        The shape-factor (or shape-artefact) can be found in the documents
        provided with the VSM-SQUID.

        Parameters
        ----------
        data : pandas.DataFrame
            Data for which the magnetization is to be calculated if the
            class-data is not used. Replaces the class data.
        shape_factor : float
            The shape-factor or shape-artefact. The data is divided by
            this factor. Replaces the class-data.
        """

        if data is not None:
            self.data = data

        if shape_factor is not None:
            self.shape_factor = shape_factor

        self.data["Moment"] = self.data["Moment"] / self.shape_factor
        self.data["M_Std_Err"] = self.data["M_Std_Err"] / self.shape_factor

        if data is not None:
            return self.data
        else:
            return self

    def calculate_magnetization(self, *, data: pandas.DataFrame = None,
                                magnetic_volume: float = None):
        """
        Method to calculate the (average) magnetization from the total
        measured moments of the sample. Also calculates the uncertainty.

        Parameters
        ----------
        data : pandas.DataFrame
            Data for which the magnetization is to be calculated if the
            class-data is not used. Replaces the class data.
        magnetic_volume : float
            The magnetic volume of the sample. Replaces the class-data.

        """

        if data is not None:
            self.data = data

        if magnetic_volume is not None:
            self.magnetic_volume = magnetic_volume

        self.data["Magnetization"] = self.data["Moment"] / self.magnetic_volume
        self.data["Ma_Std_Err"] = self.data["M_Std_Err"] / self.magnetic_volume

        if data is not None:
            return self.data
        else:
            return self

    def calculate_normalized_moment(self, *, data: pandas.DataFrame = None):
        """
        Method to calculate the normalised moment from the total measured
        moments of the sample. Also calculates the uncertainty.

        Parameters
        ----------
        data : pandas.DataFrame
            Data for which the magnetization is to be calculated if the
            class-data is not used. Replaces the class data.

        """

        if data is not None:
            self.data = data

        max_val = numpy.max(numpy.abs(self.data["Moment"]))

        self.data["Moment_Normalized"] = self.data["Moment"] / max_val
        self.data["M_Norm_Std_Err"] = self.data["M_Std_Err"] / max_val

        if data is not None:
            return self.data
        else:
            return self

    def background_subtraction(self, *, data: pandas.DataFrame = None,
                               keep_uncorrected: bool = False,
                               slope_error: float = 1e-2, edge_points: int = 4,
                               use_field_weights: bool = True):
        """
        Method to correct an hysteresis loop for the (linear) background
        signal by looking at the straight parts of the measurement at the high
        (positive and negative) fields. Method assumes at least 4 data-points
        (ramping up and down combined) at either end of the measurement to be
        linear. This can be changed by passing the edge_points argument.

        Parameters
        ----------
        data : pandas.DataFrame
            Measurement that is to be corrected.
        keep_uncorrected : bool
            Keep the uncorrected moment as a new column ("Moment_uncorrected")
            in the DataFrame.
        slope_range : float
            The maximum error (part of the slope at the ends of the
            measurement) that is accepted for the linear regions in the
            measurement.
        edge_points : int
            The number of points on the edge that is assumed to be part of the
            linear regime. Default is 4, to account for double points on
            changing from ramping up to ramping down.
        use_field_weights : bool
            Use the induction (tanh(abs(B/250mT))) as weights for fitting a
            linear curve to the linear parts of the measurement. This makes the
            outer most edges the most influential for the subtracted
            background. Default is True.

        Returns
        -------
        pandas.DataFrame
            Background subtracted measurement.
        dict
            Offset (key is "offset") and slope (key is "slope") of the linear
            background, among other intermediate results.
        """

        if data is not None:
            self.data = data.copy()

        background_parameters = {}
        offset = 0
        slope = 0

        self.data.sort_values("Time_Stamp", inplace=True)

        field = numpy.array(self.data["Induction"])
        moment = numpy.array(self.data["Moment"])
        d2moment = numpy.abs(numpy.gradient(
            numpy.gradient(moment, field), field))
        d2moment = numpy.nan_to_num(d2moment)

        # Sort the data
        sort_idx = numpy.argsort(field)
        field = field[sort_idx[::+1]]
        moment = moment[sort_idx[::+1]]
        d2moment = d2moment[sort_idx[::+1]]

        # Determine maximum value for 2nd derivative
        maxval = numpy.nanmax(
            d2moment[edge_points: -edge_points]) * slope_error
        background_parameters["maxval"] = maxval

        # Linear at lower field
        idx = numpy.argmax(d2moment[edge_points:] > maxval) + edge_points

        if use_field_weights:
            w = numpy.tanh(numpy.abs(field[:idx]) / 250)
        else:
            w = None

        p = numpy.polyfit(field[:idx], moment[:idx], 1, w=w)
        slope1, offset1 = p[0], p[1]

        background_parameters.update(
            {"f1": field, "m1": moment, "d1": d2moment})
        background_parameters.update(
            {"F1": field[:idx], "M1": moment[:idx], "D1": d2moment[:idx]})
        background_parameters.update({"offset1": offset1, "slope1": slope1})

        # Invert arrays
        field = field[::-1]
        moment = moment[::-1]
        d2moment = d2moment[::-1]

        # Linear at higher field
        idx = numpy.argmax(d2moment[edge_points:] > maxval) + edge_points

        if use_field_weights:
            w = numpy.tanh(numpy.abs(field[:idx]) / 250)
        else:
            w = None

        p = numpy.polyfit(field[:idx], moment[:idx], 1, w=w)
        slope2, offset2 = p[0], p[1]

        background_parameters.update(
            {"f2": field, "m2": moment, "d2": d2moment})
        background_parameters.update(
            {"F2": field[:idx], "M2": moment[:idx], "D2": d2moment[:idx]})
        background_parameters.update({"offset2": offset2, "slope2": slope2})

        # Average slopes and offsets
        slope = (slope1 + slope2) / 2
        offset = (offset1 + offset2) / 2

        background_parameters.update({"offset": offset, "slope": slope})

        # Correct the measured moment
        if keep_uncorrected:
            self.data["Moment_uncorrected"] = self.data["Moment"]
        self.data["Moment"] -= offset + self.data["Induction"] * slope

        self.data.sort_values("Time_Stamp", inplace=True)

        if not self.magnetic_volume == 0:
            self.calculate_magnetization()

        self.calculate_normalized_moment()

        if data is not None:
            return self.data, background_parameters
        else:
            return self
