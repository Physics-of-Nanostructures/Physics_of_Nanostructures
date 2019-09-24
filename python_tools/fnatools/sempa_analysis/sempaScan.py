from .binaryFileReader import BinaryReader
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class SEMPA_Scan:
    """
    Class to import and process SEMPA measurements
    """

    filename: str

    def __post_init__(self):
        self.read_file()
        self.reshape_data()

    def read_file(self):
        with open(self.filename, "rb") as file:
            reader = BinaryReader(file)

            # File identification
            if not file.read(8) == b"FLAT0100":
                raise ValueError("Unknown file-type")

            # Axis hierarchy description
            number_of_axes = reader.next_int()

            self.axes = [Info() for _ in range(number_of_axes)]

            for axis in self.axes:
                axis.name = reader.next_string()
                axis.name_parent = reader.next_string()
                axis.name_units = reader.next_string()

                axis.clock_count = reader.next_int()

                axis.raw_start_val = reader.next_int()
                axis.raw_increment_val = reader.next_int()

                axis.physical_start_val = reader.next_double()
                axis.physical_increment_val = reader.next_double()

                axis.mirrored = bool(reader.next_int())
                axis.table_count = reader.next_int()

                if axis.table_count != 0:
                    raise NotImplementedError(
                        "Non-zero table-count not yet implemented")

            # Channel description
            self.channel_name = reader.next_string()
            self.transfer_function_name = reader.next_string()
            self.unit_name = reader.next_string()

            parameter_count = reader.next_int()
            self.parameters = {
                reader.next_string(): reader.next_double()
                for _ in range(parameter_count)
            }

            number_of_data_views = reader.next_int()
            self.data_view = [reader.next_int()
                              for _ in range(number_of_data_views)]

            # Creation information
            self.time_stamp = datetime.fromtimestamp(
                reader.next_long())
            self.information = {
                info.split("=")[0]: info.split("=")[1]
                for info in reader.next_string().split(";")
            }

            # Raw data
            self.data_points = reader.next_int()
            self.data_points_measured = reader.next_int()

            if self.data_points != self.data_points_measured:
                raise ValueError("Measurement not completed")

            self.raw = np.array([
                reader.next_int()
                for _ in range(self.data_points)
            ])

            # Sample position information
            number_of_xy_pairs = reader.next_int()
            self.xy_offset_pairs = np.array([
                (reader.next_double(), reader.next_double())
                for _ in range(number_of_xy_pairs)
            ])

            # Experiment information
            self.experiment_name = reader.next_string()
            self.experiment_version = reader.next_string()
            self.experiment_description = reader.next_string()
            self.experiment_file_spec = reader.next_string()
            self.flat_file_creator_id = reader.next_string()
            self.result_file_creator_id = reader.next_string()
            self.matrix_user_name = reader.next_string()
            self.windows_account_name = reader.next_string()
            self.result_data_file_spec = reader.next_string()
            self.run_cycle_id = reader.next_int()
            self.scan_cycle_id = reader.next_int()

            # Experiment parameter list
            exp_element_count = reader.next_int()
            self.exp_elements = [Info() for _ in range(exp_element_count)]

            for element in self.exp_elements:
                element.name = reader.next_string()

                parameter_count = reader.next_int()
                element.parameters = dict()
                for _ in range(parameter_count):
                    name = reader.next_string()
                    val_type = reader.next_int()
                    unit = reader.next_string()
                    value = reader.next_string()

                    if val_type == 1:
                        # Integer value
                        value = int(value)
                    elif val_type == 2:
                        # Float value
                        value = float(value)
                    elif val_type == 3:
                        # Boolean value
                        value = value == "true"
                    elif val_type == 4:
                        # Enumeration value
                        value = int(value)
                    elif val_type == 5:
                        # String value
                        pass

                    element.parameters[name + "_" + unit] = value

            # Experiment element deployment parameter list
            exp_deploy_count = reader.next_int()
            self.exp_deploy = [Info() for _ in range(exp_deploy_count)]

            for element in self.exp_deploy:
                element.name = reader.next_string()

                parameter_count = reader.next_int()
                element.parameters = {
                    reader.next_string(): reader.next_string()
                    for _ in range(parameter_count)
                }

        return self

    def reshape_data(self):
        self.x = self.axes[1].physical_start_val + \
            np.arange(self.axes[1].clock_count) * \
            self.axes[1].physical_increment_val
        self.y = self.axes[3].physical_start_val + \
            np.arange(self.axes[3].clock_count) * \
            self.axes[3].physical_increment_val

        # self.X, self.Y = np.meshgrid(self.x, self.y)

        self.channels = np.reshape(
            self.raw,
            (4, self.axes[1].clock_count, self.axes[3].clock_count),
            order='F'
        )
        self.channels = np.swapaxes(self.channels, 0, 2)
        self.channels = np.rot90(self.channels, 2)
        self.channels = np.rollaxis(self.channels, 2)

        return self

    @property
    def sem(self):
        return np.sum(self.channels, 0)

    @property
    def asym_x(self):
        return (self.channels[0] - self.channels[1]) / \
            (self.channels[0] + self.channels[1])

    @property
    def asym_y(self):
        return (self.channels[2] - self.channels[3]) / \
            (self.channels[2] + self.channels[3])



class Info:
    pass
