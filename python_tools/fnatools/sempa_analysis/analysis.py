from .binaryFileReader import BinaryReader
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class SEMPA_Measurement:
    """
    Class to import and process SEMPA measurements
    """

    filename: str

    def __post_init__(self):
        self.read_file()
        self.reshape_data()

    def read_file(self):
        with open(self.filename, "rb") as file:
            if not file.read(8) == b"FLAT0100":
                raise ValueError("Unknown file-type")

            reader = BinaryReader(file)

            self.number_of_axes = reader.next_int()

            self.axes = [AxisInfo() for _ in range(self.number_of_axes)]

            for axis in self.axes:
                axis.name = reader.next_string()
                axis.name_parent = reader.next_string()
                axis.name_units = reader.next_string()

                axis.clock_count = reader.next_int()
                axis.raw_start_val = reader.next_int()
                axis.raw_increment_val = reader.next_int()

                axis.physical_start_val = reader.next_double()
                axis.physical_increment_val = reader.next_double()

                axis.mirrored = reader.next_int()
                axis.table_count = reader.next_int()

                if axis.table_count != 0:
                    raise NotImplementedError(
                        "Non-zero table-count not yet implemented")

            self.channel_name = reader.next_string()
            self.transfer_function_name = reader.next_string()
            self.unit_name = reader.next_string()

            self.parameter_count = reader.next_int()
            self.parameters = {
                reader.next_string(): reader.next_double()
                for _ in range(self.parameter_count)
            }

            self.number_of_data_views = reader.next_int()
            self.data_view = [reader.next_int()
                              for _ in range(self.number_of_data_views)]

            self.time_stamp = datetime.fromtimestamp(
                reader.next_long())
            self.information = {
                info.split("=")[0]: info.split("=")[1]
                for info in reader.next_string().split(";")
            }

            self.data_points = reader.next_int()
            self.data_points_measured = reader.next_int()

            if self.data_points != self.data_points_measured:
                raise ValueError("Measurement not completed")

            self.raw = np.array([
                reader.next_int()
                for _ in range(self.data_points)
            ])

    def reshape_data(self):
        self.x = self.axes[1].physical_start_val + \
            np.arange(self.axes[1].clock_count) * \
            self.axes[1].physical_increment_val
        self.y = self.axes[3].physical_start_val + \
            np.arange(self.axes[3].clock_count) * \
            self.axes[3].physical_increment_val

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.channel = np.reshape(
            self.raw,
            (4, self.axes[1].clock_count, self.axes[3].clock_count),
            order='F'
        )
        self.channel = np.swapaxes(self.channel, 0, 2)
        self.channel = np.rot90(self.channel, 2)
        self.channel = np.rollaxis(self.channel, 2)

        self.asym12 = (self.channel[0] - self.channel[1]) / \
            (self.channel[0] + self.channel[1])
        self.asym34 = (self.channel[2] - self.channel[3]) / \
            (self.channel[2] + self.channel[3])

        self.sem = np.sum(self.channel, 0)


class AxisInfo:
    pass
