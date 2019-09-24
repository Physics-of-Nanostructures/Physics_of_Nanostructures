from __future__ import annotations

from .binaryFileReader import BinaryReader
from .polyfit2d import polyfit2d
from dataclasses import dataclass, InitVar
from datetime import datetime
from skimage.transform import rescale
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import numpy as np
from pathlib import Path


# @dataclass
# class SEMPA_Measurements:
#     """
#     Class to import and process SEMPA measurements
#     """

#     filenames: InitVar[str]

#     def __post_init__(self, filenames):
#         Path(filenames).glob()


@dataclass
class SEMPA_Scan:
    """
    Class to import and process a single SEMPA scan
    """

    datasource: InitVar[(str, SEMPA_Scan, np.ndarray)] = None
    x_data: InitVar[np.ndarray] = None
    y_data: InitVar[np.ndarray] = None

    def __post_init__(self, datasource, x_data, y_data):
        if isinstance(datasource, str):
            self.filename = datasource
            self.read_file()
            self.reshape_data()
        elif isinstance(datasource, SEMPA_Scan):
            self.channels = datasource.channels
            self.x = datasource.x
            self.y = datasource.y
        elif isinstance(datasource, np.ndarray):
            self.channels = datasource

        if isinstance(x_data, np.ndarray):
            self.x = x_data

        if isinstance(y_data, np.ndarray):
            self.y = y_data

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

        self.channels = np.reshape(
            self.raw,
            (4, self.axes[1].clock_count, self.axes[3].clock_count),
            order='F'
        )
        self.channels = np.swapaxes(self.channels, 0, 2)
        self.channels = np.rot90(self.channels, 2)

        return self

    @property
    def sem(self):
        return np.sum(self.channels, -1)

    @property
    def ch1(self):
        return self.channels[:, :, 0]

    @property
    def ch2(self):
        return self.channels[:, :, 1]

    @property
    def ch3(self):
        return self.channels[:, :, 2]

    @property
    def ch4(self):
        return self.channels[:, :, 3]

    @property
    def asym_x(self):
        return (self.channels[:, :, 0] - self.channels[:, :, 1]) / \
            (self.channels[:, :, 0] + self.channels[:, :, 1])

    @property
    def asym_y(self):
        return (self.channels[:, :, 2] - self.channels[:, :, 3]) / \
            (self.channels[:, :, 2] + self.channels[:, :, 3])

    def correct_drift(self, source_data: SEMPA_Scan,
                      crop_region=None, upsample_factor=100,
                      channel='sem'):
        if crop_region is not None:
            raise NotImplementedError("cropping not yet implemented")

        shifted_data = SEMPA_Scan(self)

        source_image = getattr(source_data, channel)
        shifted_image = getattr(self, channel)

        shift, error, _ = register_translation(source_image, shifted_image,
                                               upsample_factor=upsample_factor)

        shifted_data.shift = shift
        shifted_data.shift_error = error

        shifted_channels = []

        for i in range(4):
            image = np.fft.fftn(shifted_data.channels[:, :, i])
            image = fourier_shift(image, shift)
            shifted_channels.append(np.abs(np.fft.ifftn(image)))

        shifted_channels = np.stack(shifted_channels, -1)
        shifted_data.channels = shifted_channels
        return shifted_data

    def crop(self, rect_size=None, rect_position=None):
        if rect_size is None:
            raise NotImplementedError("None rect_size not yet implemented")
        if rect_position is None:
            raise NotImplementedError("None rect_position not yet implemented")

        idx_x_min = rect_position[0]
        idx_x_max = rect_position[0] + rect_size[0]
        idx_y_min = rect_position[1]
        idx_y_max = rect_position[1] + rect_size[1]

        x = self.x[idx_x_min:idx_x_max]
        y = self.y[idx_y_min:idx_y_max]
        channels = self.channels[idx_x_min:idx_x_max, idx_y_min:idx_y_max, :]

        cropped_data = SEMPA_Scan(channels, x, y)
        return cropped_data

    def remove_background(self, kx=3, ky=3, max_order=None):
        background = []

        for i in range(4):
            channel = self.channels[:, :, i]
            coefs, _, _, _ = polyfit2d(
                self.x, self.y, channel, kx=kx, ky=ky, order=max_order
            )
            coefs = coefs.reshape((kx + 1, ky + 1))

            background.append(
                np.polynomial.polynomial.polygrid2d(
                    self.x, self.y, coefs
                ))

        background = np.stack(background, -1)
        background = np.swapaxes(background, 0, 1)

        new_data = SEMPA_Scan(self)
        new_data -= background
        new_data.background = background

        return new_data

    def __add__(self, other):
        if isinstance(other, SEMPA_Scan):
            newchannels = self.channels + other.channels
        elif isinstance(other, np.ndarray):
            newchannels = self.channels + other
        elif np.isscalar(other):
            newchannels = self.channels + other
        else:
            raise TypeError(
                "unsupported operand type(s) for +:" +
                f"'{type(self)}' and '{type(other)}'"
            )

        newdata = SEMPA_Scan(self)
        newdata.channels = newchannels
        return newdata

    def __sub__(self, other):
        if isinstance(other, SEMPA_Scan):
            newchannels = self.channels - other.channels
        elif isinstance(other, np.ndarray):
            newchannels = self.channels - other
        elif np.isscalar(other):
            newchannels = self.channels - other
        else:
            raise TypeError(
                "unsupported operand type(s) for -:" +
                f"'{type(self)}' and '{type(other)}'"
            )

        newdata = SEMPA_Scan(self)
        newdata.channels = newchannels
        return newdata

    def __mul__(self, other):
        if np.isscalar(other):
            newchannels = self.channels * other
        else:
            raise TypeError(
                "unsupported operand type(s) for *:" +
                f"'{type(self)}' and '{type(other)}'"
            )

        newdata = SEMPA_Scan(self)
        newdata.channels = newchannels
        return newdata

    def __truediv__(self, other):
        if np.isscalar(other):
            newchannels = self.channels / other
        else:
            raise TypeError(
                "unsupported operand type(s) for /:" +
                f"'{type(self)}' and '{type(other)}'"
            )

        newdata = SEMPA_Scan(self)
        newdata.channels = newchannels
        return newdata


class Info:
    pass
