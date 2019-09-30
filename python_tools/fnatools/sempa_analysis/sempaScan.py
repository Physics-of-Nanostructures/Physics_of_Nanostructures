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


def import_SEMPA_scans(source: str, merge=True, ):
    source = Path(source)

    if source.is_file():
        data = SEMPA_Scan(source)
    elif source.exists():
        files = source.glob("*.Detector_flat")
        data = [SEMPA_Scan(file) for file in files]
        if merge:
            data = SEMPA_Scan.average(data)
    else:
        raise FileNotFoundError("File or directory not found.")

    return data


@dataclass
class SEMPA_Scan:
    """
    Class to import and process a single SEMPA scan
    """

    datasource: InitVar[(str, SEMPA_Scan, np.ndarray)] = None
    x_data: InitVar[np.ndarray] = None
    y_data: InitVar[np.ndarray] = None

    def __post_init__(self, datasource, x_data, y_data):
        if isinstance(datasource, (str, Path)):
            self.filename = datasource
            self.read_file()
            self.reshape_data()
            # print(self.channels.shape)

        elif isinstance(datasource, SEMPA_Scan):
            self.channels = datasource.channels
            self.x = datasource.x
            self.y = datasource.y

        elif isinstance(datasource, np.ndarray):
            self.channels = datasource

            if x_data is None or y_data is None:
                ValueError(
                    "x_data and y_data should be present when " +
                    "channel data is provided as np.ndarray"
                )

            self.x = x_data
            self.y = y_data

        self.reset_corrections(datasource)

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
        self.channels = self.channels.astype(float)

        return self

    @property
    def mesh(self):
        mesh = Info()
        mesh.X, mesh.Y = np.meshgrid(self.x, self.y)
        return mesh

    @property
    def X(self):
        return self.mesh.X

    @property
    def Y(self):
        return self.mesh.Y

    @property
    def ch1(self):
        return self.channels[:, :, 0] - self.bg_channels[:, :, 0]

    @property
    def ch2(self):
        return self.channels[:, :, 1] - self.bg_channels[:, :, 1]

    @property
    def ch3(self):
        return self.channels[:, :, 2] - self.bg_channels[:, :, 2]

    @property
    def ch4(self):
        return self.channels[:, :, 3] - self.bg_channels[:, :, 3]

    @property
    def sem(self):
        return np.sum(self.channels, -1) - self.bg_sem

    @property
    def asym_x(self):
        return (self.channels[:, :, 0] - self.channels[:, :, 1]) / \
            (self.channels[:, :, 0] + self.channels[:, :, 1]) \
            - self.bg_asym_x - self.shift_asym_x

    @property
    def asym_y(self):
        return (self.channels[:, :, 2] - self.channels[:, :, 3]) / \
            (self.channels[:, :, 2] + self.channels[:, :, 3]) \
            - self.bg_asym_y - self.shift_asym_y

    def correct_drift(self, source_data: SEMPA_Scan,
                      crop_region=None, upsample_factor=100,
                      channel='sem', ret_shift=False):
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

        for i in range(shifted_data.channels.shape[-1]):
            image = np.fft.fftn(shifted_data.channels[:, :, i])
            image = fourier_shift(image, shift)
            shifted_channels.append(np.abs(np.fft.ifftn(image)))

        shifted_channels = np.stack(shifted_channels, -1)
        shifted_data.channels = shifted_channels

        if ret_shift:
            return shifted_data, shift
        else:
            return shifted_data

    def average(self, dataset=None, drift_correct=True, crop_drift=True):
        if isinstance(self, SEMPA_Scan):
            if isinstance(dataset, SEMPA_Scan):
                dataset = [dataset]
            elif isinstance(dataset, list):
                pass
            else:
                TypeError("dataset should be of type SEMPA_Scan or list")

            dataset.insert(0, self)

        if isinstance(self, list):
            dataset = self
            self = dataset[0]
        else:
            TypeError("self should be of type SEMPA_Scan or list ")

        # Drift_correct_data
        shifts = np.zeros(2)
        if drift_correct:
            for i in range(len(dataset) - 1):
                dataset[i + 1], shift = \
                    dataset[i + 1].correct_drift(dataset[0], ret_shift=True)

                shifts[0] = np.max([shifts[0], np.abs(shift[0])])
                shifts[1] = np.max([shifts[1], np.abs(shift[1])])

            shifts = np.ceil(shifts) + 1

        # Average
        average = np.sum(dataset) / len(dataset)

        if drift_correct and crop_drift:
            average = average.crop(edge=shifts)

        return average

    def crop(self, rect_size=None, rect_position=None, edge=None):
        if rect_size is not None:
            pass
        elif edge is not None:
            if np.isscalar(edge):
                edge = np.array([edge, edge])
            elif isinstance(edge, (list, tuple)):
                edge = np.array(edge)

            edge = edge.astype(int)

            rect_size = self.channels.shape[0:2] - edge * 2
        else:
            raise NotImplementedError(
                "None rect_size and edge not yet implemented")

        if isinstance(rect_size, (list, tuple)):
            rect_size = np.array(rect_size)

        if rect_position is None:
            rect_position = (self.channels.shape[0:2] - rect_size) // 2

        idx_x_min = rect_position[0]
        idx_x_max = rect_position[0] + rect_size[0]
        idx_y_min = rect_position[1]
        idx_y_max = rect_position[1] + rect_size[1]

        x = self.x[idx_x_min:idx_x_max]
        y = self.y[idx_y_min:idx_y_max]
        channels = self.channels[idx_y_min:idx_y_max, idx_x_min:idx_x_max, :]

        cropped_data = SEMPA_Scan(channels, x, y)
        return cropped_data

    def calculate_background(self, kx=3, ky=3, max_order=None):
        self.reset_corrections()
        args = [self.x, self.y, kx, ky, max_order]

        # background per channel
        bg_channels = [self.__calc_background__(
            self.channels[:, :, i], *args
        ) for i in range(self.channels.shape[-1])]

        bg_channels = np.stack(bg_channels, -1)

        self.bg_channels = bg_channels

        # background for composite images
        self.bg_sem = self.__calc_background__(self.sem, *args)
        self.bg_asym_x = self.__calc_background__(self.asym_x, *args)
        self.bg_asym_y = self.__calc_background__(self.asym_y, *args)

        return self

    def reset_corrections(self, datasource=None):
        if isinstance(datasource, SEMPA_Scan):
            self.bg_channels = datasource.bg_channels
            self.bg_sem = datasource.bg_sem
            self.bg_asym_x = datasource.bg_asym_x
            self.bg_asym_y = datasource.bg_asym_y

            self.shift_asym_x = datasource.shift_asym_x
            self.shift_asym_y = datasource.shift_asym_y

        else:
            self.bg_channels = np.zeros((1, 1, 4))
            self.bg_sem = 0
            self.bg_asym_x = 0
            self.bg_asym_y = 0

            self.shift_asym_x = 0
            self.shift_asym_y = 0

        return self

    def center_asymmetry(self, manual=False):
        self.shift_asym_x = 0
        self.shift_asym_y = 0

        if manual:
            NotImplementedError(
                "Manual asymmetry centring not yet implemented."
            )
        else:
            self.shift_asym_x = np.mean(self.asym_x)
            self.shift_asym_y = np.mean(self.asym_y)

        print("Implement value range detection")

        return self

    def process(self):
        self.calculate_background()
        self.center_asymmetry()
        return self

    @staticmethod
    def __calc_background__(channel, x, y, kx=3, ky=3, max_order=None):
        coefs, _, _, _ = polyfit2d(
            x, y, channel, kx=kx, ky=ky, order=max_order
        )
        coefs = coefs.reshape((kx + 1, ky + 1))

        background = np.polynomial.polynomial.polygrid2d(x, y, coefs)

        return background.T

    def __add__(self, other):
        other = self.__coerce_to_channel_shape__(other, '+')

        newdata = SEMPA_Scan(self)
        newdata.channels = self.channels + other

        return newdata

    def __sub__(self, other):
        other = self.__coerce_to_channel_shape__(other, '-')

        newdata = SEMPA_Scan(self)
        newdata.channels = self.channels - other

        return newdata

    def __mul__(self, other):
        other = self.__coerce_to_channel_shape__(other, '*')

        newdata = SEMPA_Scan(self)
        newdata.channels = self.channels * other

        return newdata

    def __truediv__(self, other):
        other = self.__coerce_to_channel_shape__(other, '/')

        newdata = SEMPA_Scan(self)
        newdata.channels = self.channels / other

        return newdata

    def __coerce_to_channel_shape__(self, other, operation="coercing"):
        if isinstance(other, SEMPA_Scan):
            other = other.channels
        elif isinstance(other, np.ndarray):
            if other.shape == self.channels.shape:
                pass
            elif other.shape[0:2] == self.channels.shape[0:2] and \
                    (len(other.shape) == 2 or other.shape[-1] == 1):
                other = np.tile(other, (self.channels.shape[-1], 1, 1))
                other = np.moveaxis(other, 0, 2)
            else:
                raise ValueError(
                    "Shape of arrays to add do not match: " +
                    f"{self.channels.shape} and {other.shape}"
                )
        elif np.isscalar(other):
            other = np.tile(other, self.channels.shape)
        else:
            raise TypeError(
                f"unsupported operand type(s) for {operation}:" +
                f"'{type(self)}' and '{type(other)}'"
            )

        return other


class Info:
    pass
