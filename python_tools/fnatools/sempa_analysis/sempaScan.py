from __future__ import annotations

from .binaryFileReader import BinaryReader
from .polyfit2d import polyfit2d
from dataclasses import dataclass, InitVar
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpp
import matplotlib.colors as mpc
from matplotlib.widgets import RectangleSelector
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift, gaussian_filter
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
    Class to import and process a single SEMPA scan. Importing multiple scans
    can be done using the import_SEMPA_scans function, which returns a merged
    instance of this (SEMPA_Scan) class.

    Example usage:

        data = SEMPA_Scan(file) \
            .crop(169, (29, 27)) \
            .process()

        fig, ax = plt.subplot()
        pcolormesh(data.x * 1e6, data.y * 1e6, data.asym_x,
                   norm=data.norm)


    Parameters
    ----------
    datasource : string, pathlib.Path, SEMPA_Scan, np.ndarray
        The datasource from which to obtain the SEMPA scan data. If a string of
        Path is given, the data will be read from a source file (typically with
        extension ".Detector_flat"). If another SEMPA_Scan object is provided,
        the channel-, x- and y-data will be copied. The rest of the data will
        be neglected. If a np.ndarray is provided, this will be assumed to be
        the channel data and x- and y- data should be provided to the respective
        optional parameters.
    x_data : np.ndarray
        Array of x-values. Only used if datasource is given an np.ndarray.
    y_data : np.ndarray
        Array of y-values. Only used if datasource is given an np.ndarray.

    Methods
    -------
    read_file
    reshape_data
    correct_drift
    average
    crop
    calculate_background
    reset_corrections
    center_asymmetry
    process

    Attributes
    ----------
    filename
    x : np.ndarray
        Array of x-coordinates.
    y : np.ndarray
        Array of y-coordinates.
    norm : mpl.colors.Normalize
        A normalizer that can be used for plotting the asymmetry data. Created
        after centring the asymmetry.
    smooth : bool
        Enables or disables Gaussian smoothing.
    smooth_sigma : float
        Controls the standard deviation for Gaussian smoothing.
    X : np.ndarray, read-only
        Meshed version of the x-coordinates.
    Y : np.ndarray, read-only
        Meshed version of the y-coordinates.
    ch1 : np.ndarray, read-only
        Channel data of channel 1. Background is subtracted after background
        calculation.
    ch2 : np.ndarray, read-only
        Channel data of channel 2. Background is subtracted after background
        calculation.
    ch3 : np.ndarray, read-only
        Channel data of channel 3. Background is subtracted after background
        calculation.
    ch4 : np.ndarray, read-only
        Channel data of channel 4. Background is subtracted after background
        calculation.
    sem : np.ndarray, read-only
        SEM data, or sum of all channels. Background is subtracted after
        background calculation.
    asym_x : np.ndarray, read-only
        Horizontal asymmetry data (i.e. channel 1 minus 2). Background is
        subtracted after background calculation and smoothed when enabled.
    asym_y : np.ndarray, read-only
        Vertical asymmetry data (i.e. channel 3 minus 4). Background is
        subtracted after background calculation and smoothed when enabled.
    angle : np.ndarray, read-only
        The (planar) angle of the magnetization (i.e. the arctan of asym_x and
        asym_y). Smoothed when enabled.
    mag : np.ndarray, read-only
        The magnitude of the magnetization (i.e. norm of asym_x and asym_y).
        Background is subtracted after background calculation and smoothed when
        enabled.

    """

    datasource: InitVar[(str, SEMPA_Scan, np.ndarray)] = None
    x_data: InitVar[np.ndarray] = None
    y_data: InitVar[np.ndarray] = None

    def __post_init__(self, datasource, x_data, y_data):
        if isinstance(datasource, (str, Path)):
            self.filename = datasource
            self.read_file()
            self.reshape_data()

        elif isinstance(datasource, SEMPA_Scan):
            self.channels = datasource.channels
            self.x = datasource.x
            self.y = datasource.y

        elif isinstance(datasource, np.ndarray):
            self.channels = datasource

            if x_data is None or y_data is None:
                ValueError(
                    "x_data and y_data should be present when "
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
        self.channels = np.flipud(self.channels)
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
        asym_x = (self.channels[:, :, 0] - self.channels[:, :, 1]) / \
            (self.channels[:, :, 0] + self.channels[:, :, 1]) \
            - self.bg_asym_x - self.shift_asym_x

        if self.smooth:
            asym_x = gaussian_filter(asym_x, sigma=self.smooth_sigma)
        return asym_x

    @property
    def asym_y(self):
        asym_y = (self.channels[:, :, 2] - self.channels[:, :, 3]) / \
            (self.channels[:, :, 2] + self.channels[:, :, 3]) \
            - self.bg_asym_y - self.shift_asym_y

        if self.smooth:
            asym_y = gaussian_filter(asym_y, sigma=self.smooth_sigma)
        return asym_y

    @property
    def angle(self):
        return np.arctan2(self.asym_y, self.asym_x)

    @property
    def mag(self):
        return np.sqrt(self.asym_y**2 + self.asym_x**2)

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
            print("size:", len(self.x), "x", len(self.y))

            fig, ax = plt.subplots()
            ax.pcolormesh(self.x * 1e6, self.y * 1e6, self.mag)
            ax.set_aspect('equal')
            ax.set_xlim(auto=False)
            ax.set_ylim(auto=False)
            ax.set_title("Select region for cropping\n"
                         "Confirm with double-click or enter")

            def selector(event, event_release=None):
                if event.key == "enter" or (hasattr(event, "dblclick") and event.dblclick):
                    selector.RS.set_active(False)
                    plt.close(fig)

            selector.RS = RectangleSelector(ax, selector,
                                            drawtype='box',
                                            useblit=True,
                                            button=[1],
                                            minspanx=5,
                                            minspany=5,
                                            spancoords='pixels',
                                            interactive=True)

            fig.canvas.mpl_connect('key_press_event', selector)
            fig.canvas.mpl_connect('button_press_event', selector)

            plt.show()

            xmin, xmax, ymin, ymax = np.array(selector.RS.extents) * 1e-6

            xmin = np.argmax(self.x >= xmin)
            xmax = np.argmax(self.x >= xmax) - 1
            ymin = np.argmax(self.y >= ymin)
            ymax = np.argmax(self.y >= ymax) - 1

            rect_size = (xmax - xmin, ymax - ymin)
            rect_position = (xmin, ymin)

            print("Re-use the selected crop-region using:\n"
                  ".crop({}, {})".format(rect_size, rect_position))

        if isinstance(rect_size, (list, tuple)):
            rect_size = np.array(rect_size)
        elif isinstance(rect_size, (float, int)):
            rect_size = np.array([rect_size, rect_size])

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
            self.smooth = datasource.smooth
            self.smooth_sigma = datasource.smooth_sigma

            self.shift_asym_x = datasource.shift_asym_x
            self.shift_asym_y = datasource.shift_asym_y
            self.range_asym = datasource.range_asym

        else:
            self.bg_channels = np.zeros((1, 1, 4))
            self.bg_sem = 0
            self.bg_asym_x = 0
            self.bg_asym_y = 0

            self.shift_asym_x = 0
            self.shift_asym_y = 0
            self.range_asym = 0
            self.smooth = False
            self.smooth_sigma = 0.75
            self.norm = mpc.NoNorm()

        return self

    def center_asymmetry(self, manual=False, show_histogram=False):
        self.shift_asym_x = 0
        self.shift_asym_y = 0
        self.range_asym = 0
        self.norm = mpc.NoNorm()

        asym_x_fl = self.asym_x.flatten()
        asym_y_fl = self.asym_y.flatten()

        if manual or show_histogram:

            fig, ax = plt.subplots(1, 1)
            ax.set_title('2D asymmetry histogram')
            ax.set_xlabel('x-asymmetry')
            ax.set_ylabel('y-asymmetry')
            ax.set_aspect('equal')

            ax.plot(asym_x_fl, asym_y_fl, '.')

        if manual:
            NotImplementedError(
                "Manual asymmetry centring not yet implemented."
            )
        else:
            shift_asym_x = np.mean(asym_x_fl)
            shift_asym_y = np.mean(asym_y_fl)

            radii = np.sqrt((asym_x_fl - shift_asym_x)**2 +
                            (asym_y_fl - shift_asym_y)**2)

            asym_range = np.mean(radii) + 2 * np.std(radii)

        if show_histogram:
            circle = mpp.Circle((shift_asym_x, shift_asym_y), asym_range,
                                fill=False, ec='k', zorder=5)

            ax.add_patch(circle)

            percentile = np.sum(radii < asym_range) / len(asym_x_fl)
            ax.set_title('2D asymmetry histogram\n'
                         f'{percentile:.2%} of the data lies within the range')

        self.shift_asym_x = shift_asym_x
        self.shift_asym_y = shift_asym_y
        self.asym_range = asym_range
        self.norm = mpc.TwoSlopeNorm(0, vmin=-asym_range, vmax=asym_range)

        return self

    def process(self, smooth=False, sigma=None):
        """
        Convenience-method for calculating background-signal and centring the
        asymmetry. It also incorporates the possibility to turn on Gaussian
        smoothing after the background calculation.

        Parameters
        ----------
        smooth : bool, optional
            Indicate whether smoothing is turned on or off (default) before
            centring the asymmetry.
        sigma : bool, optional
            The standard deviation used for Gaussian smoothing if turned on.
            Default value is 0.75, and current value is not replaced when left
            empty.

        Returns
        -------
        SEMPA_Scan
            Contains itself after background calculation and asymmetry centring.

        """

        self.smooth = False
        self.calculate_background()

        self.smooth = smooth
        if sigma is not None and isinstance(sigma, (float, int)):
            self.smooth_sigma = sigma

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
                    "Shape of arrays to add do not match: "
                    f"{self.channels.shape} and {other.shape}"
                )
        elif np.isscalar(other):
            other = np.tile(other, self.channels.shape)
        else:
            raise TypeError(
                f"unsupported operand type(s) for {operation}:"
                f"'{type(self)}' and '{type(other)}'"
            )

        return other


class Info:
    pass
