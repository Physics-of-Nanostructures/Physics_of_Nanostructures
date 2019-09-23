from .binaryFileReader import BinaryReader
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class SEMPA_Measurement:
    """
    Class to import and process SEMPA measurements
    """

    filenames: {list, str}

    def __post_init__(self):
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]

        self.import_files()

    def import_files(self):
        for filename in self.filenames:
            info, raw_data = self.read_file(filename)
            data = self.reshape_data(info, raw_data)

    @staticmethod
    def read_file(file):
        with open(file, "rb") as file:
            if not file.read(8) == b"FLAT0100":
                raise ValueError("Unknown file-type")

            reader = BinaryReader(file)

            info = dict()

            info["number_of_axes"] = reader.next_int()

            for i in range(info["number_of_axes"]):
                info[i] = dict()

                info[i]["name"] = reader.next_string()
                info[i]["name_parent"] = reader.next_string()
                info[i]["name_units"] = reader.next_string()

                info[i]["clock_count"] = reader.next_int()
                info[i]["raw_start_val"] = reader.next_int()
                info[i]["raw_increment_val"] = reader.next_int()

                info[i]["physical_start_val"] = reader.next_double()
                info[i]["physical_increment_val"] = reader.next_double()

                info[i]["mirrored"] = reader.next_int()
                info[i]["table_count"] = reader.next_int()

                if info[i]["table_count"] != 0:
                    raise NotImplementedError(
                        "Non-zero table-count not yet implemented")

            info["channel_name"] = reader.next_string()
            info["transfer_function_name"] = reader.next_string()
            info["unit_name"] = reader.next_string()

            info["parameter_count"] = reader.next_int()
            info["parameters"] = {
                reader.next_string(): reader.next_double()
                for _ in range(info["parameter_count"])
            }

            info["number_of_data_views"] = reader.next_int()
            info["data_view"] = [reader.next_int()
                                 for _ in range(info["number_of_data_views"])]

            info["time_stamp"] = datetime.fromtimestamp(
                reader.next_long())
            info["information"] = {
                info.split("=")[0]: info.split("=")[1]
                for info in reader.next_string().split(";")
            }

            info["data_points"] = reader.next_int()
            info["data_points_measured"] = reader.next_int()

            if info["data_points"] != info["data_points_measured"]:
                raise ValueError("Measurement not completed")

            raw_data = np.array([
                reader.next_int()
                for _ in range(info["data_points"])
            ])

        return info, raw_data

    @staticmethod
    def reshape_data(info, raw_data):
        x = info[1]["physical_start_val"] + np.arange(info[1]["clock_count"]) * info[1]["physical_increment_val"]
        y = info[3]["physical_start_val"] + np.arange(info[3]["clock_count"]) * info[3]["physical_increment_val"]

        X, Y = np.meshgrid(x, y)

        channel_data = raw_data.reshape((4, info[1]["clock_count"], info[3]["clock_count"]), order='F')
        channel_data = np.swapaxes(channel_data, 0, 2)
        channel_data = np.rot90(channel_data, 2)
        channel_data = np.rollaxis(channel_data, 2)

        data_asym12 = (channel_data[0] - channel_data[1]) / (channel_data[0] + channel_data[1])
        data_asym34 = (channel_data[2] - channel_data[3]) / (channel_data[2] + channel_data[3])
        data_asym = (channel_data[(0, 2)] - channel_data[(1, 3)]) / (channel_data[(0, 2)] + channel_data[(1, 3)])
        data_sem = np.sum(channel_data, 0)
        print(data_asym.shape)

        print(channel_data[0, :10, :5])


# %Setting up the matrix of coordinates;
# data_l.x=linspace(axis(2).physical_start_val,(axis(2).clock_count-1)*axis(2).physical_increment_val,axis(2).clock_count);
# data_l.x=repmat(data_l.x,axis(4).clock_count,1);

# data_l.y=linspace(axis(4).physical_start_val,(axis(4).clock_count-1)*axis(4).physical_increment_val,axis(4).clock_count);
# data_l.y=repmat(data_l.y',1,axis(2).clock_count);


# %Extracting the raw channeltron data
# data_l.channel_1=raw_data(1:4:end);
# data_l.channel_2=raw_data(2:4:end);
# data_l.channel_3=raw_data(3:4:end);
# data_l.channel_4=raw_data(4:4:end);


# data_l.channel_1=reshape(data_l.channel_1,[axis(2).clock_count axis(4).clock_count])';
# data_l.channel_2=reshape(data_l.channel_2,[axis(2).clock_count axis(4).clock_count])';
# data_l.channel_3=reshape(data_l.channel_3,[axis(2).clock_count axis(4).clock_count])';
# data_l.channel_4=reshape(data_l.channel_4,[axis(2).clock_count axis(4).clock_count])';

# %Have to rotate all by 180 degrees because with a sample rotation in the
# %SmartSEM of 210 it doesn't match to the SEMPA axis. Now,
# %imagesc(data_1.channel_1) gives the correct rotation/orientation with
# %respect to up/down left/right magnetization.
# data_l.channel_1=rot90(data_l.channel_1,2);
# data_l.channel_2=rot90(data_l.channel_2,2);
# data_l.channel_3=rot90(data_l.channel_3,2);
# data_l.channel_4=rot90(data_l.channel_4,2);

# data_l.asym12=(data_l.channel_1-data_l.channel_2)./(data_l.channel_1+data_l.channel_2);
# data_l.asym34=(data_l.channel_3-data_l.channel_4)./(data_l.channel_3+data_l.channel_4);
# data_l.sem=(data_l.channel_1+data_l.channel_2+data_l.channel_3+data_l.channel_4);
