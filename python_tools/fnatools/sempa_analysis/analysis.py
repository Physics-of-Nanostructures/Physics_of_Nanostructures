from .binaryFileReader import BinaryReader
from dataclasses import dataclass


@dataclass
class SEMPA_Measurement:
    """
    Class to import and process SEMPA measurements
    """

    filenames: {list, str}

    def __post_init__(self):
        if isinstance(self.filenames, str):
            self.filenames

        self.import_files()

    def import_files(self):
        for filename in self.filenames:
            self.read_file(filename)

    def read_file(self, file):
        with open(file, "rb") as file:
            print(type(file))
            if not file.read(8) == b"FLAT0100":
                raise ValueError("Unknown file-type")

            reader = BinaryReader(file)

            number_of_axes = reader.next_int()

            for i in range(number_of_axes):
                name = reader.next_string()
                name_parent = reader.next_string()
                name_units = reader.next_string()

                clock_count = reader.next_int()
                raw_start_val = reader.next_int()
                raw_increment_val = reader.next_int()

                physical_start_val = reader.next_double()
                physical_increment_val = reader.next_double()

                mirrored = reader.next_int()
                table_count = reader.next_int()

                if table_count != 0:
                    raise NotImplementedError(
                        "Non-zero table-count not yet implemented")

            channel_name = reader.next_string()
            transfer_function_name = reader.next_string()
            unit_name = reader.next_string()

            parameter_count = reader.next_int()
            parameters = {
                reader.next_string(): reader.next_double()
                for _ in range(parameter_count)
            }

            number_of_data_views = reader.next_int()
            data_view = [reader.next_int()
                         for _ in range(number_of_data_views)]

            time_stamp = datetime.datetime.fromtimestamp(reader.next_long())
            information = {
                info.split("=")[0]: info.split("=")[1]
                for info in reader.next_string().split(";")
            }

            data_points = reader.next_int()
            data_points_measured = reader.next_int()

            if data_points != data_points_measured:
                raise ValueError("Measurement not completed")

            raw_data = [
                reader.next_int()
                for _ in range(data_points)
            ]
