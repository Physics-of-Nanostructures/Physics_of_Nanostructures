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
        for filename in filenames:
            self.import_file(filename)

    def import_file(self, file):

        with open(file, 'rb'):