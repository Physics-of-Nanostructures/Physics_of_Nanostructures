from dataclasses import dataclass
from typing import List
from pathlib import Path
from pandas import read_csv


@dataclass
class DataSet:
    folder: str
    file_format: str

    delimiter: str = ""
    comment: str = ""
    # starts_of_data: List[str] = ["[Data]", "#Data:"]

    def __post_init__(self):
        self.path = Path(self.folder)
        self.files = self.path.glob(self.file_format)

        self.import_data_table()

    def import_data_table(self):
        dataset = []
        metadataset = []

        for file in self.files:
            metadata, skiplines = self.extract_metadata(file)
            data = self.extract_data(file, skiplines)

    def extract_data(self, file, skiplines=None):
        data = read_csv(file,
                        delimiter=self.delimiter,
                        comment=self.comment,
                        skiprows=skiplines,
                        )
        return data

    def extract_metadata(self, file):

        metadata = {}

        with open(file, "r") as f:
            line_idx = 0
            for line in f:
                line_idx += 1
                line = line.strip()

                # Check if Data section begins
                if line in self.starts_of_data:
                    break

                # Extract info from preamble
                if line.startswith("INFO"):
                    info = line[5:].rsplit(",", maxsplit=1)
                    metadata[info[1]] = info[0]
