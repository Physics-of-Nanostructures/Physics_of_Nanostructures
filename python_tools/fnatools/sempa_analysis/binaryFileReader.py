from dataclasses import dataclass
import struct
import io

@dataclass
class BinaryReader:
    filebuffer: io.BufferedReader

    def next_string(self):
        number_of_chars = self.next_int()
        string = struct.unpack("cc" * number_of_chars,
                               self.filebuffer.read(2 * number_of_chars))
        string = b"".join(string)
        string = string.decode("utf-16")
        return string

    def next_int(self):
        value = struct.unpack("i", self.filebuffer.read(4))
        return value[0]

    def next_double(self):
        value = struct.unpack("d", self.filebuffer.read(8))
        return value[0]

    def next_long(self):
        value = struct.unpack("q", self.filebuffer.read(8))
        return value[0]