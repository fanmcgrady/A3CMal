import binascii
import os


class GenerateFeature():
    def __init__(self, path):
        # pe 文件名
        self.pe_path = path
        # 生成的bytes样本文件
        self.bytes_path = self.pe_path + '.bytes'
        self.file_name = 'sample_file'

        self.generate_bytes()

    def generate_bytes(self):
        list = []
        line = []
        with open(self.pe_path, 'rb') as cur_file:
            size = os.path.getsize(self.pe_path)  # 获得文件大小
            for i in range(size):
                data = cur_file.read(1)  # 每次输出一个字节
                hex_string = str.upper(binascii.b2a_hex(data).decode('ascii'))
                line.append(hex_string)

                if (i + 1) % 16 == 0:
                    # address = generate_address(count)
                    list.append("00000000 {}".format(" ".join(line)))
                    line = []

        with open(self.bytes_path, 'w') as out:
            out.write('\n'.join(list))

    def get_features(self):
        pass
