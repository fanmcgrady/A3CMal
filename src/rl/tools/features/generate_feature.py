class GenerateFeature():
    def __init__(self, bytez):
        # pe 文件名
        self.bytez = bytez
        self.bytes_list = []
        self.file_name = 'tmp.bytes'

        self.generate_bytes()

    def generate_bytes(self):
        line = []
        size = len(self.bytez)  # 获得文件大小
        for i in range(size):
            data = self.bytez[i]  # 每次输出一个字节
            data = hex(data)  # 变成ascii
            data = str(data)[2:]
            hex_string = str.upper(data)
            if len(hex_string) == 1:
                hex_string = '0' + hex_string
            line.append(hex_string)

            if (i + 1) % 16 == 0:
                # address = generate_address(count)
                self.bytes_list.append("00000000 {}".format(" ".join(line)))
                line = []

        # 写到文件中，但是只有image_fea的时候用到
        with open(self.file_name, 'w') as f:
            f.write("\n".join(self.bytes_list))

    def get_features(self):
        pass
