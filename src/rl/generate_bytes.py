import binascii
import os

list = []
line = []
with open('KMSELDI.exe', 'rb') as cur_file:
    size = os.path.getsize('KMSELDI.exe')  # 获得文件大小
    for i in range(size):
        data = cur_file.read(1)  # 每次输出一个字节
        hex_string = str.upper(binascii.b2a_hex(data).decode('ascii'))
        line.append(hex_string)

        if (i + 1) % 16 == 0:
            # address = generate_address(count)
            list.append("00000000 {}".format(" ".join(line)))
            line = []

with open('KMSELDI.bytes', 'w') as out:
    out.write('\n'.join(list))

def generate_address(count):
    pass