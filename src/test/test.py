import binascii
import os

list = []
line = []
with open('java.exe', 'rb') as cur_file:
    size = os.path.getsize('java.exe')  # 获得文件大小
    for i in range(size):
        data = cur_file.read(1)  # 每次输出一个字节
        hex_string = str.upper(binascii.b2a_hex(data).decode('ascii'))
        line.append(hex_string)

        if (i + 1) % 16 == 0:
            list.append("00000000 {}".format(" ".join(line)))
            line = []
print('\n'.join(list))

# with open('java.bytes', 'w') as out:
#     out.write('\n'.join(list))