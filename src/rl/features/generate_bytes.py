import binascii
import os


def generate_bytes(file_name, saved_path):
    real_name = file_name.split('/')[-1]
    bytes_name = real_name + '.bytes'

    list = []
    line = []
    with open(file_name, 'rb') as cur_file:
        size = os.path.getsize(file_name)  # 获得文件大小
        for i in range(size):
            data = cur_file.read(1)  # 每次输出一个字节
            hex_string = str.upper(binascii.b2a_hex(data).decode('ascii'))
            line.append(hex_string)

            if (i + 1) % 16 == 0:
                # address = generate_address(count)
                list.append("00000000 {}".format(" ".join(line)))
                line = []

    with open(os.path.join(saved_path, bytes_name), 'w') as out:
        out.write('\n'.join(list))

if __name__ == '__main__':
    pe = '../../../Dataset/pe'
    train = '../../../Dataset/train'
    label_file = '../../../Dataset/trainLabels.csv'

    with open(label_file, 'w') as csv:
        csv.write('"Id","Class"\n')
        fold_list = os.listdir(pe)
        for i, fold in enumerate(fold_list):
            files = os.listdir(os.path.join(pe, fold))
            for j, f in enumerate(files):
                csv.write('"{}",{}\n'.format(f, i))
                generate_bytes(os.path.join(os.path.join(pe, fold), f), train)
                print("{}: processing {}".format(j, f))