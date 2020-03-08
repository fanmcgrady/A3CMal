# 使用os.walk方法遍历：
import os
import shutil


# 获取文件的大小,结果保留两位小数，单位为MB
def get_file_size(path):
    fsize = os.path.getsize(path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


# 寻找benign文件
def find_benign():
    count = 0
    path = "C:\\"
    copy_path = "C:\\benign"
    for dirpath, dirnames, filenames in os.walk(path):
        # 找到符合条件的文件，拷贝
        for file in filenames:
            try:
                extension = os.path.splitext(file)[1]
                full_path = os.path.join(dirpath, file)
                dest_path = os.path.join(copy_path, file)
                if extension == '.exe' or extension == '.dll' or extension == '.sys' or extension == '.com':
                    if get_file_size(full_path) <= 1:
                        shutil.copyfile(full_path, dest_path)
                        print("{}.{}".format(count, file))
                        count += 1
            except:
                continue
    return count


if __name__ == '__main__':
    print('Find benign size:{}'.format(find_benign()))
