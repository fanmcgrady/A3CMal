import time
from csv import writer

import numpy

from feature_extraction import *
from header_construction import *
from settings import *

# path = DATASET_PATH
# os.chdir(path)


# Hex dump-based features
def byte_extraction(dataset_type):
    directory_name = os.path.join(DATASET_PATH, dataset_type) + '/'
    files = os.listdir(directory_name)
    files = numpy.sort(files)
    byte_files = [i for i in files if i.endswith('.bytes')]
    # byte_csv = dataset_type + '_byte_csv.gz'

    # if not os.path.isdir(SAVED_PATH_CSV + dataset_type):
    #    os.makedirs(SAVED_PATH_CSV + dataset_type)

    oneg_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_oneg.csv', 'w')
    m_data_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_meta_data.csv', 'w')
    img1_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_img1.csv', 'w')
    # img2_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_img2.csv', 'w')
    entropy_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_entropy.csv', 'w')
    str_lengths_csv = open(SAVED_PATH_CSV + dataset_type + '/byte_str_lengths.csv', 'w')

    oneg_time = []
    twog_time = []
    m_data_time = []
    img1_time = []
    # img2_time = []
    entropy_time = []
    str_lengths_time = []

    # with gzip.open(byte_csv, 'w') as f:

    # Header construction
    # fw = writer(f)
    colnames = ['filename']
    colnames += header_byte_1gram()
    # colnames += header_byte_2grams()
    colnames += header_byte_meta_data()
    colnames += header_byte_img1()
    # colnames += header_byte_img2()
    colnames += header_byte_entropy()
    colnames += header_byte_str_len()
    # fw.writerow(colnames)

    meta_data_colnames = header_byte_meta_data()
    onegram_colnames = header_byte_1gram()
    img1_colnames = header_byte_img1()
    # img2_colnames = header_byte_img2()
    entropy_colnames = header_byte_entropy()
    str_len_colnames = header_byte_str_len()

    m_data_csv_w = writer(m_data_csv)
    m_data_csv_w.writerows([meta_data_colnames])
    oneg_csv_w = writer(oneg_csv)
    oneg_csv_w.writerows([onegram_colnames])
    img1_csv_w = writer(img1_csv)
    img1_csv_w.writerows([img1_colnames])
    # img2_csv_w = writer(img2_csv)
    # img2_csv_w.writerows([img2_colnames])
    entropy_csv_w = writer(entropy_csv)
    entropy_csv_w.writerows([entropy_colnames])
    str_lengths_csv_w = writer(str_lengths_csv)
    str_lengths_csv_w.writerows([str_len_colnames])

    # Creating row set
    rows = []
    for t, fname in enumerate(byte_files):
        with open(directory_name + fname, 'r') as f:
            # One Gram
            start_time = time.time()
            oneg = byte_1gram(f)
            required_time = time.time() - start_time
            oneg_time.append(required_time)
            oneg_csv_w.writerows([oneg])

            f.seek(0)

            # Two grams
            start_time = time.time()
            # TwoByte = byte_2gram(f)
            required_time = time.time() - start_time
            twog_time.append(required_time)

            f.seek(0)

            # Meta data
            start_time = time.time()
            meta_data = byte_meta_data(directory_name + fname, f)
            required_time = time.time() - start_time
            m_data_time.append(required_time)
            m_data_csv_w.writerows([meta_data])

            f.seek(0)

            # Images 1
            start_time = time.time()
            image1 = byte_image1(f)
            required_time = time.time() - start_time
            img1_time.append(required_time)
            img1_csv_w.writerows([image1])

            f.seek(0)

            # Images 2
            # start_time = time.time()
            # # image2 = byte_image2(f)
            # required_time = time.time() - start_time
            # img2_time.append(required_time)
            # # img2_csv_w.writerows([image2])
            #
            # f.seek(0)

            # Entropy
            start_time = time.time()
            entropy = byte_entropy(f)
            required_time = time.time() - start_time
            entropy_time.append(required_time)
            entropy_csv_w.writerows([entropy])

            f.seek(0)

            # Strings
            start_time = time.time()
            str_lengths = byte_string_lengths(f)
            required_time = time.time() - start_time
            str_lengths_time.append(required_time)
            str_lengths_csv_w.writerows([str_lengths])

            # Row added
            # whole = oneg + meta_data + image1 + image2 + entropy + str_lengths
            # rows.append([fname[:fname.find('.bytes.gz')]] + whole)

            # Writing rows after every 100 files processed
            if (t + 1) % 10 == 0:
                print(t + 1, 'byte files loaded from ', dataset_type)
                # fw.writerows(rows)
                rows = []
                # break

            # if (t + 1) % 50 == 0:
            #     break

    # Save the time
#   oneg_time_str = ', '.join(str(x) for x in oneg_time)
#   twog_time_str = ', '.join(str(x) for x in twog_time)
#   m_data_time_str = ', '.join(str(x) for x in m_data_time)
#   img1_time_str = ', '.join(str(x) for x in img1_time)
    # img2_time_str = ', '.join(str(x) for x in img2_time)
#   ent_time_str = ', '.join(str(x) for x in entropy_time)
#   str_lengths_time_str = ', '.join(str(x) for x in str_lengths_time)
#   io.save_txt(('one_gram_' + dataset_type, oneg_time_str, 'two_gram_' + dataset_type, twog_time_str,
#                'bc_meta_data_' + dataset_type, m_data_time_str, 'img1_' + dataset_type, img1_time_str,
#                'ent_' + dataset_type, ent_time_str,
#                'str_len_' + dataset_type, str_lengths_time_str,
#                ), BYTE_TIME_PATH)
