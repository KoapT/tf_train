'''
write all the pictures which need to be trained to train.txt
'''

import os
import random
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float('r', 0.9,
                   'Ratio of train samples in all samples')

FLAGS = flags.FLAGS

CURRENT_DIR = os.path.abspath('.')

data_dir = CURRENT_DIR
image_dir = os.path.join(data_dir, 'JPEGImages')
traintxt_path = os.path.join(data_dir, 'Set', 'train.txt')
evaltxt_path = os.path.join(data_dir, 'Set', 'eval.txt')


# get all the .jpg file names in 'path', and put them into 'list_name'.
def listdir(path, list_name):
    for (root, dirs, files) in os.walk(path):
        for file in files:
            jpg_name, format_name = os.path.splitext(file)
            if format_name.lower() in ['.jpg', '.jpeg']:
                list_name.append(jpg_name)
    print('list length is: %d', len(list_name))
    print(list_name)


# write all the file names in 'listname' into 'txt_flie'.
def write_names_to_txt(listname, txt_file):
    ## check whether the txt_file is exited
    if os.path.exists(txt_file):
        print('txt file already exits, rewriting it.')
    else:
        os.mknod(txt_file)
        print('txt file does not exit, a new one has been created.')

    # begin to write
    fp = open(txt_file, 'w')
    for i in listname:
        fp.writelines(i + '\n')


if __name__ == '__main__':
    print('start to write file name to txt of sample set...')
    if FLAGS.r > 1.0:
        FLAGS.r = 1.0
    elif FLAGS.r < 0.0:
        FLAGS.r = 0.0

    names = list()
    names_train = list()
    names_eval = list()

    listdir(image_dir, names)

    count = len(names)
    count_train = int(FLAGS.r * float(count))

    mark = [True] * count
    index = list(range(0, count))
    index_train = random.sample(index, count_train)

    for i in range(0, len(index_train)):
        names_train.append(names[index_train[i]])
        mark[index_train[i]] = False

    for i in range(0, count):
        if mark[i]:
            names_eval.append(names[i])

    write_names_to_txt(names_train, traintxt_path)
    write_names_to_txt(names_eval, evaltxt_path)

    print(str(len(names_train)) + ' names for train are writed to ' + traintxt_path)
    print(str(len(names_eval)) + ' names for eval are writed to ' + evaltxt_path)
    print('write file name done!')
