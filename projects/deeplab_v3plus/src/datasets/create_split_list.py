'''
write all the pictures which need to be trained to train.txt
'''

import os
import random
import tensorflow as tf




flags = tf.app.flags
flags.DEFINE_float('train_split_ratio', 0.9, 
                   'Ratio of train split in all, <=1.0 and >=0.0')
flags.DEFINE_string('label_dir', 'Annotations',
                    'Label directory')
flags.DEFINE_enum('label_format', 'png', ['xml', 'png', 'jpeg', 'jpg'],
                  'Label format.')
flags.DEFINE_string('output_train_split_list_path', 'Set/train.txt',
                    'Training split list file path of output')
flags.DEFINE_string('output_eval_split_list_path', 'Set/eval.txt',
                    'Evaluation split list file path of output')

FLAGS = flags.FLAGS




#get all the file names in 'path', and put them into 'list_name'.
def listdir(directory, nameList):
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            listdir(path, nameList)
        elif os.path.splitext(path)[1].lower().strip('.') == FLAGS.label_format:
            nameList.append(os.path.splitext(os.path.basename(path))[0])

#write all the file names in 'name_list' into 'txt_flie'.
def write_names_to_txt(name_list, txt_file):
    ## check whether the txt_file is exited
    if os.path.exists(txt_file):
        print ('txt file already exits, rewriting it.')
    else:
        os.mknod(txt_file)
        print ('txt file does not exit, a new one has been created.')
    #begin to write
    fp = open(txt_file,'w')
    for name in name_list:
        fp.writelines(name + '\n')
	

if __name__ == '__main__':
    print('start to write file name to txt of sample set...')
    if FLAGS.train_split_ratio > 1.0:
        FLAGS.train_split_ratio = 1.0
    elif FLAGS.train_split_ratio < 0.0:
        FLAGS.train_split_ratio = 0.0

    names = list()
    names_train = list()
    names_eval = list()

    listdir(FLAGS.label_dir, names)

    count = len(names)
    count_train = int(FLAGS.train_split_ratio * float(count))

    mark = [True] * count
    index = list(range(0, count))
    index_train = random.sample(index, count_train)

    for i in range(0, len(index_train)):
        names_train.append(names[index_train[i]])
        mark[index_train[i]] = False

    for i in range(0, count):
        if mark[i]:
            names_eval.append(names[i])

    write_names_to_txt(names_train, FLAGS.output_train_split_list_path)
    write_names_to_txt(names_eval, FLAGS.output_eval_split_list_path)

    print(str(len(names_train)) + ' names for train are writed to ' + FLAGS.output_train_split_list_path)
    print(str(len(names_eval)) + ' names for eval are writed to ' + FLAGS.output_eval_split_list_path)
    print('write file name done!')

