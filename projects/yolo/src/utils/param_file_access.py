import numpy as np
import json
from lxml import etree
import os
import random
from PIL import Image
import tensorflow as tf

IMAGE_FORMAT = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'JPG': 'jpeg',
    'jpeg': 'jpeg',
    'JPEG': 'jpeg',
    'png': 'png',
    'PNG': 'png',
}


def list_all_filepaths(directory):
    path_list = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            path_list.extend(list_all_filepaths(path))
        if os.path.isfile(path):
            path_list.append(path)
    return path_list


def get_json_params(file_path):
    params = []
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params


def get_category_index(file_path, key_name='name'):
    with open(file_path, 'r') as f:
        params = json.load(f)
    category_index = {i: {'name': cat[key_name]} for i, cat in enumerate(params)}
    return category_index


def put_json_params(file_path, content):
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(content, f)


def get_txt_params(file_path, splitStr=' '):
    params = []
    with open(file_path, 'r') as f:
        for line in f:
            params.extend(line.strip().split(splitStr))
    return params


def put_txt_params(file_path, str_list):
    ## check whether the file is exited
    if os.path.exists(file_path):
        print('txt file already exits, rewriting it.')
    else:
        os.mknod(file_path)
        print('txt file does not exit, a new one has been created.')
    # begin to write
    with open(file_path, 'w') as fp:
        for s in str_list:
            fp.writelines(s + '\n')


def get_label_map_palette(label_map_path):
    labels = get_json_params(label_map_path)
    palette = np.array([[0, 0, 0] for i in range(256)]).astype(np.uint8)
    for label in labels:
        palette[label['id'], 0] = label['vis']['r']
        palette[label['id'], 1] = label['vis']['g']
        palette[label['id'], 2] = label['vis']['b']
    return palette


def get_label_map_value_list_by_key(label_map_path, key='name'):
    value_list = []
    items = get_json_params(label_map_path)
    for item in items:
        value_list.append(item[key])
    return value_list


def get_label_map_class_id_list(label_map_path):
    class_id_list = []
    labels = get_json_params(label_map_path)
    for label in labels:
        class_id_list.append(label['id'])
    return class_id_list


def get_label_map_min_id(label_map_path):
    class_id_list = get_label_map_class_id_list(label_map_path)
    class_id_list.sort()
    return class_id_list[0]


def get_label_map_max_id(label_map_path):
    class_id_list = get_label_map_class_id_list(label_map_path)
    max_id = class_id_list[0]
    for i in range(1, len(class_id_list)):
        if class_id_list[i] > max_id:
            max_id = class_id_list[i]
    return max_id


def get_label_map_class_id_name_dict(label_map_path, key_name='name'):
    id_name_dict = {}
    labels = get_json_params(label_map_path)
    for label in labels:
        id_name_dict[label['id']] = label[key_name]
    return id_name_dict


def recursive_parse_xml_to_dict(xml, multi_appear_keys):
    """Recursively parses XML contents to python dict.
      Args:
          xml: xml tree obtained by parsing XML file contents using lxml.etree
          multi_appear_keys: the tags appeared multiple times at the same level of a tree
      Returns:
          Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child, multi_appear_keys)
        if child.tag not in multi_appear_keys:
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def get_xml_params(xml_path, multi_appear_keys):
    with open(xml_path, 'r') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    return recursive_parse_xml_to_dict(xml, multi_appear_keys)


def random_list_split_traineval(name_list, rate):
    train_name_list = []
    eval_name_list = []
    count = len(name_list)
    train_count = int(rate * float(count))
    mark = [True] * count
    index = list(range(0, count))
    train_index = random.sample(index, train_count)
    for i in range(0, train_count):
        train_name_list.append(name_list[train_index[i]])
        mark[train_index[i]] = False
    for i in range(0, count):
        if mark[index[i]]:
            eval_name_list.append(name_list[index[i]])
    return train_name_list, eval_name_list


def create_traineval_split_list(data_dir, data_format, rate,
                                train_list_output_path, eval_list_output_path):
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0

    file_name_list = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isdir(file_path):
            continue
        file_name_base = os.path.splitext(file_name)[0]
        file_name_ext = os.path.splitext(file_name)[-1].split('.')[-1]
        if IMAGE_FORMAT_MAP[file_name_ext.lower()] == IMAGE_FORMAT_MAP[data_format.lower()]:
            file_name_list.append(file_name_base)
    train_file_name_list, eval_file_name_list = random_list_split_traineval(file_name_list, rate)
    put_txt_params(train_list_output_path, train_file_name_list)
    put_txt_params(eval_list_output_path, eval_file_name_list)


def remove_annotation_color(anno_file_path):
    return np.array(Image.open(anno_file_path))


def save_annotation(annotation, save_path):
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.gfile.Open(save_path, mode='w') as f:
        pil_image.save(f, 'PNG')
