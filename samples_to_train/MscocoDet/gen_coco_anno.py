import os
import json
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('anno_dir', 'Annotations',
                    'VOC format annotation path')

flags.DEFINE_string('set_file', 'Set/eval.txt',
                    'Eval dataset txt file.')

flags.DEFINE_string('label_map_path', 'LabelMap/label_map.json',
                    'Label map path, json file.')

flags.DEFINE_string('out_coco_anno_path', 'cocoAnno/annotations_eval.json',
                    'Output coco anno path,json file.')

anno_dir = FLAGS.anno_dir
set_file = FLAGS.set_file
label_map_path = FLAGS.label_map_path
out_coco_anno_path = FLAGS.out_coco_anno_path

coco_annotaton = {}
images_list = []
annotations_list = []
categories_list = []

with open(set_file, 'r') as f:
    file_list = list(map(lambda x: x.strip(), f.readlines()))

with open(label_map_path, 'r') as fp:
    label_map = json.load(fp)

categories_list = label_map
coco_annotaton['categories'] = categories_list
categories_dict = {cate['name']: cate['id'] for cate in categories_list}


def recursive_parse_xml_to_dict(xml):
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


for fname in file_list:
    print(fname)
    try:
        with open(os.path.join(anno_dir, fname + '.xml'), 'r') as fid:
            xml_root = ET.parse(fid).getroot()
            root_anno = recursive_parse_xml_to_dict(xml_root)['annotation']
            try:
                filename = root_anno['filename']
                size = root_anno['size']
            except:
                continue
            img_id = len(images_list)
            images_list.append({'id': img_id,
                                'file_name': os.path.split(filename)[-1],
                                'width': int(size['width']),
                                'height': int(size['height'])})
            try:
                objects = root_anno['object']
            except:
                continue
            for obj in objects:
                bndbox = obj['bndbox']
                x = float(bndbox['xmin'])
                y = float(bndbox['ymin'])
                w = float(bndbox['xmax']) - float(bndbox['xmin'])
                h = float(bndbox['ymax']) - float(bndbox['ymin'])
                annotations_list.append({'id': len(annotations_list),
                                         'image_id': img_id,
                                         'category_id': categories_dict[obj['name']],
                                         'bbox': [x, y, w, h],
                                         'iscrowd': 0,
                                         'area': w * h * np.pi * .25})
    except FileNotFoundError:
        print('No file \'%s\' found!' % os.path.join(anno_dir, fname + '.xml'))
coco_annotaton['images'] = images_list
coco_annotaton['annotations'] = annotations_list

with open(out_coco_anno_path, 'w') as fw:
    json.dump(coco_annotaton, fw)
