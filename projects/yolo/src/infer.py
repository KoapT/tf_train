import os
import numpy as np
import tensorflow as tf
import cv2
import time

from yolo.src.utils import param_file_access, postprocess_utils
from yolo.src import models_rigister

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('images_dir', '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/JPEGImages',
                    'Images directory')

flags.DEFINE_string('pb_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/model_608.pb',
                    'Check point path')

flags.DEFINE_string('model_info_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/config.json',
                    'Model information file path.')

flags.DEFINE_string('sample_info_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/LabelMap/label_map.json',
                    'Sample information file path.')
flags.DEFINE_bool('coco_id',
                  True,
                  'Is coco dataset or not.')
flags.DEFINE_integer('batch_size',
                     10,
                     'Image batch size for inference.')

EXTS = ['.jpg', '.png', '.jpeg', '.bmp']


def is_img(fileName):
    ext = os.path.splitext(fileName)[1].lower()
    if ext in EXTS:
        return True
    else:
        return False


if __name__ == '__main__':
    pb_path = FLAGS.pb_path
    model_info_path = FLAGS.model_info_path
    sample_info_path = FLAGS.sample_info_path
    images_dir = FLAGS.images_dir
    coco_id = FLAGS.coco_id
    batch_size = FLAGS.batch_size

    model_cfg = param_file_access.get_json_params(model_info_path)
    key_name = model_cfg['sample_config']['key_name']
    cls_names = param_file_access.get_label_map_class_id_name_dict(sample_info_path, key_name)
    print(cls_names)

    imagePathList = []
    for imageFileName in os.listdir(images_dir):
        if is_img(imageFileName):
            imagePathList.append(os.path.join(images_dir, imageFileName))
    num_images = len(imagePathList)
    print('%d images for reference: -------------------------' % num_images)
    # print(imagePathList)
    print('-------------------------------------')

    model_name = model_cfg['model_name']
    model_input_size = model_cfg['model_config']['input_size']
    model = models_rigister.get_model(model_name)
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        with tf.Session() as sess:
            batch_images = []
            count = 0
            for imagePath in imagePathList:
                count += 1
                print('Process: {}'.format(imagePath))
                try:
                    image_bgr = cv2.imread(imagePath)
                except:
                    print('Path wrong:%s' % imagePath)
                    continue
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (model_input_size, model_input_size), cv2.INTER_LINEAR)
                batch_images.append(image)
                if len(batch_images) < batch_size and count < num_images:
                    continue
                images = np.array(batch_images)

                # Run inference
                logits_np = model.inference(graph, sess, feeds=images)  # xywh

                # Show the image with bboxes
                t = time.time()
                results = postprocess_utils.nms_np(logits_np, iou_threshold=0.3,
                                                   confidence_threshold=0.5, coco_id=coco_id,
                                                   iou_method='diou')  # xywh
                print('nms cost time:', time.time() - t)

                for i, result in enumerate(results):
                    postprocess_utils.draw_boxes_cv2(result, images[i], cls_names, model_input_size)
                    cv2.imshow('image', images[i][..., ::-1])
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
                batch_images = []
        cv2.destroyAllWindows()
