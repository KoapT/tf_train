import os
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import json

from yolo.src.utils import param_file_access, postprocess_utils,preprocess_utils
from yolo.src import models_rigister

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('export_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/model_608_eval.pb',
                    'Path to output Tensorflow frozen graph.')
flags.DEFINE_string('model_info_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/config.json',
                    'Model information file path.')
flags.DEFINE_string('sample_info_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/LabelMap/label_map.json',
                    'Sample information file path.')
flags.DEFINE_string('ckpt_dir',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/train_dir',
                    'Training log file path.')
flags.DEFINE_string('log_dir',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/train_dir',
                    'Training log file path.')
flags.DEFINE_string('coco_ann_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/cocoAnno/annotations_eval.json',
                    'COCO format annotation file path.')
flags.DEFINE_string('image_dir',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/JPEGImages',
                    'Evaluation dataset file path.')
flags.DEFINE_bool('coco_id',
                  True,
                  'Is coco dataset or not.')

EXTS = ['.jpg', '.png', '.jpeg', '.bmp']


def is_img(fileName):
    ext = os.path.splitext(fileName)[-1].lower()
    if ext in EXTS:
        return True
    else:
        return False


def main(argv):
    pb_path = FLAGS.export_path
    model_info_path = FLAGS.model_info_path
    sample_info_path = FLAGS.sample_info_path
    ckpt_dir = FLAGS.ckpt_dir
    annFile = FLAGS.coco_ann_path
    image_dir = FLAGS.image_dir
    coco_id = FLAGS.coco_id

    # 读取最新的ckpt文件的名字以获取当前global_step
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = int(ckpt_name.split('-')[-1])
    except:
        print('No ckpt file found in %s!' % ckpt_dir)
        raise NameError

    model_cfg = param_file_access.get_json_params(model_info_path)
    model_name = model_cfg['model_name']
    model_input_size = model_cfg['model_config']['input_size']
    batch_size = model_cfg['eval_config']['batch_size']
    model = models_rigister.get_model(model_name)
    model.initialize(model_info_path,sample_info_path,False)
    key_name = model_cfg['sample_config']['key_name']
    cls_names = param_file_access.get_label_map_class_id_name_dict(sample_info_path, key_name)

    cocoGt = COCO(annFile)
    cls_dict = {id: cocoGt.cats[id]['name'] for id in cocoGt.cats.keys()}
    filename_to_id_dict = {cocoGt.imgs[id]['file_name']: id for id in cocoGt.imgs.keys()}
    num_images = len(filename_to_id_dict)

    print('step:%d' % global_step)
    print('%d images for reference: -------------------------' % num_images)
    print('class dict:', cls_dict)
    # print(filename_list)
    print('-------------------------------------')

    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=session_config) as sess:
        saver = tf.train.import_meta_graph(
            '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/model_608_eval.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection'))
        # saver = tf.train.import_meta_graph(
        #     '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/train_dir/model.ckpt-972663.meta')
        # saver.restore(sess, tf.train.latest_checkpoint(
        #     '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/train_dir'))

        graph = tf.get_default_graph()


        result_list = []
        batch_images = []
        batch_bboxes = []
        batch_categories = []
        batch_bbox_numbers = []
        image_names = []
        image_heights = []
        image_widths = []
        count = 0
        for imageName in filename_to_id_dict.keys():
        # for imageName in ['000000397133.jpg']:
            print(imageName)
            count += 1
            bboxes = []
            categories = []
            bbox_numbers = 0
            print('\r%d' % count, end='')
            if is_img(imageName):
                imagePath = os.path.join(image_dir, imageName)
            else:
                imagePaths = glob(os.path.join(image_dir, imageName + '.*'))
                imagePath = ''
                for ip in imagePaths:
                    if is_img(ip):
                        imagePath = ip
                if imagePath == '':
                    continue
            try:
                image_bgr = cv2.imread(imagePath)
            except:
                print('Path wrong:%s' % imagePath)
                continue
            image_id = filename_to_id_dict[imageName]
            for ann in cocoGt.imgToAnns[image_id]:
                bboxes.append(ann['bbox'])
                category_id = ann['category_id']
                if coco_id:
                    COCO_ID_REVERSE = {ID:i for i,ID in postprocess_utils.COCO_ID.items()}
                    category_id = COCO_ID_REVERSE[category_id]
                categories.append(category_id)
                bbox_numbers += 1
            if len(bboxes)==0:
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            image = cv2.resize(image, (model_input_size, model_input_size), cv2.INTER_LINEAR)
            bboxes = np.array(bboxes)
            bboxes[...,:2] = bboxes[...,:2]+.5*bboxes[...,2:]
            bboxes[...,0] = bboxes[...,0]/width*model_input_size
            bboxes[...,1] = bboxes[...,1] / height * model_input_size
            bboxes[...,2] = bboxes[...,2] / width * model_input_size
            bboxes[...,3] = bboxes[...,3] / height * model_input_size

            batch_images.append(image)
            batch_bboxes.append(bboxes)
            batch_categories.append(categories)
            batch_bbox_numbers.append(bbox_numbers)
            image_names.append(imageName)
            image_heights.append(height)
            image_widths.append(width)
            if len(batch_images) < batch_size and count < num_images:
                continue
            images = np.array(batch_images)
            input_bbox = np.array(batch_bboxes)
            input_box_category = np.array(batch_categories)
            input_box_numbers = np.array(batch_bbox_numbers)
            print('input_bbox',input_bbox)
            print('input_box_category',input_box_category)
            print('input_box_numbers',input_box_numbers)

            # Run inference

            input_images = graph.get_tensor_by_name('input_image:0')
            input_bbox_tensor = graph.get_tensor_by_name('input_bbox:0')
            input_box_category_tensor = graph.get_tensor_by_name('input_box_category:0')
            input_box_numbers_tensor = graph.get_tensor_by_name('input_box_numbers:0')
            loss = graph.get_tensor_by_name('loss:0')
            loss1 = graph.get_tensor_by_name('loss1:0')
            model_output = graph.get_tensor_by_name('output_label:0')
            label_sbbox, label_mbbox, label_lbbox = tf.py_func(preprocess_utils.gen_ground_truth_np,
                                                               inp=[input_bbox_tensor[0],
                                                                    input_box_category_tensor[0],
                                                                    input_box_numbers_tensor[0],
                                                                    model._input_size,
                                                                    model._strides,
                                                                    model._anchors,
                                                                    model._anchor_per_scale,
                                                                    model._num_classes],
                                                               Tout=[tf.float32, tf.float32, tf.float32],
                                                               name='pyfunc_genGT')


            loss,loss1 = sess.run([loss,loss1], feed_dict={input_images: images,
                                             input_bbox_tensor: input_bbox,
                                             input_box_category_tensor: input_box_category,
                                             input_box_numbers_tensor: input_box_numbers})
            print('loss:', loss)
            print('loss1:',loss1)


            logits_np = sess.run(model_output, {input_images: images})


            # Show the image with bboxes
            t = time.time()
            results = postprocess_utils.nms_np(logits_np, iou_threshold=0.45,
                                               confidence_threshold=0.3, coco_id=coco_id,
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

            batch_images = []
            batch_bboxes = []
            batch_categories = []
            batch_bbox_numbers = []
            image_names = []
            image_heights = []
            image_widths = []
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
