import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
from PIL import Image
import cv2
from glob import glob
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from deepin.src.utils import param_file_access, postprocess_utils
from deepin.src import models_rigister

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('export_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/deepin/run/pretrain_models/Object_Detection/saved_model/model_608.pb',
                    'Path to output Tensorflow frozen graph.')
flags.DEFINE_string('model_info_path',
                    '/home/psdz/TK/tensorFlowTrain/projects/deepin/run/pretrain_models/Object_Detection/config.json',
                    'Model information file path.')
flags.DEFINE_string('ckpt_dir',
                    '/home/psdz/TK/tensorFlowTrain/projects/deepin/run/pretrain_models/Object_Detection/logs',
                    'Training log file path.')
flags.DEFINE_string('log_dir',
                    '/home/psdz/TK/tensorFlowTrain/projects/deepin/run/pretrain_models/Object_Detection/logs/log_eval',
                    'Training log file path.')
flags.DEFINE_string('coco_ann_path',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/ObjectDetectExample/cocoAnno/annotations_eval.json',
                    'COCO format annotation file path.')
flags.DEFINE_string('image_dir',
                    '/home/psdz/TK/tensorFlowTrain/samples_to_train/ObjectDetectExample/JPEGImages/',
                    'Evaluation dataset file path.')
flags.DEFINE_bool('coco_id',
                  False,
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

    cocoGt = COCO(annFile)
    cls_dict = {id: cocoGt.cats[id]['name'] for id in cocoGt.cats.keys()}
    filename_to_id_dict = {cocoGt.imgs[id]['file_name']: id for id in cocoGt.imgs.keys()}

    print('step:%d' % global_step)
    print('%d images for reference: -------------------------' % (len(filename_to_id_dict)))
    print('class dict:', cls_dict)
    # print(filename_list)
    print('-------------------------------------')

    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        ap = tf.placeholder(tf.float32, shape=[])
        ap50 = tf.placeholder(tf.float32, shape=[])
        ap75 = tf.placeholder(tf.float32, shape=[])
        aps = tf.placeholder(tf.float32, shape=[])
        apm = tf.placeholder(tf.float32, shape=[])
        apl = tf.placeholder(tf.float32, shape=[])
        ar1 = tf.placeholder(tf.float32, shape=[])
        ar10 = tf.placeholder(tf.float32, shape=[])
        ar100 = tf.placeholder(tf.float32, shape=[])
        ars = tf.placeholder(tf.float32, shape=[])
        arm = tf.placeholder(tf.float32, shape=[])
        arl = tf.placeholder(tf.float32, shape=[])

        tf.summary.scalar("AP/IoU=0.50:0.95|area=all|maxDets=100", ap)
        tf.summary.scalar("AP/IoU=0.50|area=all|maxDets=100", ap50)
        tf.summary.scalar("AP/IoU=0.75|area=all|maxDets=100", ap75)
        tf.summary.scalar("AP/IoU=0.50:0.95|area=small|maxDets=100", aps)
        tf.summary.scalar("AP/IoU=0.50:0.95|area=medium|maxDets=100", apm)
        tf.summary.scalar("AP/IoU=0.50:0.95|area=large|maxDets=100", apl)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=all|maxDets=1", ar1)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=all|maxDets=10", ar10)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=all|maxDets=100", ar100)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=small|maxDets=100", ars)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=medium|maxDets=100", arm)
        tf.summary.scalar("AR/IoU=0.50:0.95|area=large|maxDets=100", arl)

        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=session_config) as sess:
            result_list = []
            batch_images = []
            image_names = []
            for imageName in filename_to_id_dict.keys():
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
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                image = cv2.resize(image, (model_input_size, model_input_size), cv2.INTER_LINEAR)
                batch_images.append(image)
                image_names.append(imageName)
                if len(batch_images) < batch_size:
                    continue
                images = np.array(batch_images)

                # Run inference
                logits_np = model.inference(  # xywh
                    graph, sess, feeds=images)

                for n in range(batch_size):
                    logit_np = logits_np[n]
                    result = logit_np[logit_np[:, 4] > 0.001]
                    for i in range(result.shape[0]):
                        category_id = np.argmax(result[i][5:])
                        if coco_id:
                            category_id = postprocess_utils.COCO_ID[category_id]
                        score = result[i][4]
                        bbox = result[i][:4] / (1. * model_input_size)
                        result_list.append({'image_id': filename_to_id_dict[image_names[n]],
                                            'category_id': category_id,
                                            'score': score,
                                            'bbox': [(bbox[0] - .5 * bbox[2]) * width,
                                                     (bbox[1] - .5 * bbox[3]) * height,
                                                     bbox[2] * width,
                                                     bbox[3] * height]})
                batch_images = []
                image_names = []

            # Running evaluation
            cocoDt = cocoGt.loadRes(result_list)
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            merged_summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(FLAGS.log_dir)
            scalar_list = [ap, ap50, ap75, aps, apm, apl, ar1, ar10, ar100, ars, arm, arl]
            summary = sess.run(merged_summaries,
                               feed_dict={scalar_list[i]: cocoEval.stats[i] for i in range(len(scalar_list))})
            # print('summary:', summary)
            writer.add_summary(summary=summary, global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
