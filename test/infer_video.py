import numpy as np
import os
import sys
import tensorflow as tf

#import matplotlib
#matplotlib.use('Agg')


#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#import pylab
#import skimage
#import imageio
#注释的代码执行一次就好，以后都会默认下载完成
#imageio.plugins.ffmpeg.download()

import cv2
import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")



#usr===============================
TH_MIN_SCORE = 0.4
VIDEO_SHOW_MODE = 'play' #'save'
#==================================

#environment deppend-------------------
#Ubuntu16.04下安装FFmpeg:
#sudo add-apt-repository ppa:djcj/hybrid
#sudo apt-get update
#sudo apt-get install ffmpeg

#pip install moviepy --ignore-installed
#--------------------------------------


# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images.
# %matplotlib inline


print("Hello,TensorFlow!!!")

image_exts = ['.jpg', '.png', '.jpeg', '.bmp']
video_exts = ['.mp4', '.avi', '.rmvb', '.mp3', '.flv', '.qsv', '.mts', '.dav']

def is_image(fileName):
    ext = os.path.splitext(fileName)[1] 
    ext = ext.lower() 
    if ext in image_exts:
        return True
    else:
        return False

def is_video(fileName):
    ext = os.path.splitext(fileName)[1] 
    ext = ext.lower()
    if ext in video_exts:
        return True
    else:
        return False

# What model to download.
# MODEL_NAME = 'pre-train_models/ssdlite_mobilenet_v2_voc_2018_05_18_3'  # 'ssd_mobilenet_v2_coco_2018_03_29  ssd_mobilenet_v2_voc_2018_04_11'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('samples_to_train/circuit_breaker_off/label_map', 'label_map.pbtxt')
# mscoco_label_map  pascal_label_map pascal_label_map_cutout

CURRENT_DIR = os.path.abspath('.')
PATH_TO_LABELS = os.path.join(CURRENT_DIR, 'test_ckpt', 'label_map.pbtxt')
PATH_TO_CKPT = os.path.join(CURRENT_DIR, 'test_ckpt', 'frozen_inference_graph.pb')
TEST_IMAGE_PATHS = []
TEST_IMAGE_DIR = os.path.join(CURRENT_DIR, 'test_imgs')
for fileName in os.listdir(TEST_IMAGE_DIR):
    if is_image(fileName):
        TEST_IMAGE_PATHS = TEST_IMAGE_PATHS + [os.path.join(TEST_IMAGE_DIR, fileName)]
TEST_IMAGE_PATHS = []
print('%d images for test: -------------------------'%(len(TEST_IMAGE_PATHS)))
print(TEST_IMAGE_PATHS)
print('-------------------------------------')

TEST_VIDEO_PATHS = []
TEST_VIDEO_OUT_PATHS = []
TEST_VIDEO_DIR = os.path.join(CURRENT_DIR, 'test_videos')
TEST_VIDEO_OUT_DIR = os.path.join(TEST_VIDEO_DIR, 'out')
for fileName in os.listdir(TEST_VIDEO_DIR):
    if is_video(fileName):
        TEST_VIDEO_PATHS = TEST_VIDEO_PATHS + [os.path.join(TEST_VIDEO_DIR, fileName)]
        TEST_VIDEO_OUT_PATHS = TEST_VIDEO_OUT_PATHS + [os.path.join(TEST_VIDEO_OUT_DIR, os.path.basename(fileName))]
print('%d videos for test: -------------------------'%(len(TEST_VIDEO_PATHS)))
print(TEST_VIDEO_PATHS)
print('-------------------------------------')

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session()

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
num_classes = label_map_util.get_max_label_map_index(label_map)
print('num_classes= {}'.format(num_classes))
categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                            max_num_classes=num_classes,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
#    (im_width, im_height) = image.size
#    return np.array(image.getdata()).reshape(
#        (im_height, im_width, 3)).astype(np.uint8)
    image_np = np.array(image)
    return image_np


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'tmp_images'  # test_images cut_out
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(85, 130)]
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image ({}).jpg'.format(i)) for i in range(16, 25)]
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{:05d}.jpg'.format(i)) for i in range(1, 4)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            start = time.time()
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})
            end = time.time()
            seconds = end - start
            print('inference time taken per frame: {} seconds'.format(seconds))
            fps = 1 / seconds
            print( 'inference FPS: {0}'.format(fps))

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

plt.switch_backend('TkAgg')
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=TH_MIN_SCORE,#.85,
        line_thickness=8)
    print(os.path.basename(image_path))
    num = output_dict['num_detections']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    for i in range(num):
        score = scores[i]
        if score >= TH_MIN_SCORE:
            print('class = %s, score = %f'%(category_index[classes[i]]['name'], score))
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()

    # isEnabled = True
    # for i in range(output_dict['detection_boxes'].shape[0]):
    #     if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > .5:
    #         print(image_path, ":", output_dict['detection_scores'][i])
    #         if isEnabled:
    #             # Image.fromarray(image_np).show()
    #             plt.figure(figsize=IMAGE_SIZE)
    #             plt.imshow(image_np)
    #             # plt.colorbar()
    #             plt.show()
    #             isEnabled = False


def detect_videos(image_np, sess):
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    start = time.time()
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image_np, 0)})
    end = time.time()
    seconds = end - start
    print('inference time taken per frame: {} seconds'.format(seconds))
    fps = 1 / seconds
    print( 'inference FPS: {0}'.format(fps))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    
#        vis_util.visualize_boxes_and_labels_on_image_array(
#            image_np,
#            output_dict['detection_boxes'],
#            output_dict['detection_classes'],
#            output_dict['detection_scores'],
#            category_index,
#            instance_masks=output_dict.get('detection_masks'),
#            use_normalized_coordinates=True,
#            min_score_thresh=TH_MIN_SCORE,
#            line_thickness=1)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    shape = np.shape(image_np)
    height = shape[0]
    width = shape[1]
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] < TH_MIN_SCORE:
            continue
        score = output_dict['detection_scores'][i]
        cls = output_dict['detection_classes'][i]
        color = colors[cls % 6]
        box = output_dict['detection_boxes'][i]
        left_top = (int(box[1]*width), int(box[0]*height))
        right_bottom = (int(box[3]*width), int(box[2]*height))
        cv2.rectangle(image_np, left_top, right_bottom, color, 2)
        cv2.putText(image_np, '{:.2f}%'.format(score * 100),
                    left_top, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return image_np


def process_image(image):
    global counter
    
    if counter % 1 == 0:
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_np = detect_videos(image, sess)

    counter += 1
    
    return image_np#image




from moviepy.editor import VideoFileClip
from IPython.display import HTML



if VIDEO_SHOW_MODE == 'save' and not os.path.exists(TEST_VIDEO_OUT_DIR):
    os.mkdir(TEST_VIDEO_OUT_DIR)

for i in range(len(TEST_VIDEO_PATHS)):
    counter = 0

    filepath = TEST_VIDEO_PATHS[i]
    new_loc = TEST_VIDEO_OUT_PATHS[i]

    if VIDEO_SHOW_MODE == 'save':
        print('write to video file===================')
    #    white_output = new_loc
    #    clip1 = VideoFileClip(filepath).subclip(6,7)#VideoFileClip(filepath).subclip(60,68)
    #    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
    #    # %time white_clip.write_videofile(white_output, audio=False)
    #    white_clip.write_videofile(white_output, audio=False)
    elif VIDEO_SHOW_MODE == 'play':
        print('real time play===================')
    #    vid = imageio.get_reader(filepath,  'ffmpeg')
    #    for num,im in enumerate(vid):
    #        image = skimage.img_as_float(im).astype(np.float64)
    #        fig = pylab.figure()
    #        fig.suptitle('image #{}'.format(num), fontsize=20)
    #        pylab.imshow(im)
    #        pylab.show()
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('Original video FPS: {}'.format(fps))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
        print('Frame size: {}'.format(size))  
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  #encode: mpge-4
    #    outVideo = cv2.VideoWriter(new_loc, fourcc, fps, size)
        cv2.namedWindow("cap video", 0)
        cnt = 0
        with detection_graph.as_default():
            with tf.Session() as sess:
                while True:  
                    ret, frame = cap.read()
                    if ret == True:
                        print('{}--------'.format(cnt))
                        start = time.time()
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        #do proccess
                        frame_proccessed = detect_videos(frame, sess)
                        
                        end = time.time()
                        seconds = end - start
                        print('Proccess time taken per frame: {0} seconds'.format(seconds))
                        fps = 1 / seconds
                        print( 'Proccess FPS: {0}'.format(fps))
                        cv2.imshow('cap video', frame)
        #                    outVideo.write(frame_proccessed)
                        cnt = cnt + 1
                    else:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        break
                    
    #    outVideo.release()  
        cap.release()  
        cv2.destroyAllWindows()
    else:
        print('please set right mode for video show!!!')


