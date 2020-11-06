import os.path
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', '',
                    'Check point path')

flags.DEFINE_string('images_dir', '',
                    'Images directory')

flags.DEFINE_string('predictions_save_dir', '',
                    'Predictions save directory')

flags.DEFINE_string('predictions_color_save_dir', '',
                    'Predictions with color save directory')

flags.DEFINE_string('label_map_path', '',
                    'Label map path')

flags.DEFINE_float('mask_vis_alpha', 0.7,
                    'mask visible alpha on original image')


# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported model.
_OUTPUT_NAME = 'SemanticPredictions'


EXTS = ['.jpg', '.png', '.jpeg', '.bmp']


def is_img(fileName):
    ext = os.path.splitext(fileName)[1].lower()
    if ext in EXTS:
        return True
    else:
        return False

def get_palette_from_label_map(label_map_file_path):
    labels = []
    with open(label_map_file_path, 'r') as f:
        labels = json.load(f)
    
    palette = [0 for i in range(len(labels)*3)]
    for label in labels:
        idx = label['id']
        color = label['vis']
        palette[idx*3] = color['r']
        palette[idx*3 + 1] = color['g']
        palette[idx*3 + 2] = color['b']
        
    return palette

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
    imagePathList = []
    for imageFileName in os.listdir(FLAGS.images_dir):
        if is_img(imageFileName):
            imagePathList.append(os.path.join(FLAGS.images_dir, imageFileName))
    print('%d images for reference: -------------------------'%(len(imagePathList)))
    print(imagePathList)
    print('-------------------------------------')
    
    palette = get_palette_from_label_map(FLAGS.label_map_path)
    
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.checkpoint_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        tensor_dict = {}
        for key in [
            _INPUT_NAME, _OUTPUT_NAME
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        
        with tf.Session() as sess:
            for imagePath in imagePathList:
                print('Process: {}'.format(imagePath))
                images = np.expand_dims(
                        load_image_into_numpy_array(Image.open(imagePath)), 0)
                
                # Run inference
                start = time.time()
                predictions = sess.run(
                        tensor_dict[_OUTPUT_NAME], 
                        feed_dict={tensor_dict[_INPUT_NAME]:images})
                end = time.time()
                print('Time taken: {} seconds'.format(end - start))
                
                for i in range(predictions.shape[0]):
                    image_show = Image.fromarray(images[i])
                    prediction_show = Image.fromarray(
                            predictions[i].astype(np.uint8))
                    prediction_show.putpalette(palette)
                    
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(image_show)
                    plt.subplot(1,3,2)
                    plt.imshow(prediction_show)
                    plt.subplot(1,3,3)
                    plt.imshow(image_show)
                    plt.imshow(prediction_show, alpha=FLAGS.mask_vis_alpha)
                    plt.show()
    