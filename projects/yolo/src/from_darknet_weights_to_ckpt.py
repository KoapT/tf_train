import tensorflow as tf
import numpy as np
import os
from yolo.src import models_rigister
from yolo.src.utils import param_file_access

MODEL_NAME = 'Object_detection'
darknet_weights = '/home/psdz/TK/darknet/backup/yolov4.weights'
output_dir = '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/'
model_info_path = '/home/psdz/TK/tensorFlowTrain/projects/yolo/run/pretrain_models/Object_Detection/config.json'
sample_info_path = '/home/psdz/TK/tensorFlowTrain/samples_to_train/MscocoDet/LabelMap/label_map.json'

model_cfg = param_file_access.get_json_params(model_info_path)
model_input_size = model_cfg['model_config']['input_size']
out_ckpt_name = 'model_%d_eval.ckpt' % model_input_size
ckpt_path=os.path.join(output_dir,out_ckpt_name)


def weights_transfer(model_name):
    model = models_rigister.get_model(model_name)
    model.initialize(
        model_info_path=model_info_path,
        sample_info_path=sample_info_path,
        is_training=False)
    with tf.Graph().as_default() as graph:
        # model.create_for_inference()
        model.create_for_evaluation()
        load_ops = load_weights(tf.global_variables(), darknet_weights)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(load_ops)
            save_path = saver.save(sess, save_path=ckpt_path)
            print('Model saved in path: {}'.format(save_path))


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)  # np.ndarray
    print('weights_num:', weights.shape[0])
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for vari in batch_norm_vars:
                    shape = vari.shape.as_list()
                    num_params = np.prod(shape)
                    vari_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(
                        tf.assign(vari, vari_weights, validate_shape=True))  # tf.sssign() Assign a value to a variable

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                           bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(
                    tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    print('ptr:', ptr)
    return assign_ops


if __name__ == '__main__':
    weights_transfer(MODEL_NAME)
