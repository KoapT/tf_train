#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : det_aug_utils.py
#   Author      : Koap
#   Created date: 2019/12/19 上午10:29
#   Description :
#
# ================================================================

import numpy as np
import cv2
import random
import imgaug as ia
import tensorflow as tf
import time
from imgaug import augmenters as iaa

COLOR = {'probability': .5, 'multiply': (0.7, 1.6), 'add_to_hue_value': (-20, 20), 'gamma': (0.5, 2.0),
         'per_channel': 0.3}
BLUR = {'probability': .5, 'gaussian_sigma': (0.0, 1.5), 'average_k': (2, 5), 'median_k': (1, 11)}
NOISE = {'probability': .5, 'gaussian_scale': (0.0, 1.5), 'salt_p': 0.005, 'drop_out_p': (0, 0.01), 'per_channel': 0.3}
CROP = {'probability': 0.5}
PAD = {'probability': 0.5, 'size': (0.05, 0.2)}
FLIPUD = {'probability': .0}
FLIPLR = {'probability': .5}
PIECEWISEAFFINE = {'probability': .0}
"""
参数说明:
目前img_augment函数包含7类变换，每类变换含1种到多种的具体变换方式并随机取其一，具体变换及其参数说明如下：

公有参数——'probability'： 各种增强方式的使用几率，取值范围[.0,1.]；
         'per_channel': 各种增强方式是否对于图像的各通道使用不同处理方式。默认False，即对各通道处理方式相同，True则表示不同，
                        输入浮点数p，范围(.0,1.)，则表示有p的概率为True。
COLOR——色彩变换参数，
        'multiply'：RGB模式下，对于各通道像素值做乘法的算子的范围，范围内随机选取. 也可指定固定值（以下的“范围”参考此项）；
        'add_to_hue_value'：HSV模式下，对饱和度值的增减范围；
        'gamma'：对比度变换的gamma参数范围。
BLUR——模糊度变换参数，
        'gaussian_sigma': 高斯模糊的参数范围，值越大越模糊； 
        'average_k': 均值模糊的参数范围，越大越模糊；
        'median_k': 中值模糊的参数范围，越大越模糊；
NOISE——添加噪声变换参数，
        'gaussian_scale': 高斯噪声参数范围，越大噪声越多；
        'salt_p': 椒盐噪声的参数范围，越大噪声越多。
        'drop_out_p': 随机噪点的参数范围，噪点即被随机变成0或其他值的像素点，越大噪点越多。
CROP——随机裁剪，随机裁剪掉有目标的连通区域之外的部分。
PAD——随机填充，
        'size':随机填充的尺寸占原尺寸比例的范围，>0。
FLIPUD和FLIPLR——分别是竖直翻转和水平翻转。
PIECEWISEAFFINE——分段仿射，造成图像扭曲，处理耗时较大。
"""


def continue_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('\'{}\' time consumption:{}'.format(func.__name__, end_time - start_time))
        return result

    return wrapper


class Augment(object):
    def __init__(self, image, num_objs, bboxes, fixed_height, fixed_width,
                 max_num_objects=100, img_dtype=np.float32, bbox_dtype=np.float32):
        super(Augment, self).__init__()
        self.im = image
        self.num = num_objs
        self.bbox = bboxes
        self.h = fixed_height
        self.w = fixed_width
        self.max_num = max_num_objects
        self.img_dtype = img_dtype
        self.bbox_dtype = bbox_dtype

    def __call__(self, p=0.5):
        return self.act_aug(p)

    def act_aug(self, p):
        # @continue_time
        def deal_with_np(image, num_objs, bboxes, h, w):
            if num_objs > 0:
                bboxes1 = bboxes[:num_objs, :]
                if random.random() <= p:
                    image, bboxes1 = self.img_augment(image, bboxes1, num_objs)
                image = self.resize_img(image, h, w)
                image = image.astype(self.img_dtype)
                bboxes[:num_objs, :] = bboxes1
                bboxes = bboxes.astype(self.bbox_dtype)
                return image, bboxes
            else:
                if random.random() <= p:
                    image, bboxes1 = self.img_augment(image, bboxes, num_objs)
                image = self.resize_img(image, h, w)
                image = image.astype(self.img_dtype)
                bboxes = bboxes.astype(self.bbox_dtype)
                return image, bboxes

        image_after, bboxes_after = tf.py_func(deal_with_np, [self.im, self.num, self.bbox, self.h, self.w],
                                               [tf.float32, tf.float32])

        image_after.set_shape([self.h, self.w, 3])
        bboxes_after.set_shape([self.max_num, 4])
        return image_after, bboxes_after

    def img_augment(self, image, bboxes, num_objs):
        """
        使用imgaug库进行的图像增强。
        :param image: np.array, images.
        :param bboxes: np.array, bboxes of object detection.
        :param n: max number of augmenters.
        :return: image and bboxes after augmenting.
        """
        h, w, _ = image.shape
        if num_objs > 0:
            bboxes = bboxes * [h, w, h, w]
            bboxes = bboxes.astype(np.int32)
            bboxes_list = bboxes.tolist()
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            top = max_bbox[0]
            left = max_bbox[1]
            bottom = h - max_bbox[2]
            right = w - max_bbox[3]
        else:
            top = int(h * 0.25)
            left = int(w * 0.25)
            bottom = int(h * 0.25)
            right = int(w * 0.25)

        while True:
            new_bndbox_list = []
            seq = iaa.Sequential(
                children=[
                    # color
                    iaa.Sometimes(
                        COLOR['probability'],
                        iaa.SomeOf(2, [
                            iaa.Multiply(COLOR['multiply'], per_channel=COLOR['per_channel']),
                            iaa.AddToHueAndSaturation(COLOR['add_to_hue_value'], per_channel=COLOR['per_channel']),
                            iaa.GammaContrast(COLOR['gamma'], per_channel=COLOR['per_channel']),
                            iaa.ChannelShuffle()
                        ])
                    ),

                    # blur
                    iaa.Sometimes(
                        BLUR['probability'],
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=BLUR['gaussian_sigma']),
                            iaa.AverageBlur(k=BLUR['average_k']),
                            iaa.MedianBlur(k=BLUR['median_k'])
                        ])
                    ),

                    # noise
                    iaa.Sometimes(
                        NOISE['probability'],
                        iaa.OneOf([
                            iaa.AdditiveGaussianNoise(scale=NOISE['gaussian_scale'], per_channel=NOISE['per_channel']),
                            iaa.SaltAndPepper(p=NOISE['salt_p'], per_channel=NOISE['per_channel']),
                            iaa.Dropout(p=NOISE['drop_out_p'], per_channel=NOISE['per_channel']),
                            iaa.CoarseDropout(p=NOISE['drop_out_p'], size_percent=(0.05, 0.1),
                                              per_channel=NOISE['per_channel'])
                        ])
                    ),

                    # crop and pad
                    iaa.Sometimes(CROP['probability'], iaa.Crop(px=(
                        random.randint(0, top), random.randint(0, right),
                        random.randint(0, bottom), random.randint(0, left)),
                        keep_size=False)),
                    iaa.Sometimes(PAD['probability'], iaa.Pad(
                        percent=PAD['size'],
                        # pad_mode=ia.ALL,
                        pad_mode=["constant", "edge", "linear_ramp", "maximum", "mean", "median",
                                  "minimum"] if num_objs > 0 else ia.ALL,
                        pad_cval=(0, 255)
                    )),

                    # flip
                    iaa.Flipud(FLIPUD['probability']),
                    iaa.Fliplr(FLIPLR['probability']),

                    iaa.Sometimes(PIECEWISEAFFINE['probability'],
                                  iaa.PiecewiseAffine(scale=(0.01, 0.04)))
                ])
            seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变
            # 读取图片
            image_aug = seq_det.augment_images([image])[0]
            n_h, n_w, _ = image_aug.shape
            if num_objs > 0:
                for box in bboxes_list:
                    x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    ], shape=image.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    n_x1 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(image_aug.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(image_aug.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    new_bndbox_list.append([n_y1, n_x1, n_y2, n_x2])
                bboxes_aug = np.array(new_bndbox_list, dtype=np.float32) / [n_h, n_w, n_h, n_w]
            else:
                bboxes_aug = bboxes
            # 长宽比太大的图片不要，产生新的image和bboxes
            if 1 / 3 <= image_aug.shape[0] / image_aug.shape[1] <= 3:
                break
        return image_aug, bboxes_aug

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 1:
            image = image[:, ::-1, :]
            bboxes[:, [1, 3]] = 1 - bboxes[:, [3, 1]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 1:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_u_trans = max_bbox[0]
            max_l_trans = max_bbox[1]
            max_d_trans = 1 - max_bbox[2]
            max_r_trans = 1 - max_bbox[3]

            crop_ymin = max(0, max_bbox[0] - random.uniform(0, max_u_trans))
            crop_xmin = max(0, max_bbox[1] - random.uniform(0, max_l_trans))
            crop_ymax = min(1, max_bbox[2] + random.uniform(0, max_d_trans))
            crop_xmax = min(1, max_bbox[3] + random.uniform(0, max_r_trans))

            image[:int(crop_ymin * h), :] = random.choice([0, image[int(crop_ymin * h):int(crop_ymin * h) + 1, :]])
            image[:, :int(w * crop_xmin)] = random.choice([0, image[:, int(w * crop_xmin):int(w * crop_xmin) + 1]])
            image[int(crop_ymax * h):, :] = random.choice([0, image[int(crop_ymax * h) - 1:int(crop_ymax * h), :]])
            image[:, int(w * crop_xmax):] = random.choice([0, image[:, int(w * crop_xmax) - 1:int(w * crop_xmax)]])

            # image = image[int(crop_ymin * h): int(crop_ymax * h), int(w * crop_xmin): int(w * crop_xmax)]

            # bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_scale(self, image, bboxes, minsize=10):
        pass

    def random_translate(self, image, bboxes):
        if random.random() < 1:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_u_trans = max_bbox[0]
            max_l_trans = max_bbox[1]
            max_d_trans = 1 - max_bbox[2]
            max_r_trans = 1 - max_bbox[3]

            tx = random.uniform(-max_l_trans, max_r_trans)
            ty = random.uniform(-max_u_trans, max_d_trans)

            M = np.array([[1, 0, tx * w], [0, 1, ty * h]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + ty
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + tx

        return image, bboxes

    def resize_img(self, image, h, w):
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)
        return image

# 直接操作tensor的 水平翻转
# def _random_horizontal_flip(image->tensor, bboxes->tensor):
#     global image_after, bboxes_after
#     if random.random() < 12:
#         _, w, _ = image.shape
#         image_after = image[:, ::-1, :]
#         ymin, xmin, ymax, xmax = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
#         xmin_after = tf.add(-xmax, 1)
#         xmax_after = tf.add(-xmin, 1)
#         bboxes_after = tf.concat([ymin, xmin_after, ymax, xmax_after], axis=1)
#     return image_after, bboxes_after