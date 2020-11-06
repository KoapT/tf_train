import numpy as np
import tensorflow as tf
import cv2
from yolo.src.utils.ious import bbox_iou_np

COCO_ID = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
           16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33,
           29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
           42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60,
           55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77,
           68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}


def nms_np(predictions_with_boxes, iou_threshold=0.5, confidence_threshold=0.3,
           coco_id=False, iou_method='diou'):  # box:xywh
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array[batch,num_boxes,(4+1+2)],
    first 4 values in 3rd dimension are bbox attrs, 5th is confidence, 6/7th classifications
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    results = []
    for i, image_pred in enumerate(predictions_with_boxes):
        result = {}
        image_pred = image_pred[image_pred[..., 4] >= confidence_threshold]

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
        if coco_id:
            classes_coco = np.zeros_like(classes)
            for j in range(classes.shape[0]):
                classes_coco[j] = COCO_ID[classes[j]]
            classes = classes_coco

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[cls_mask]  # get all the boxes of this class
            cls_boxes = cls_boxes[
                cls_boxes[:, -1].argsort()[::-1]]  # np.argsort() sort from min to max，return the index.
            cls_scores = cls_boxes[:, -1]  # the last column refers score
            cls_boxes = cls_boxes[:, :-1]  # the fore 4 columns refers the location

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]  # choose the most confident box&score, as the baseline
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([bbox_iou_np(box, x, method=iou_method) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[iou_mask]
                cls_scores = cls_scores[iou_mask]
        results.append(result)
    # print (results)
    return results


def draw_boxes_cv2(boxes: dict, img: np.ndarray, cls_names: dict, model_input_size):
    # draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (0, 0, 0), (255, 255, 255)]
    height, width = img.shape[:2]
    for cls, bboxes in boxes.items():
        color = colors[cls % 8]
        for bbox, score in bboxes:
            bbox = 1. * bbox / model_input_size
            bbox_xyxy = [bbox[0] - .5 * bbox[2], bbox[1] - .5 * bbox[3],
                         bbox[0] + .5 * bbox[2], bbox[1] + .5 * bbox[3]]
            # print(bbox_xyxy)
            box = np.array([bbox_xyxy[0] * width, bbox_xyxy[1] * height,
                            bbox_xyxy[2] * width, bbox_xyxy[3] * height], dtype=np.int32)
            # print(box)
            box = [max(1, box[0]), max(1, box[1]),
                   min(img.shape[1] - 1, box[2]), min(img.shape[0] - 1, box[3])]
            left_top, right_bottom = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(img, left_top, right_bottom, color, 2)
            cv2.putText(img, '{}{:.2f}%'.format(cls_names[cls].strip(), score * 100),
                        left_top, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
            print('name:{0},\t location:{1[0]:>4d},{1[1]:>4d},{1[2]:>4d},{1[3]:>4d},\t confidence:{2:.2%}'
                  .format(cls_names[cls].strip(), box, score))


def decode(conv_output, anchors, num_class, stride):
    """
    :return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
           contains (x, y, w, h, score, probability)
    """

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = tf.reshape(conv_output,
                             [batch_size, output_size, output_size, anchor_per_scale, 5 + num_class])

    conv_raw_dxdy = conv_output[..., 0:2]
    conv_raw_dwdh = conv_output[..., 2:4]
    conv_raw_conf = conv_output[..., 4:5]
    conv_raw_prob = conv_output[..., 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, None], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[None, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, None], y[:, :, None]], axis=-1)
    xy_grid = tf.tile(xy_grid[None, :, :, None, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    result = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    detections = tf.reshape(result, (-1, output_size * output_size * anchor_per_scale, 5 + num_class))

    return result, detections


def nms(batched_predictions, iou_threshold, confidence_threshold, num_class, max_num_obj_per_class=20):
    num_batch = batched_predictions.shape.as_list()[0]
    results = []
    for i in range(num_batch):
        result = []
        predictions = batched_predictions[i]
        predictions = tf.boolean_mask(predictions, predictions[..., 4] > confidence_threshold)
        box_attr = predictions[..., :5]
        box_cls = predictions[..., 5:]
        box_cls = tf.cast(tf.argmax(box_cls, axis=-1), dtype=tf.float32)
        for cls in range(num_class):
            cls_mask = tf.logical_and(box_cls > cls - 1, box_cls < cls + 1)
            cls_pred = tf.boolean_mask(box_attr, cls_mask)
            cls_box_cls = tf.boolean_mask(box_cls[:, None], cls_mask)
            if cls_pred.shape.as_list()[0] == 0:
                result.append(tf.zeros([max_num_obj_per_class, 6]))
            else:
                boxes = cls_pred[..., :4]
                scores = cls_pred[..., 4:5]
                boxes_yxyx = boxes_xywh2yxyx(boxes)
                boxes_with_score_and_cls = tf.concat([boxes_yxyx, scores, cls_box_cls], axis=-1)
                selected_indices = tf.image.non_max_suppression(boxes_yxyx, scores[:, 0], max_num_obj_per_class,
                                                                iou_threshold)
                selected_boxes = tf.gather(boxes_with_score_and_cls, selected_indices)
                number = tf.shape(selected_boxes)[0]
                selected_boxes = tf.cond(tf.less(number, max_num_obj_per_class),
                                         lambda: tf.concat([selected_boxes,
                                                            tf.zeros([max_num_obj_per_class - number, 6])], axis=0),
                                         lambda: selected_boxes[:max_num_obj_per_class, :]
                                         )
                result.append(selected_boxes)
        result_tensor = tf.concat(result, axis=0)
        results.append(result_tensor)
    return tf.stack(results, axis=0)


def boxes_xywh2yxyx(boxes):
    x, y, w, h = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    boxes_yxyx = tf.concat(
        [y - h / 2., x - w / 2., y + h / 2., x + w / 2.], axis=-1)
    return boxes_yxyx
