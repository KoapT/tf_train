import numpy as np
import cv2

COCO_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
           34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
           63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def nms_np(predictions_with_boxes, iou_threshold=0.5, confidence_threshold=0.3, coco_id=False):  # box:xywh
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array[batch,num_boxes,(4+1+2)],
    first 4 values in 3rd dimension are bbox attrs, 5th is confidence, 6/7th classifications
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    results = []
    for i, image_pred in enumerate(predictions):
        result = {}
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
        if coco_id:
            classes_coco = np.zeros_like(classes)
            for i in range(classes.shape[0]):
                classes_coco[i] = COCO_ID[classes[i]]
            classes = classes_coco

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]  # get all the boxes of this class
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
                ious = np.array([bbox_iou_np(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
        results.append(result)
    # print (results)
    return results


def bbox_iou_np(boxes1, boxes2):  # boxes1,boxes2: xywh
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


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
