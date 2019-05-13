# License: MIT
# Author: Karl Stelzner

import numpy as np


def z_where_to_bboxes(z_where, conf):
    bbox = np.zeros(list(z_where.shape[:2]) + [4])
    width, height = conf.patch_width, conf.patch_height
    bbox[..., 0] = z_where[..., 5] - 5
    bbox[..., 1] = z_where[..., 2] - 5

    bbox[..., 2] = bbox[..., 0] + height * z_where[..., 4] + 10
    bbox[..., 3] = bbox[..., 1] + width * z_where[..., 0] + 10

    return bbox


def scene_intersection_over_union(n_pred, n_real, bboxes_pred, bboxes_real):
    num_images = bboxes_pred.shape[0]

    score = 0.0
    num_boxes = np.sum(n_real)
    for i in range(num_images):
        matched = [False] * n_real[i]
        for j in range(min(n_real[i], n_pred[i])):
            best_iou = -1
            best_match = -1
            for k in range(n_real[i]):
                if matched[k]:
                    continue
                cur_iou = intersection_over_union(bboxes_pred[i, j],
                                                  bboxes_real[i, k])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_match = k

            if best_match >= 0:
                score += best_iou
                matched[best_match] = True

    return score / num_boxes


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
