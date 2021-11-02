import cv2
import numpy as np

import torch


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 5 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, xyxys, color=(255, 255, 255), thickness=2):
    """
    Draw boxes of xyxy format on an image
    """
    for xyxy in xyxys:
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        cv2.rectangle(img, pt1=xyxy[:2], pt2=xyxy[2:], color=color, thickness=thickness)
    return img


def draw_one_box(img, xyxy, color=(255, 255, 255), thickness=2):
    """ Draw one box of xyxy format on an image"""
    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.cpu().numpy()
    cv2.rectangle(img, pt1=xyxy[:2], pt2=xyxy[2:], color=color, thickness=thickness)
    return img


def draw_boxes_with_label(img, xyxys, labels, thickness=2, no_label=False):
    for i, xyxy in enumerate(xyxys):
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        id = int(labels[i])
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=thickness)
        if not no_label:
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[0] + t_size[0] + 3, xyxy[1] + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (xyxy[0], xyxy[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

if __name__ == '__main__':
    img = np.random.rand(200, 200, 3)
    xyxys = np.asarray([
        [20, 20, 50, 50]
    ])
    img = draw_boxes(img, xyxys)
    cv2.imshow('a', img)
    cv2.waitKey(0)