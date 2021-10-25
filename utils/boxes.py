import numpy as np
import copy

import torch
import torchvision


def xyxy_to_xywh(xyxy):
    # convert [x1, y1, x2, y2] box format to [cX, cY, w, h] box format
    # xyxy must not be a normalized box
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        cX = round((xyxy[0] + xyxy[2]) / 2)
        cY = round((xyxy[1] + xyxy[3]) / 2)
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        return [cX, cY, w, h]
    elif isinstance(xyxy, np.ndarray):
        cX = np.round((xyxy[:, 0] + xyxy[:, 2]) / 2)
        cY = np.round((xyxy[:, 1] + xyxy[:, 3]) / 2)
        center = np.asarray([cX, cY], dtype=np.int_)
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        wh = np.asarray([w, h], dtype=np.int_)
        return np.vstack((center, wh)).T
    elif isinstance(xyxy, torch.Tensor):
        cX = torch.round((xyxy[:, 0] + xyxy[:, 2]) / 2)
        cY = torch.round((xyxy[:, 1] + xyxy[:, 3]) / 2)
        center = torch.vstack([cX, cY])
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        wh = torch.vstack([w, h])
        return torch.vstack((center, wh)).type(torch.int).T
    else:
        raise TypeError('Argument xyxy must be a list, tuple, numpy array or torch Tensor.')


def xywh_to_xyxy(xywh):
    # Convert [cx cy w h] box format to [x1 y1 x2 y2] format.
    # xywh must not be a normalized box
    if isinstance(xywh, (list, tuple)):
        assert len(xywh) == 4
        return [int(xywh[0] - xywh[2]/2),int(xywh[1] - xywh[3]/2), int(xywh[0] + xywh[2]/2), int(xywh[1] + xywh[3] / 2)]
    elif isinstance(xywh, np.ndarray):
        x1 = np.asarray(np.round(xywh[:, 0] - xywh[:, 2] / 2), dtype=np.int_)
        x2 = np.asarray(np.round(xywh[:, 0] + xywh[:, 2] / 2), dtype=np.int_)
        y1 = np.asarray(np.round(xywh[:, 1] - xywh[:, 3] / 2), dtype=np.int_)
        y2 = np.asarray(np.round(xywh[:, 1] + xywh[:, 3] / 2), dtype=np.int_)
        return np.vstack((x1, y1, x2, y2)).T
    elif isinstance(xywh, torch.Tensor):
        x1 = torch.round(xywh[:, 0] - xywh[:, 2] / 2)
        x2 = torch.round(xywh[:, 0] + xywh[:, 2] / 2)
        y1 = torch.round(xywh[:, 1] - xywh[:, 3] / 2)
        y2 = torch.round(xywh[:, 1] + xywh[:, 3] / 2)
        return torch.vstack((x1, y1, x2, y2)).type(torch.int).T


def scale_box(img, box):
    """ Scale normalized box w.r.t to image height and width """
    if isinstance(box, (tuple, list)):
        h, w = img.shape[:2]
        nb1 = round(box[0] * w)
        nb2 = round(box[1] * h)
        nb3 = round(box[2] * w)
        nb4 = round(box[3] * h)
        return [nb1, nb2, nb3, nb4]
    if isinstance(box, np.ndarray):
        h, w = img.shape[:2] # img shape (h, w, c)
        nb1 = np.round(box[:, 0] * w)
        nb2 = np.round(box[:, 1] * h)
        nb3 = np.round(box[:, 2] * w)
        nb4 = np.round(box[:, 3] * h)
        return np.vstack((nb1, nb2, nb3, nb4)).astype(np.int_).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            w, h = img.shape[2:]  # img shape (batch, c, w, h)
        else:
            h, w = img.shape[:2]
        nb1 = torch.round(box[:, 0] * w)
        nb2 = torch.round(box[:, 1] * h)
        nb3 = torch.round(box[:, 2] * w)
        nb4 = torch.round(box[:, 3] * h)
        return torch.vstack((nb1, nb2, nb3, nb4)).type(torch.int).T

def scale_xyxy(img, box):
    """
    Scale normalized box w.r.t image true height and width.
    Box and image must be the same type, both numpy array or torch.Tensor
    Here we assume box format is xyxy
    """
    new_box = copy.deepcopy(box)
    if isinstance(box, (tuple, list)):
        h, w = img.shape[:2]
        x1 = round(new_box[0] * w)
        x2 = round(new_box[2] * w)
        y1 = round(new_box[1] * h)
        y2 = round(new_box[3] * h)
        return [x1, y1, x2, y2]
    if isinstance(box, np.ndarray):
        h, w = img.shape[:2] # img shape (h, w, c)
        x1 = np.round(new_box[:, 0] * w)
        x2 = np.round(new_box[:, 2] * w)
        y1 = np.round(new_box[:, 1] * h)
        y2 = np.round(new_box[:, 3] * h)
        return np.vstack((x1, y1, x2, y2)).astype(np.int_).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            w, h = img.shape[2:]  # img shape (batch, c, w, h)
        else:
            h, w = img.shape[:2]
        x1 = torch.round(new_box[:, 0] * w)
        x2 = torch.round(new_box[:, 2] * w)
        y1 = torch.round(new_box[:, 1] * h)
        y2 = torch.round(new_box[:, 3] * h)
        return torch.vstack((x1, y1, x2, y2)).type(torch.int).T


def scale_xywh(img, box):
    """
    Scale normalized box w.r.t image true height and width.
    Box and image must be the same type, both numpy array or torch.Tensor
    Here we assume box format is xywh
    """
    if isinstance(box, (tuple, list)):
        height, width = img.shape[:2]
        cX = int(round(box[0] * width))
        cY = int(round(box[1] * height))
        w = int(round(box[2] * width))
        h = int(round(box[3] * height))
        return [cX, cY, w, h]
    if isinstance(box, np.ndarray):
        height, width = img.shape[:2] # img shape (h, w, c)
        cX = np.round(box[:, 0] * width)
        cY = np.round(box[:, 1] * height)
        w = np.round(box[:, 2] * width)
        h = np.round(box[:, 3] * height)
        return np.vstack((cX, cY, w, h)).astype(np.int_).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            width, height = img.shape[2:] # img shape (batch, c, w, h)
        else:
            height, width = img.shape[:2]
        cX = torch.round(box[:, 0] * width)
        cY = torch.round(box[:, 1] * height)
        w = torch.round(box[:, 2] * width)
        h = torch.round(box[:, 3] * height)
        return torch.vstack((cX, cY, w, h)).type(torch.int).T


def normalize_box(img, box):
    """ Normalize any box format w.r.t image size """
    if isinstance(box, (tuple, list)):
        height, width = img.shape[:2]
        return [round(box[0] / width, 6), round(box[1] / height, 6), round(box[2] / width, 6), round(box[3] / height, 6)]
    elif isinstance(box, np.ndarray):
        height, width = img.shape[:2]
        return np.vstack((np.round(box[:, 0] / width, 6), np.round(box[:, 1] / height, 6),
                          np.round(box[:, 2] / width, 6), np.round(box[:, 3] / height, 6))).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            width, height = img.shape[2:]  # img shape (batch, c, w, h)
        else:
            height, width = img.shape[:2]
        return torch.vstack((box[:, 0] / width, box[:, 1] / height,
                             box[:, 2] / width, box[:, 3] / height)).T


if __name__ == '__main__':
    np_xyxy = np.asarray([
        [3, 4, 5, 6],
        [2, 4, 6, 8]
    ])
    torch_xyxy = torch.tensor([
        [3, 4, 5, 6],
        [2, 4, 6, 8]
    ])
    np_xywh = xyxy_to_xywh(np_xyxy)
    print(normalize_box(np.zeros((100, 100, 3)), np_xyxy))
    print(np_xywh)
    print(xywh_to_xyxy(np_xywh))


    torch_xywh = xyxy_to_xywh(torch_xyxy)
    print(normalize_box(torch.zeros((1, 3, 100, 100)), torch_xyxy))
    print(torch_xywh)
    print(xywh_to_xyxy(torch_xywh))

    normalized_np_xyxy = np.asarray([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5]
    ])
    normalized_torch_xyxy = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5]
    ])
    print(scale_xyxy(np.ones((1, 240, 240, 3)), normalized_np_xyxy))
    print(scale_xyxy(torch.ones((1, 3, 240, 240)), normalized_torch_xyxy))