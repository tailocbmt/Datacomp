import argparse
import copy
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import os.path as osp
import os

from models.experimental import attempt_load
from utils.datasets import LoadImages, create_dataloader
from utils.general import check_img_size, check_requirements, colorstr, is_ascii, \
    non_max_suppression, scale_coords, box_iou, set_logging, increment_path, \
    xywh2xyxy
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())


@torch.no_grad()
def run(weights,  # model.pt path(s)
        source,  # file/dir
        img_size,  # inference size (pixels)
        conf_threshold,  # confidence threshold
        iou_threshold,  # NMS IOU threshold
        max_det,  # maximum detections per image
        device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img,  # show results
        save_txt,  # save results to *.txt
        save_conf,  # save confidences in --save-txt labels
        save_crop,  # save cropped prediction boxes
        nosave,  # do not save images
        classes,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms,  # class-agnostic NMS
        augment,  # augmented inference
        visualize,  # visualize features
        dir,  # save results to results/detect/
        exist_ok,  # existing results/detect/ ok, do not increment
        line_thickness,  # bounding box thickness (pixels)
        hide_labels,  # hide labels
        hide_conf,  # hide confidences
        half,  # use FP16 half-precision inference
        debug):
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    '''# Directories
    save_dir = increment_path(Path(dir), exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir'''
    dst_dir = osp.join(source, 'visualize')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(osp.join(dst_dir, 'train')):
        os.mkdir(osp.join(dst_dir, 'train'))
    if not os.path.exists(osp.join(dst_dir, 'val')):
        os.mkdir(osp.join(dst_dir, 'val'))
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    suffix = Path(w).suffix.lower()
    assert suffix == ".pt"

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16

    img_size = check_img_size(img_size, s=stride)  # check image size

    # Dataloader
    grid_size = max(int(model.stride.max()), 32)
    dataset = create_dataloader(source, imgsz=img_size, batch_size=1, stride=grid_size,
                                hyp=None, augment=False, cache='disk', pad=0.5, rect=False)[0]
    # ------------------------------------------------------------

    # Run inference
    # ------------------------------------------------------------
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    for i, (img, targets, paths, shapes) in enumerate(tqdm(dataset)): # read single image at a time
        #im0 = np.ascontiguousarray(img[0].permute((1, 2, 0)).numpy()[:, :, ::-1]) # get np.ndarray image
        im_name = '/'.join(paths[0].split('\\')[-2:])
        print(im_name)
        im0 = cv2.imread(paths[0])
        annotator = Annotator(im0, line_width=1, pil=False)

        img = img.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        batch_size, _, height, width = img.shape  # batch size, channels, height, width

        out = model(img, augment=augment)[0]  # inference outputs
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        """
            Post process bboxes
            1. Non max suppression

        """
        out = non_max_suppression(out, conf_threshold, iou_threshold, labels=classes, multi_label=True)[0]

        # gt bboxes
        labels_per_img = targets[: , 1:]
        labels_per_img[:, 1:] = xywh2xyxy(labels_per_img[:, 1:])
        scale_labels = copy.deepcopy(labels_per_img)
        scale_labels[:, 1:] = scale_coords(img.shape[2:], scale_labels[:, 1:], im0.shape[:2])
        # Draw gt bboxes
        for j, *target_xyxy in enumerate(scale_labels):
            target_xyxy = target_xyxy[0].cpu().numpy()
            gt_label, gt_xyxy = str(int(target_xyxy[0])),target_xyxy[1:] 
            try:
                annotator.box_label(gt_xyxy, label=gt_label, color=(0, 0, 255), text_position='br') # red box for ground truth (only label)
            except: # catch un-detected prediction
                annotator.box_label(gt_xyxy, label=None, color=(0, 0, 255), text_position='br') # red box for ground truth


        # Draw predicted bboxes
        for si, pred in enumerate(out): # per bbox
            normed_pred = pred.clone()

            # Evaluate iou
            same_labels = labels_per_img[labels_per_img[:, 0] == normed_pred[5]]
            if len(same_labels) == 0:
                continue
            normed_pred = torch.unsqueeze(pred, dim=0)
            
            ious = box_iou(same_labels[:, 1:], normed_pred[:, :4]) #torch.Tensor (num_detections, iou)
            # change ious from torch.Tensor to string to display
            ious = ious.cpu()

            idmax = torch.argmax(ious)
            max_iou = ious[idmax][0]

            # scale box for original image
            normed_pred[:, :4] = scale_coords(img.shape[2:], normed_pred[:, :4], im0.shape[:2])
            same_labels[:, 1:] = scale_coords(img.shape[2:], same_labels[:, 1:], im0.shape[:2])
            
            # Loop through prediction to draw intersection of predicted bbox and gt bbox
            *xyxy, conf, cls = normed_pred[0]

            gt_xyxy = same_labels[idmax].cpu().numpy()[1:]

            annotator.box_label(xyxy, label="{} {:.2f}".format(int(cls) ,max_iou), color=(0, 255, 0)) # green box for prediction box with iou (iou and label)

            if max_iou < 0.5:
                continue
            elif max_iou >= 0.5:
                annotator.intersect_box(xyxy, gt_xyxy, box_color=(255, 0, 0))
            else:
                raise ValueError('IOU cannot be negative')            

        # Display image
        cv2.imwrite(osp.join(dst_dir, im_name), im0)


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--weights', type=str, help='specify your weight path', default='Datacomp/best.pt')
    args.add_argument('--source', type=str, help='folder contain image', default='Datacomp/baseline')
    args.add_argument('--dir',type=str, help='save results to dir', required=False)
    args.add_argument('--conf-threshold', type=float, default=0.25, help='confidence threshold')
    args.add_argument('--iou-threshold', type=float, default=0.6, help='NMS IoU threshold')
    args.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    args.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    args.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    args.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    args.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args.add_argument('--debug', action='store_true', help='go into debug mode and inspect box IoU')
    args = args.parse_args()

    args.agnostic_nms = False
    args.augment = False
    args.classes = None
    args.exist_ok = False
    args.img_size = 640
    args.nosave = False
    args.view_img = False
    if args.debug:
        args.view_img = True
    args.visualize = False
    args.max_det = 1000
    args.line_thickness = 2

    return args


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    main(parser())
