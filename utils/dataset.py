import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import yaml
from pathlib import Path

from utils.draw import draw_boxes_with_label
from utils.boxes import xywh_to_xyxy, scale_box

# This script is dedicated for FSOFT Datacomp only
'''
Disable for FSOFT Datacomp
def check_num_files(dir):
    """ Function to check if number of annotations file is correspond with
    number of images
    If not, return a *.txt file of unmatched image or annotation file
    :param dir: folder contains 'images' folder and 'labels' folder"""

    image_names = []
    annotation_files = []
    for image in os.listdir(os.path.join(dir, "images")):
        name = image.split('.')[0]
        image_names.append(name)
    for annotations in os.listdir(os.path.join(dir, "labels")):
        name = annotations.split('.')[0]
        annotation_files.append(name)
    print("Begin checking for matching images and annotations")
    print('-' * 20)
    with open(f'{os.path.join(dir, "img_mismatch.txt")}', 'a') as f:
        for image_name in image_names:
            if image_name not in annotation_files:
                f.write(f'{image_name} \n')
    with open(f'{os.path.join(dir, "label_mismatch.txt")}', 'a') as file:
        for annotation in annotation_files:
            if annotation not in image_names:
                file.write(f'{annotation} \n')
    print("Done checking")'''


def inspect_every_images(dir, image_folder, label_folder, phrase=None):
    """
    Function to inspect every image in a directory with its annotation one by one
    annotations file must have YOLO format
    """
    image_dir = os.path.join(dir, image_folder)
    label_dir = os.path.join(dir, label_folder)
    for label in tqdm(os.listdir(label_dir), leave=False):
        if phrase:
            assert isinstance(phrase, str), "phrase to check must be a string"
            if phrase in label:
                image_name = label.split('.')[0]
                label_path = os.path.join(label_dir, label)
                data = np.loadtxt(label_path)
                if data.ndim == 1:
                    data = np.expand_dims(data, axis=0)
                image = cv2.imread(f'{image_dir}/{image_name}.jpg')
                labels = data[:, 0]
                bboxes = data[:, 1:]
                xywhs = scale_box(image, bboxes) # Update 27/10: change to scale_box instead of scale_xywh
                xyxys = xywh_to_xyxy(xywhs)
                img = draw_boxes_with_label(image, xyxys, labels)
                img = img[:, :, ::-1]
                plt.imshow(img)
                plt.show()
        else:
            image_name = label.split('.')[0] # change accordingly to dataset
            label_path = os.path.join(label_dir, label)
            data = np.loadtxt(label_path)
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            image = cv2.imread(f'{image_dir}/{image_name}.jpg')
            labels = data[:, 0]
            bboxes = data[:, 1:]
            xywhs = scale_box(image, bboxes) # Update 27/10: change to scale_box instead of scale_xywh
            xyxys = xywh_to_xyxy(xywhs)
            img = draw_boxes_with_label(image, xyxys, labels)
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.title(image_name)
            plt.show()


def YOLO_to_COCO(root_dir, image_dir, label_dir, dataset_path):
    # Read yaml
    if isinstance(dataset_path, (str, Path)):
        with open(dataset_path, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary
    # Parse yaml
    path = Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]
