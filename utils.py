import os
import glob

import cv2
import numpy as np

from constants import IMAGE_SIZE


def LoG_filter(image, sigma, size=None):
    # Generate LoG kernel
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7

    if size % 2 == 0:
        size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))

    # Perform convolution using OpenCV filter2D
    result = cv2.filter2D(image, -1, kernel)

    return result


def to_interval(value, start=0, end=IMAGE_SIZE-1):
    if value < start:
        return start
    if value > end:
        return end
    return int(value)


def get_folder_key(path, last_n=5, short=False, short_cut=3, preserve_npy=True):
    '''
    Function returns key for a specific folder path (last_n folders in path, excludes npy by default)
    '''
    return os.sep.join(path.split(os.sep)[-last_n:-short_cut if short else None]) if 'npy' not in path else \
           os.sep.join(path.split(os.sep)[-last_n-1:-(short_cut+1) if short else (-1 if not preserve_npy else None)])


def parse_detected_bboxes(path, img_size):
    dct = {
        "img_size": img_size,
        "labels": {
            #"ct_lungs_artefacts_71_1/031/10000000/10000001/10000007": {
            #    "10000001": {
            #        "bboxes": [[0, start_x, start_y, end_x, end_y, conf],
            #                   [0, start_x, start_y, end_x, end_y, conf]]
            #    }
            #}
        }
    }

    for p in glob.glob(os.path.join(path, '**/*'), recursive=True):
        if not os.path.isfile(p):
            continue
        folder_key = get_folder_key(os.path.dirname(p))
        ext = ".npy" if "npy" in p else ""
        filename = os.path.splitext(os.path.basename(p))[0]+ext
        if folder_key not in dct["labels"]:
            dct["labels"][folder_key] = {}
        if filename not in dct["labels"][folder_key]:
            dct["labels"][folder_key][filename] = {"bboxes": []}
        with open(p, mode='r') as file:
            for line in file:
                # class_id, x_center, y_center, width, height, conf
                elements = list(map(float, line.rstrip().split(" ")[1:]))  # stripping class_id as we have only one class
                x_center, y_center, width, height = map(lambda x: min(int(x * img_size), img_size), elements[:-1])
                confidence = elements[-1]
                start_x = x_center - width // 2
                start_y = y_center - height // 2
                end_x = x_center + width // 2
                end_y = y_center + height // 2
                dct["labels"][folder_key][filename]["bboxes"].append([0, start_x, start_y, end_x, end_y, confidence])

    return dct


def createMIP(np_img, slices_num=15):
    return np.max(np_img, axis=0)
