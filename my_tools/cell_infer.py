import os
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmdet.apis import (inference_detector, init_detector)


# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_masks(im_name, model):
    im = cv2.imread(im_name)
    outputs = inference_detector(model, im)
    # print(outputs[0][0].shape)
    # print(outputs[0][0][0])
    # print(len(outputs[1]))
    # print(len(outputs[1][0]))
    # print(len(outputs[1][1]))
    # print(len(outputs[1][2]))
    # print(outputs[1][0][0].shape)
    # print(outputs[1][0][0])
    pred_masks = []
    for mask_output in outputs[1]:
        pred_masks.extend(mask_output)
        # print(len(pred_masks))
    res = []
    used = np.zeros(im.shape[:2], dtype=int) 
    for mask in pred_masks:
        mask = mask * (1-used)
        used += mask
        res.append(rle_encode(mask))
    return res


if __name__ == '__main__':
    dataDir='D:/137/dataset/sartorius-cell-instance-segmentation/'
    ids, masks=[],[]
    test_names = glob.glob(dataDir + 'test/*')
    THR = 0.50

    config  = 'work_dirs_cell/tmp/cascade_mask_rcnn_r101_fpn_20e_coco_cell_001/cascade_mask_rcnn_r101_fpn_20e_coco_cell.py'
    ckpt = 'work_dirs_cell/tmp/cascade_mask_rcnn_r101_fpn_20e_coco_cell_001/epoch_20.pth'
    cfg_options = {'model.test_cfg.rcnn.score_thr': THR}

    cfg = Config.fromfile(config)
    model = init_detector(cfg, ckpt, cfg_options=cfg_options)
    encoded_masks = get_masks(test_names[0], model)
