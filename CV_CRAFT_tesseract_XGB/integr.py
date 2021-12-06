import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pytesseract
import pandas as pd
from pytesseract import Output
from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from scipy import interpolate
from collections import OrderedDict
from scipy.spatial import distance as dist

from craft import CRAFT



def remove_characters(image, mask):
    image = np.array(image, dtype="float32")
    mask= np.array(mask, dtype="float32")
    image[(mask > 0) ] = np.nan

    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])

    image = np.ma.masked_invalid(image)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~image.mask]
    y1 = yy[~image.mask]
    image = image[~image.mask]

    out = interpolate.griddata((x1, y1), image.ravel(), (xx, yy), method='nearest').astype("uint8")
    return out

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
def test_net(net, image, text_threshold, link_threshold, low_text, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, square_size=2000, interpolation=cv2.INTER_LINEAR, mag_ratio=2)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()



    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    render_img = score_text.copy()

    ret_score_text = imgproc.resize_threshold_mask(render_img,image.shape[0],image.shape[1])



    return boxes, polys, ret_score_text


def create_craft(weights='weights/craft_mlt_25k.pth'):
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(weights, map_location='cpu')))
    net.eval()
    return net

def detect_save_text(image,net):
    t = time.time()

    image = imgproc.loadImage(image)

    bboxes, polys, score_text = test_net(net, image, text_threshold=0.8,link_threshold=0.3, low_text=0.2, poly=False)

    text_label=file_utils.saveResult( polys)


    return text_label

def shift_image(img, dx, dy):
 num_rows, num_cols = img.shape[:2]
 translation_matrix= np.float32([[1, 0, dx], [0, 1, dy]])
 img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
 return img_translation

def preproc_mean_text(labels,img):
    mean_text_list = []
    text_cords=[]
    img_shift_right = shift_image(img, 1, 0)
    img_diff1 = cv2.absdiff(img, img_shift_right)
    img_shift_bottom = shift_image(img, 0, 1)
    img_diff2 = cv2.absdiff(img, img_shift_bottom)
    img_borders = cv2.max(img_diff1, img_diff2)
    img_borders = cv2.cvtColor(img_borders, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_borders, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    thresh=cv2.bitwise_not(thresh)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for i in labels:
        poly=np.array([int(q) for q in i.split(',')]).reshape(-1,2)
        x_mean,y_mean=int((np.mean(poly[:,0]))),int(np.mean(poly[:,1]))
        mean_text_list.append((x_mean,y_mean))
        cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))],255)
        x, y, w, h = np.array(poly)[:, 0].min(), np.array(poly)[:, 1].min(), \
                     np.array(poly)[:, 0].max() - np.array(poly)[:, 0].min(), np.array(poly)[:, 1].max() - np.array(poly)[:,1].min()
        text_cords.append((x, y, w, h,(x_mean,y_mean)))
    without_text = remove_characters(thresh, mask)

    kernel = np.ones((3, 3), np.uint8)
    without_text = cv2.morphologyEx(without_text, cv2.MORPH_DILATE, kernel, iterations=2)
    return without_text, mean_text_list,text_cords

def get_contours(without_text):
    return cv2.findContours(without_text, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def get_rect_line(contours):
    rect_dict= {}
    line_dict= {}
    for i,c in enumerate(contours[0]):
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)

        if cv2.contourArea(c) > 100 :
                    if len(approx)>2   :
                        x, y, w, h = cv2.boundingRect(approx)
                        rect_dict[i]=(x, y, w, h )

                    elif len(approx)==2:
                        line_dict[i]=approx
    return rect_dict,line_dict

def euclide(x_mean,y_mean,text_mean):
    return np.sqrt((text_mean[:, 0] - x_mean) **2 + (text_mean[:, 1] - y_mean)**2)
def get_intersec(mask_1,mask_2):
    and_mask = cv2.bitwise_and(mask_1, mask_2)
    print(np.sum(and_mask))
    or_mask = cv2.bitwise_or(mask_1, mask_2)
    print(np.sum(or_mask))
    return np.sum(and_mask) / np.sum(or_mask)

def filter_rect(img,rect_dict,mean_text_list,threshold_inter=0.05,distance_text=100):
    del_list=[]
    for ind,rect in rect_dict.items():
        coord=rect
        mask_1  = np.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(mask_1, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), 1, -1)
        for ind2,rect_2 in rect_dict.items():
            mask_2 = np.zeros(img.shape[:2], dtype="uint8")
            if ind==ind2:
                continue
            coord2=rect_2
            cv2.rectangle(mask_2, (coord2[0], coord2[1]), (coord2[0] + coord2[2], coord2[1] + coord2[3]), 1, -1)
            intersec= get_intersec(mask_1,mask_2)
            if intersec >threshold_inter:
                if ((coord[0]+coord[2])*( coord[1] + coord[3])) < ((coord2[0]+coord2[2])* (coord2[1] + coord2[3])):
                        del_list.append(ind2)
                else:
                        del_list.append(ind)

    for ind, rect in rect_dict.items():
        x, y, w, h = rect
        x_mean, y_mean = x + w / 2, y + h / 2
        mean_text = np.array(mean_text_list)

        res = euclide(x_mean, y_mean, mean_text)

        if np.sum(res < distance_text) == 0:
            del_list.append(ind)

    del_list = list(set(del_list))
    for del_rec in del_list:
        del rect_dict[del_rec]

    return rect_dict


def get_intersec(mask_1, mask_2):
    and_mask = cv2.bitwise_and(mask_1, mask_2)

    or_mask = cv2.bitwise_or(mask_1, mask_2)

    return np.sum(and_mask) / np.sum(or_mask)


def contourArea_rect(coord):
    x, y, w, h = coord
    return w * h



