# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import pickle 
import random
from time import gmtime, strftime

import CRAFT.imgproc

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

'''
  return index 1->4
  x = poly[0][0]
  y = poly[0][1]
'''
# def getIndexBbox(point_x, point_y, center_x, center_y):
#   if point_x < center_x:
#     if point_y < center_y:
#       return 0
#     else
#       return 1 
#   else:
#     if point_y < center_y:
#       return 2
#     else
#       return 3

#def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
def saveResult(img, boxes, image_path =None, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        img_raw = img.copy()
        print("img_raw shape: ", img_raw.shape)
        my_LP = []
        list_ROI = []
        #with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))

            img_tmp = img_raw.copy()
            poly = poly.reshape(-1, 2)

            ROI = img_tmp[poly[0][1]:poly[3][1], poly[0][0]:poly[1][0], :]

            time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            number = str(random.randint(0, 10000))
            time_number = time + '_' + number + '_'

            if image_path != None:
                ROI_name = dirname + "ROI_" + (image_path.split("/")[-1]).split('.')[0] + '_' + str(i) + '.jpg'
            else:
                ROI_name = dirname + "ROI_" + time_number + str(i) + '.jpg'
                
            cv2.imwrite(ROI_name, ROI)
            
            list_ROI.append(ROI_name)

            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)
            
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # return path output photos craft
        return list_ROI
