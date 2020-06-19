import sys
import os
import time
import argparse
import pickle 
import random
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import CRAFT.craft_utils as craft_utils
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
import json
import zipfile

from CRAFT.craft import CRAFT

from collections import OrderedDict
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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.3, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')
parser.add_argument('--image_path', default='CRAFT/data/demo.jpg', type=str, help='folder path to input image')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

# load net
net = CRAFT()     # initialize

print('Loading weights from checkpoint (' + args.trained_model + ')')
if args.cuda:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

result_folder = 'result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

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

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, ratio_w, ratio_h



def run_craft_image(img):
    bboxes, polys, score_text, ratio_w, ratio_h = test_net(net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

    list_ROI = file_utils.saveResult(img[:,:,::-1], bboxes, dirname=result_folder)

    return list_ROI

# def run_craft_folder(folder_name):

#     """ For test images in a folder """
    
#     image_list, _, _ = file_utils.get_files(args.test_folder)
#     # load net
#     net = CRAFT()     # initialize

#     print('Loading weights from checkpoint (' + args.trained_model + ')')
#     if args.cuda:
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
#     else:
#         net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

#     if args.cuda:
#         net = net.cuda()
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = False

#     net.eval()

#     # LinkRefiner
#     refine_net = None
#     if args.refine:
#         from refinenet import RefineNet
#         refine_net = RefineNet()
#         print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
#         if args.cuda:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
#             refine_net = refine_net.cuda()
#             refine_net = torch.nn.DataParallel(refine_net)
#         else:
#             refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

#         refine_net.eval()
#         args.poly = True

#     list_LP = []
#     # load data
#     img = None
#     for k, image_path in enumerate(image_list):
#         print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
#         img = imgproc.loadImage(image_path)

#         bboxes, polys, score_text, ratio_w, ratio_h = test_net(net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

#         list_ROI = file_utils.saveResult(img[:,:,::-1], bboxes, image_path, dirname=result_folder)
#         list_LP.append(list_ROI)

#     return list_LP

if __name__ == '__main__':
    List_ROI = run_craft_image(cv2.imread("data/demo.jpg"))
    for ROI in List_ROI:
        print(ROI.shape)
    
    # list_LP = run_craft_folder(args.test_folder)
    # for LP in list_LP:
    #     for ROI in LP:
    #         print(ROI.shape)
    #         print('--------------')