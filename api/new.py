import os
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
import cv2

from xml.etree import ElementTree as et

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import transforms as T

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
OUTPUT_DIR = 'C:/Users/atuli/OneDrive/Desktop/darkine_new/api'

arr1 = ['pagination',
 'accordion',
 'hero image',
 'content',
 'illustration',
 'image',
 'badge',
 'USP',
 'CAT',
 'CTA',
 'data feed',
 'navigation',
 'footer',
 'header',
 'Input field',
 'label',
 'alert',
 'heading',
 'tab',
 'video',
 'progress bar',
 'tag',
 'QR code',
 'pop-up',
 'logo',
 'copy',
 'date picker',
 'tooltip',
 'placeholder',
 'bag',
 'breadcrumbs',
 'ad',
 'icon',
 'card',
 'slider',
 'decoration',
 'title',
 'carousel',
 'dropdown',
 'false'
]

def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = get_object_detection_model(len(arr1))

def inference(img, model, detection_threshold=0.70):
  '''
  Infernece of a single input image
  inputs:
    img: input-image as torch.tensor (shape: [C, H, W])
    model: model for infernce (torch.nn.Module)
    detection_threshold: Confidence-threshold for NMS (default=0.7)

  returns:
    boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
  '''
  model.eval()

  img = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
  outputs = model([img])

  boxes = outputs[0]['boxes'].data.cpu().numpy()
  scores = outputs[0]['scores'].data.cpu().numpy()
  labels = outputs[0]['labels'].data.cpu().numpy()

  boxes = boxes[scores >= detection_threshold].astype(np.int32)
  labels = labels[scores >= detection_threshold]
  scores = scores[scores >= detection_threshold]

  return boxes, scores, labels


import matplotlib.patches as patches

def plot_image(img, boxes, scores, labels, dataset, save_path=None):
  '''
  Function that draws the BBoxes, scores, and labels on the image.

  inputs:
    img: input-image as numpy.array (shape: [H, W, C])
    boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
    labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
  '''

  cmap = plt.get_cmap("tab20b")
  class_labels = np.array(dataset)
  colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
  height, width, _ = img.shape
  # Create figure and axes
  fig, ax = plt.subplots(1, figsize=(16, 8))
  # Display the image
  ax.imshow(img)
  for i, box in enumerate(boxes):
    class_pred = labels[i]
    conf = scores[i]
    width = box[2] - box[0]
    height = box[3] - box[1]
    rect = patches.Rectangle(
        (box[0], box[1]),
        width,
        height,
        linewidth=2,
        edgecolor=colors[int(class_pred)],
        facecolor="none",
    )
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.text(
        box[0], box[1],
        s=class_labels[int(class_pred)] + " " + str(int(100*conf)) + "%",
        color="white",
        verticalalignment="top",
        bbox={"color": colors[int(class_pred)], "pad": 0},
    )

  # Used to save inference phase results
  if save_path is not None:
    plt.savefig(save_path)

  plt.show()


img = cv2.imread('api\default_1280-720-screenshot.webp')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
img_res = cv2.resize(img_rgb, (640, 640), cv2.INTER_AREA)
img_res /= 255.0

checkpoint_dir = f"{OUTPUT_DIR}/epoch__29__54__model.pth"
checkpoint = torch.load(checkpoint_dir, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

boxes, scores, labels = inference(img_res, model)

def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(orig_prediction[0], orig_prediction[1], iou_thresh)

    final_prediction = orig_prediction
    final_prediction[0] = final_prediction[0][keep]
    final_prediction[1] = final_prediction[1][keep]
    final_prediction[2] = final_prediction[2][keep]

    return final_prediction

# predictions = apply_nms(inference(img_res, model))
# predictions = inference(img_res, model)
# predictions = apply_nms(inference(img_res, model))

# print(predictions)
print(boxes)
print(scores)
print(labels)

def plot_img_bbox1(img, target):
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10, 10)
    a.imshow(img)
    for i, box in enumerate(target):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
#         if arr[target['labels'][i]] == 'ad':
        rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth = 2,
                                     edgecolor = 'r',
                                     facecolor = 'none')
        a.text(x, y-20, arr1[labels[i]], color='b', verticalalignment='top')

        a.add_patch(rect)
    plt.show()

plot_img_bbox1(img, boxes)