#!/usr/bin/env python
import numpy as np
import json
import os

# cfg = json.load(open('../config.json','r'))
cfg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json'),'r'))

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 450


def bbox_error_reward(gt_bbox, pred_bbox):
    (x,y,w,h) = gt_bbox
    if w*h == 0:
        return -1
    err = 0
    gt_bbox[0] += gt_bbox[2]/2
    gt_bbox[1] += gt_bbox[3]/2
    pred_bbox[0] += pred_bbox[2]/2
    pred_bbox[1] += pred_bbox[3]/2
    for i in range(len(gt_bbox)):
        err += (gt_bbox[i]-pred_bbox[i])**2
    return -err/(300*300*4)

def gt_bbox_reward(gt_bbox):
    (x,y,w,h) = gt_bbox
    reward = w*h
    done = False if w*h == 0 else True
    return reward, done




def pred_bbox_reward(gt_bbox,pred_bbox):
    (x1,y1,w1,h1) = gt_bbox
    x1 -= (1.0*w1)/2
    y1 -= (1.0*h1)/2
    (x2,y2,w2,h2) = pred_bbox
    x2 -= (1.0*w2)/2
    y2 -= (1.0*h2)/2
    if x2+w2<=x1 or x2>=x1+w1 or y2+h2<=y1 or y2>=y1+h1:
        return 0,False
    x=sorted([x1,x1+w1,x2,x2+w2])
    overlap_w = x[2]-x[1]
    y=sorted([y1,y1+h1,y2,y2+h2])
    overlap_h = y[2]-y[1]
    overlap_area = overlap_w*overlap_h
    max_area = max(w1,w2)*max(h1,h2)
    reward = overlap_area/max_area
    done = False if reward<0.6 else True
    return reward, done


def get_threshod(scene_name, target_name, mode='min', use_semantic=False, use_gt=False):
    if target_name == 'background':
        return IMAGE_HEIGHT*IMAGE_WIDTH
    if use_semantic:
        targets_path = '%s/Environment/houses/%s/targets_info_semantic.json' % (cfg['codeDir'], scene_name)
    elif use_gt:
        targets_path = '%s/Environment/houses/%s/targets_info_all.json' % (cfg['codeDir'], scene_name)
    else:
        targets_path = '%s/Environment/houses/%s/targets_info_all_pred.json'%(cfg['codeDir'],scene_name)
    targets_info = json.load(open(targets_path,'r'))
    if not target_name in targets_info:
        return IMAGE_HEIGHT*IMAGE_WIDTH
    if mode == 'max':
        i = 0
    else:
        i = len(targets_info[target_name]) - 1
    threshold = targets_info[target_name][i][-1]
    while threshold == 0 and i != 0:
        i -= 1
        threshold = targets_info[target_name][i][-1]
    if threshold == 0:
        threshold = IMAGE_HEIGHT*IMAGE_WIDTH
    return threshold


def intuitive_reward(scene_name, target_name, bbox):
    threshold = get_threshod(scene_name, target_name)
    (x,y,w,h) = bbox
    area = w*h
    if area < threshold:
        return -0.01, False
    else:
        return 10, True

def increasing_bbox_reward(scene_name,target_name,prev_area, curr_area, use_gt=False):
    if scene_name == 'real':
        min_threshold = max_threshold = 6000
    else:
        min_threshold = get_threshod(scene_name, target_name, use_gt=use_gt)
        max_threshold = get_threshod(scene_name, target_name, use_gt=use_gt, mode='max')
    reward = 0
    done = False if curr_area < min_threshold else True
    if curr_area > prev_area:
        reward = (1.0*curr_area)/max_threshold
    return reward, done


def increasing_area_reward(scene_name,target_name, prev_area, curr_area, resize=(10,10)):
    threshold = get_threshod(scene_name, target_name, use_semantic=True)
    threshold = (1.0*threshold * resize[0] * resize[1]) / (IMAGE_WIDTH * IMAGE_HEIGHT)
    #threshold = 0.8*threshold
    reward = 0
    if threshold == 0:
        print scene_name
        print target_name
    done = False if curr_area < threshold else True
    if curr_area > prev_area:
        reward = (100.0*curr_area)/threshold
    return reward, done, threshold
    
    
