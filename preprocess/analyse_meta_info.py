#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:53:50 2018

@author: Arpan
@Description: Script to explore the AVA dataset files
"""

import json
import os
import numpy as np
import pandas as pd
import cv2

# Localhost
PREFIX = "/home/hadoop/VisionWorkspace/ava_taskB/"
META_PATH = PREFIX + "scripts/meta_files"
# Server
if os.path.exists("/home/arpan/DATA_Drive/video_datasets/ava/"):
    PREFIX = "/home/arpan/DATA_Drive/video_datasets/ava/"
    META_PATH = "/home/arpan/VisionWorkspace/ava_taskB/meta_files"
INFOFILES_PATH = PREFIX + "info_files"
TRAIN_VAL_PATH = PREFIX + "data"
TEST_PATH = PREFIX + "test_data"
LABS_FNAME = "ava_action_list_v2.1_for_activitynet_2018.pbtxt"


# get the unnormalized bounding box values, given the frame dimensions (H, W)
# bbox: [x1, y1, x2, y2]
# dims: [H, W]
def get_unnorm_bbox(bbox, dims):
    x1 = int(bbox[0]*dims[1])
    y1 = int(bbox[1]*dims[0])
    x2 = int(bbox[2]*dims[1])
    y2 = int(bbox[3]*dims[0])
    return (x1, y1, x2, y2)

# extract the given frame from srcVideo 
# bbox: is the normalized values of the bounding box
# vmeta is the meta information dictionary of the form
# {'nFrames':12395.0, 'fps':29.9382, 'dimensions':[360, 640]}
def get_frame_from_video(srcVid, fno, bbox, label, vmeta):
    cap = cv2.VideoCapture(srcVid)
    if not cap.isOpened():
        import sys
        print "Capture object not opened! Abort"
        sys.exit(0)
        
    cap.set(cv2.CAP_PROP_POS_MSEC, fno*1000)
    ret, frame = cap.read()
    if ret:
        bbox = get_unnorm_bbox(bbox, vmeta['dimensions'])
        cv2.rectangle(frame, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,255,0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        
    cap.release()
    cv2.destroyAllWindows()
    return
    

# parse the labels file line by line and get the values in the dictionary
def get_labels(labelsfile):
    flag = False
    d = {}
    key, val = None, None
    with open(labelsfile) as f:
        for line in f:
            # if 'item' occurs in the line
            if not flag and 'item {' in line:
                flag = True
                continue
            if flag and '}' in line:
                d[key] = val
                key, val = None, None
                flag = False
                continue
            if flag:
                if 'name:' in line:
                    val = line.strip().split(':')[-1].strip().strip("\"")
                if 'id:' in line:
                    key = int(line.strip().split(':')[-1].strip())
    return d


if __name__ == '__main__':
    
    filepath = os.path.join(INFOFILES_PATH, "ava_test_v2.1.txt")
    
    with open(filepath, 'r') as fp:
        test_files = fp.readlines()
        
    test_files = [x.strip() for x in test_files]
    print test_files
    
    trainfile = os.path.join(INFOFILES_PATH, "ava_train_v2.1.csv")

    col_names = ['vid', 'fno', 'x1', 'y1', 'x2', 'y2', 'label']
    df_train = pd.read_csv(trainfile, header=None, sep=',', names=col_names)
    #df_train.columns = col_names
    
    tr_ids = df_train.loc[:, 'vid']  # get column 1  shape is (837318, )
    tr_ids = tr_ids.drop_duplicates() # shape is (235,)

    valfile = os.path.join(INFOFILES_PATH, "ava_val_v2.1.csv")

    df_val = pd.read_csv(valfile, header=None, sep=',', names=col_names)
    #df_val.columns = col_names
    
    val_ids = df_val.loc[:, 'vid']  # get column 1  shape is (837318, )
    val_ids = val_ids.drop_duplicates() # shape is (235,)

    ###########################################################################
    # read meta files 
    with open(os.path.join(META_PATH, "meta_trainval.json"), 'r') as fobj:
        meta_trainval = json.load(fobj)
    with open(os.path.join(META_PATH, "meta_test.json"), 'r') as fobj:
        meta_test = json.load(fobj)
    ###########################################################################
    # Read the labels from file
    labs_path = os.path.join(INFOFILES_PATH, LABS_FNAME)
    labels = get_labels(labs_path)      # 60 labels for activitynet
    #print labels    
    ###########################################################################    
    keys_list = meta_trainval.keys()
    for i in range(df_train.shape[0]):
        srcVid = df_train.loc[i,'vid']
    
        srcVid_idx = [idx for idx,k in enumerate(keys_list) if srcVid in k][0]
        srcVidPath = os.path.join(TRAIN_VAL_PATH, keys_list[srcVid_idx])
        print "Vid : {} :: Label : {}".format(srcVid, labels[df_train.loc[i]['label']])
        
        bbx = (df_train.loc[i]['x1'], df_train.loc[i]['y1'], \
               df_train.loc[i]['x2'], df_train.loc[i]['y2'])
        get_frame_from_video(srcVidPath, df_train.loc[i]['fno'], bbx, \
                             df_train.loc[i]['label'], meta_trainval[keys_list[srcVid_idx]])
        
    #datafile = '/home/hadoop/VisionWorkspace/ava_taskB/data.txt'
    #with open(datafile, 'r') as fp:
    #    tr_files = fp.readlines()
        
    #tr_files = [x.strip() for x in tr_files]
    #tr_files = [x.split('.')[0] for x in tr_files]
    #print tr_files
    
    #list(set(tr_files) - set(tr_ids) - set(val_ids))
    
    ###########################################################################
    