#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:00:08 2017

@author: hadoop
"""
import cv2
import os
import json

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
notOpened, opened = 0, 0

def get_meta_data(srcVid):
    global notOpened, opened
    cap = cv2.VideoCapture(srcVid)
    if not cap.isOpened():
        notOpened += 1
        print "Not Opened !!! "+srcVid
        return
    dimensions = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    ret, frame = cap.read()
    opened+=1
    print "{} File: {}".format(opened, srcVid)
    print "\t Dim: {}, FPS: {}, #Frames: {}, ret: {}" \
                 .format(dimensions, fps, nFrames, ret)             
#    fcount = 0
#    # Iterate over all the frames of the video.
#    while cap.isOpened():
#        ret, frame = cap.read()
#        print "fcount = {}".format(fcount)
#        fcount+=1
#        if not ret:
#            print "Frame not returned."
    
    cap.release()
    return {'nFrames':nFrames, 'fps':fps, 'dimensions':dimensions}

if __name__=="__main__":
    vid_files = os.listdir(TRAIN_VAL_PATH)
    d_train = {}
    for vid in vid_files:
    # iterate over files
        d_train[vid] = get_meta_data(os.path.join(TRAIN_VAL_PATH, vid))
    print "Files Opened (Train/Val): {}, Not Opened: {}".format(opened, notOpened)
    
    destfilename = os.path.join(META_PATH, "meta_trainval.json")
#    with open(destfilename, 'w') as fobj:
#        json.dump(d_train, fobj)
#    print "Written trainval meta-info to JSON file."
    
    vid_files = os.listdir(TEST_PATH)
    d_test = {}
    opened, notOpened = 0, 0
    for vid in vid_files:
    # iterate over files
        d_test[vid] = get_meta_data(os.path.join(TEST_PATH, vid))
    print "Files Opened (Test): {}, Not Opened: {}".format(opened, notOpened)
    
    destfilename = os.path.join(META_PATH, "meta_test.json")
#    with open(destfilename, 'w') as fobj:
#        json.dump(d_test, fobj)
#    print "Written test meta-info to JSON file."
    