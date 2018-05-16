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

# Localhost
PREFIX = "/home/hadoop/VisionWorkspace/ava_taskB/"
# Server
#PREFIX = "/home/arpan/DATA_Drive/video_datasets/ava"
METAFILE_PATH = PREFIX + "info_files"
TRAIN_VAL_PATH = PREFIX + "data"
TEST_PATH = PREFIX + "test_data"

if __name__ == '__main__':
    
    filepath = os.path.join(METAFILE_PATH, "ava_test_v2.1.txt")
    
    with open(filepath, 'r') as fp:
        test_files = fp.readlines()
        
    test_files = [x.strip() for x in test_files]
    print test_files
    
    trainfile = os.path.join(METAFILE_PATH, "ava_train_v2.1.csv")

    df_train = pd.read_csv(trainfile, header=None, sep=',')
    
    tr_ids = df_train.loc[:, 0]  # get column 1  shape is (837318, )
    tr_ids = tr_ids.drop_duplicates() # shape is (235,)

    valfile = os.path.join(METAFILE_PATH, "ava_val_v2.1.csv")

    df_val = pd.read_csv(valfile, header=None, sep=',')
    
    val_ids = df_val.loc[:, 0]  # get column 1  shape is (837318, )
    val_ids = val_ids.drop_duplicates() # shape is (235,)

    
    #datafile = '/home/hadoop/VisionWorkspace/ava_taskB/data.txt'
    #with open(datafile, 'r') as fp:
    #    tr_files = fp.readlines()
        
    #tr_files = [x.strip() for x in tr_files]
    #tr_files = [x.split('.')[0] for x in tr_files]
    #print tr_files
    
    #list(set(tr_files) - set(tr_ids) - set(val_ids))