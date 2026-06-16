import cv2
import numpy as np
import math
import random
import time
import sys
import operator
import os
from numpy import zeros, newaxis
import re
import glob

from util.Generate_pm_pa import *
from util.UAV_subfunctions import *
from util.Extract_Patch import *
from util.Detect_Patch import *

import types
import joblib
import joblib.numpy_pickle
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Monkeypatching for old scikit-learn pickles
if not hasattr(sklearn, 'externals'):
    sklearn.externals = types.ModuleType('sklearn.externals')
    sys.modules['sklearn.externals'] = sklearn.externals
sklearn.externals.joblib = joblib
sys.modules['sklearn.externals.joblib'] = joblib

import sklearn.ensemble._weight_boosting as weight_boosting
sys.modules['sklearn.ensemble.weight_boosting'] = weight_boosting
import sklearn.tree._classes as tree_classes
sys.modules['sklearn.tree.tree'] = tree_classes

original_find_class = joblib.numpy_pickle.NumpyUnpickler.find_class
def new_find_class(self, module, name):
    if module == 'sklearn.ensemble.weight_boosting':
        module = 'sklearn.ensemble._weight_boosting'
    elif module == 'sklearn.tree.tree':
        module = 'sklearn.tree._classes'
    return original_find_class(self, module, name)
joblib.numpy_pickle.NumpyUnpickler.find_class = new_find_class

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

def run_prediction(input_video, output_video, max_frames=0):
    app_model_path = './models/max500_1_10_threelayers/'
    app_model_path_track = './models/Appearance_OriImage/'
    mvmodel_path = './models/motion/'
    bimodel_path = './models/Adaboost/'
    bimodel_path_track = './models/Adaboost_track/'

    ind = 1 # Use fold 1
    print(f"Loading models for fold {ind}...")
    appmodel = load_model(app_model_path + str(ind) + '.h5')
    appmodel_track = load_model(app_model_path_track + str(ind) + '.h5')
    mvmodel = load_model(mvmodel_path + str(ind) + '.h5')
    combinemodel = joblib.load(bimodel_path + 'fold' + str(ind) + '.pkl')
    combinemodel_track = joblib.load(bimodel_path_track + 'fold' + str(ind) + '.pkl')

    a = 0.001
    b = 50
    maxD = 4
    track_len = 10
    detect_interval = 6
    radius = 10
    lamda = 0
    use_ransac = True

    feature_params = dict(maxCorners=600, qualityLevel=0.005, minDistance=25, blockSize=5)
    lk_params = dict(winSize=(19, 19), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    feature_params_track = dict(maxCorners=500, qualityLevel=a/20.0, minDistance=b, blockSize=9)
    lk_params_track = dict(winSize=(19, 19), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03), minEigThreshold=1e-4)
    lk_params_track_ori = dict(winSize=(25, 25), maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03), minEigThreshold=1e-4)
    feature_params_Detect = dict(maxCorners=10, qualityLevel=0.00000015, minDistance=0, blockSize=3)

    cam = cv2.VideoCapture(input_video)
    if not cam.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    ret, color = cam.read()
    if not ret or color is None:
        print("Error: Could not read first frame")
        return

    frameidx = 1
    h, w, channel = color.shape
    prepreFrame = np.float32(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or np.isnan(fps): fps = 30.0
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    pImg = None
    H_back = None

    ret, color = cam.read()
    if not ret or color is None:
        print("Error: Could not read second frame")
        return
    
    frameidx += 1
    Xtminus1 = np.float32(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    blocks = np.ones((h, w), dtype='float32')
    
    Dotft = []
    Patchft = []
    maxPatchId = 0
    
    # Mock ground truth related variables
    gt_mask = np.zeros((h, w), dtype='uint8')
    gt_ft_maske = np.zeros((h, w), dtype='uint8')
    centers = []

    print("Processing video...")
    while True:
        gray = Xtminus1.copy()
        ret, future_color = cam.read()
        if not ret or future_color is None:
            break
        
        frameidx += 1
        if frameidx % 100 == 0:
            print(f"Frame {frameidx}")
            
        if max_frames > 0 and frameidx > max_frames:
            print(f"Reached max_frames limit: {max_frames}")
            break
            
        Xt = np.float32(cv2.cvtColor(future_color, cv2.COLOR_BGR2GRAY))
        oriImage = future_color.copy()

        if pImg is None or frameidx % track_len == 0:
            pImg = cv2.goodFeaturesToTrack(np.uint8(gray), **feature_params)
            pImg = maskOut(blocks, pImg)

        if (frameidx) % detect_interval == 0:
            weightedError, H_back, pImg = backgroundsubtraction(gray, prepreFrame, Xt, pImg, blocks, lamda, lk_params, use_ransac)
        else:
            H_back, pImg = backgroundMotion(gray, prepreFrame, Xt, pImg, blocks, lamda, lk_params, use_ransac)

        if len(Dotft) > 0:
            d1, d, p1, pPers, p0, st1, ft_mv, ft_app, gt_labels_dummy = generatePatches_MV_trackV1(Dotft, gray, Xt, H_back, lk_params_track_ori, radius, w, h, color, gt_ft_maske)
            score_mv = mvmodel.predict(ft_mv, batch_size=1000000)
            score_app = appmodel_track.predict(ft_app, batch_size=2560)
            
            bifeature = np.hstack([score_app[:, 0].reshape(-1, 1), score_mv[:, 0].reshape(-1, 1)])
            trst = combinemodel_track.predict(bifeature)
            
            Dotft, indrm = dotupdate(Dotft, Patchft)
            oriImage = visDotft(oriImage, Dotft, w, h)
            Dotft = dottrack_detect(Dotft, p1[indrm], pPers[indrm], trst[indrm], st1[indrm], d1[indrm], d[indrm], Patchft)
            oriImage = visPtV1(oriImage, p0, st1, d1)

        if len(Patchft) > 0:
            oriImage = visDetect_Kalman(Patchft, oriImage, radius, w, h)
            Patchft = patch_KalmanTracking(Dotft, Patchft, H_back, w, h)

        if (frameidx) % detect_interval == 0:
            mv, detectedPatches, errorPatches, gt_labels_dummy, detectedLocs, curLocslll, hit, ftNo, FAno = Extract_Patch(frameidx, gray, Xt, weightedError, centers, H_back, feature_params_track, feature_params_track, lk_params_track, radius, color, future_color, gt_mask, oriImage)
            
            if mv.shape[0] > 0:
                errorPatches = errorPatches[:, :, :, newaxis]
                mv = np.hstack([mv[:, 4:6], mv[:, 10:]])
                data_np_test = np.concatenate([detectedPatches/255.0, errorPatches/255.0, errorPatches/255.0, errorPatches/255.0], axis=3)
                test_output_app = appmodel.predict(data_np_test, batch_size=2560)
                test_output_mv = mvmodel.predict(mv, batch_size=1000000)
                
                mvmafeature = np.hstack([test_output_app[:, 0].reshape(-1, 1), test_output_mv[:, 0].reshape(-1, 1)])
                dt_lable = combinemodel.predict(mvmafeature)
                
                oriImage = visPosPatch_Kalman(dt_lable, gt_labels_dummy, detectedLocs, oriImage, radius)
                oriImage, Dotft, Patchft, maxPatchId = DetectOnX_V2(maxD, maxPatchId, oriImage, gray, Xt, lk_params_track_ori, H_back, detectedLocs, curLocslll, dt_lable, detectedPatches, feature_params_Detect, radius, Dotft, Patchft)

        draw_str(oriImage, 20, 60, 'frame ID: %d' % (frameidx-1))
        video_writer.write(oriImage)
        
        prepreFrame = Xtminus1.copy()
        color = future_color.copy()
        Xtminus1 = Xt.copy()

    cam.release()
    video_writer.release()
    print(f"Done! Output saved to: {output_video}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()
    run_prediction(args.input, args.output, args.max_frames)
