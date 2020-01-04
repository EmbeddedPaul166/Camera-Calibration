#!/usr/bin/env python

import numpy as np
import cv2
import glob
import sys

nRows = 9
nCols = 6
dimension = 26

workingFolder   = "images/"
imageType       = "jpg"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

objp = np.zeros((nRows*nCols,3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

objpoints = []
imgpoints = []

filename    = workingFolder + "/*." + imageType
images      = glob.glob(filename)

nPatternFound = 0

for fname in images:
    if "calibresult" in fname:
        continue
    img     = cv2.imread(fname)
    gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print("Pattern found!")
        nPatternFound += 1
        objpoints.append(objp)
        imgpoints.append(corners2)

print("Found %d patterns" % (nPatternFound))
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("RMS error: ", rms)

print("Results saved in .npz file. Use numpy.load() to use them inside your python application.")
 
np.savez("cameraCalibResults.npz", MAT=mtx, DIST=dist, RVECS=rvecs, TVECS=tvecs, RMS_ERROR=rms)





