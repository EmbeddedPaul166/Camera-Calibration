#!/usr/bin/env python

import numpy as np
import cv2
import glob
import sys

nRows = 9
nCols = 6
dimensions_in_mm = 26

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimensions_in_mm, 0.001)

objp = np.zeros((nRows*nCols,3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

objpoints = []
imgpoints = []

images      = sorted(glob.glob("./images/mono/snapshot*.jpg"))

print("Processing images...")

counter_patterns = 0

for fname in images:
    if "calibresult" in fname:
        continue
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows), None, flags = flags)

    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        counter_patterns += 1

print("Done")

print("Patterns found: ", counter_patterns)

print("Calibrating camera...")

rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

np.savez("cameraCalibResults.npz", MAT=mtx, DIST=dist, RVECS=rvecs, TVECS=tvecs, RMS_ERROR=rms)

print("Done.")

print("RMS error: ", rms)

print("Results saved in .npz file. Use numpy.load() to use them inside your python application.")
 





