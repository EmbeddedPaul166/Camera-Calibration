#!/usr/bin/env python

import cv2
import glob
import numpy as np

#Parameters
w = 9
h = 6
square_dimension_in_mm = 26
accuracy = 0.001

#Script
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, square_dimension_in_mm, accuracy)

objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

objpoints_l = [] 
objpoints_r = [] 
objpoints_p = [] 
imgpoints_l = []
imgpoints_r = [] 
imgpoints_lp = []
imgpoints_rp = [] 

image_paths_l = sorted(glob.glob("./images/left/left*.jpg"))   
image_paths_r = sorted(glob.glob("./images/right/right*.jpg")) 

print("Processing image pairs...")

left_count = 0
right_count = 0
pair_count = 0

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

for (image_path_l, image_path_r) in zip(image_paths_l, image_paths_r):
    image_l = cv2.imread(image_path_l)
    image_r = cv2.imread(image_path_r)

    gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (w, h), None, flags = flags)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (w, h), None, flags = flags)


    if ret_l == True and ret_r == True:
        objpoints_p.append(objp)
        
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1,-1), criteria)
        imgpoints_lp.append(corners_l2)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1,-1), criteria)
        imgpoints_rp.append(corners_r2)
        
        pair_count += 1
        
    elif ret_l == True:
        objpoints_l.append(objp)
        
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1,-1), criteria)
        imgpoints_l.append(corners_l2)

        left_count += 1
        
    elif ret_r == True:
        objpoints_r.append(objp)
        
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1,-1), criteria)
        imgpoints_r.append(corners_r2)

        right_count += 1
        
    else:
        continue

image_size_left = tuple(gray_l.shape[::-1])

image_size_right = tuple(gray_r.shape[::-1])

print("Done")

print("Correct left images: ", left_count)

print("Correct right images: ", right_count)

print("Correct image pairs: ", pair_count)

print("Calibrating individual cameras...")

flags = 0

rms_left, mat_left, dist_left, r_left, t_left = cv2.calibrateCamera(objpoints_l, imgpoints_l, image_size_left, None, None, flags=flags)
rms_right, mat_right, dist_right, r_right, t_right = cv2.calibrateCamera(objpoints_r, imgpoints_r, image_size_right, None, None, flags=flags)

np.savez("leftCamCalibResults.npz", MAT_LEFT=mat_left, DIST_LEFT=dist_left, R_LEFT=r_left, T_LEFT=t_left, RMS_ERROR_LEFT=rms_left)
np.savez("rightCamCalibResults.npz", MAT_RIGHT=mat_right, DIST_RIGHT=dist_right, R_RIGHT=r_right, T_RIGHT=t_right, RMS_ERROR_RIGHT=rms_right)

print("Done.")

print("Left camera RMS error: ", rms_left)

print("Right camera RMS error: ", rms_right)

print("Calibrating stereo...")

flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_SAME_FOCAL_LENGTH

rms, mat_left, dist_left, mat_right, dist_right, R, T, E, F = cv2.stereoCalibrate(objpoints_p, imgpoints_lp, imgpoints_rp, mat_left, dist_left, mat_right, dist_right, image_size_left,
criteria = criteria, flags = flags)

np.savez("stereoCalibResults.npz",MAT_LEFT=mat_left, MAT_RIGHT=mat_right, DIST_LEFT=dist_left, DIST_RIGHT=dist_right, R=R, T=T, E=E, F=F, RMS_ERROR=rms)

print("Done.")

print("Stereo RMS error: ", rms)

print("Rectifying images...")

r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(mat_left, dist_left, mat_right, dist_right, image_size_left, R, T, alpha=0, newImageSize=image_size_left)

np.savez("stereoRectResults.npz", MAT_LEFT=mat_left, MAT_RIGHT=mat_right, DIST_LEFT=dist_left, DIST_RIGHT=dist_right, R_LEFT=r_left, R_RIGHT=r_right, P_LEFT=p_left, P_RIGHT=p_right, ROI_LEFT=roi_left, ROI_RIGHT=roi_right, Q=q, RMS_ERROR=rms)

print("Done.")

print("Results are stored in .npz files. Use numpy.load() to use them inside your python application.")



