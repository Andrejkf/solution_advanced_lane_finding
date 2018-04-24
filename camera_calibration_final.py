#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:23:27 2018

@author: maxwell
"""

# cd '/home/maxwell/ownCloud/v0CarND-Advanced-Lane-Lines-master'

import numpy as np
import cv2
import glob
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# paths 
folder_path= './camera_cal/'


# chess dimension (9x6)
nx = 9
ny = 6

# this is used to make work corners detection on images
nxy = [(9,5), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (9,6), (5,6), (7,6), (9,6), (9,6), (9,6), (9,6)]

# read file location of images_for_calibration
imgNames = sorted(glob.glob( os.path.join(folder_path,'calib*.jpg') ))
#images.sort(key=str.title)


class CalibratorCamera:
    '''
    Used to get Camera matrix and distortion coefficients from 9x6 chessboard for project 4.
    '''
    
    
    def __init__(self, nxy, imgNames, folder_path):
        '''
        Used to initialize variables from Calibratorcamera class
        '''
        self.nxy = nxy
        self.img_names = imgNames
        self.folder_path = folder_path
                
        self.flag_find_calibration_numbers = False
    
    def find_calibration_numbers(self):
        '''
        Returns dist (distortion coeficients) and mtx (camera matrix) and save then in a pickle file.
        
        nxy: (list of tuples) vector with number of corners for each chess image. Used as input for camera calibration.
        imgNames: (list of str) list with location of all images used for camera calibration.
        folder_path_out: output folder path to save "dist" and "mtx" in a pickle file.
        '''
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        for i, fname in enumerate(imgNames):
            nx = self.nxy[i][0]
            ny = self.nxy[i][1]
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros( (nx*ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
            # 1. Read image
            img = cv2.imread(fname)
            # 2. convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 3. get chessboard corners and confimation flag
            ret, corners = cv2.findChessboardCorners(gray, self.nxy[i], None)
            # 4. if corners where correctly found , then append chess_corners
            if ret == True:
                
                objpoints.append(objp)
                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(img, self.nxy[i], corners, ret)
            else:
                print('Error in function find_corners at image',i)
            
        img_size = (img.shape[1], img.shape[0]) # fancy way to get the first two items of shape ( in reversed order)
        
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        
        self.mtx    = mtx
        self.dist   = dist
        self.flag_find_calibration_numbers = True
        
    
    def save_coefficients(self, fileName="wide_dist_pickle.p"):
        '''
        Saves the calculated camera matrix = mtx, and the coefficients = dst used on previous calibration for the camera ( find_calibration_numbers() ).
        
        fileName: (str) name for the output pickle file. Example: "wide_dist_pickle.p"
        '''
        # save coefficients and camera matrix in a pickle file
        dist_pickle = {}
        dist_pickle['mtx'] = self.mtx
        dist_pickle['dist'] = self.dist
        pickle.dump( dist_pickle, open( os.path.join(self.folder_path, fileName) , "wb" ) )
        if os.path.exists(os.path.join(self.folder_path, fileName)):
            return True
    
    def load_coefficients(self, fileName="wide_dist_pickle.p"):
        '''
        Loads the calculated camera matrix = mtx, and the coefficients = dst used on previous calibration for the camera ( find_calibration_numbers() ).
        
        fileName: (str) name for the output pickle file. Example: "wide_dist_pickle.p"
        '''
        assert os.path.exists(os.path.join(self.folder_path, fileName)) == True
        # load coefficients and camera matrix from pickle file
        dist_pickle = pickle.load( open(os.path.join(self.folder_path, fileName), "rb"))
        self.mtx = dist_pickle['mtx']
        self.dist= dist_pickle['dist']
        return True
    
    def get_coefficients(self, doFlag=False, getValuesFlag=True):
        '''
        Gets radial undistorted images
        
        self.nxy: (list of tuples) vector with number of corners for each chess image. Used as input for camera calibration.
        self.imgNames: (list of str) list with location of all images used for camera calibration.
        doflag: (Bool) to specifies if coefficients are calculated or loaded from pickle data
        getValuesFlag: (Bool) to espcify if you want to inmediately retrieve "mtx" and "dist" values
        '''
        if (doFlag == False):
            self.load_coefficients()
        else:
            self.find_calibration_numbers()
        if (getValuesFlag ==True):
            return self.mtx, self.dist
    
    def get_undistorted(self, img):
        '''
        Gets undistorted images
    
        img: input image to correct radial distortion
        self.flag_find_calibration_numbers: (Bool). Flag from the method find_calibration_numbers() used in here to choose if used calculatd values from pickle file or re-calculate-them
        self.mtx:  matrix corection. Used for correction
        self.dist: coefficients used for correction
        '''
        condition = (self.find_calibration_numbers == False)
        if condition:
            self.find_calibration_numbers()
        
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistorted
    


         
def draw_corners(nxy, images):
    '''
    Returns an aray of images with detected corners plottled.
    
    nxy: (list of tuples) vector with number of corners for each chess image. Used as input for camera calibration.
    images: (list of str) list with location of all images used for camera calibration.
    '''
    import cv2
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    imgsCorners = []
    for i, fname in enumerate(images):
        #print(i, fname)
        nx = nxy[i][0]
        ny = nxy[i][1]
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros( (nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, nxy[i], None)
        
        # correction in the corrdinate of the first corner found for 9th image :
        #if i == 8:
        #    corners[0] = [402.0, 298.0]
        
        # if corners where correctly found , then append chess_corners
        if ret == True:
            
            objpoints.append(objp)
            imgpoints.append(corners)
            
            cv2.drawChessboardCorners(img, nxy[i], corners, ret)
            imgsCorners.append(img)
        else:
            print('Error in function draw_corners at image',i)
    
    return imgsCorners
    

def show_images(imgs, tdelay= 1000):
    '''
    Iterator to Show in cv2 the lsit of images given
    imgs: (list of npuin8) list of images to display
    tdelay: (int) time to delay (in miliseconds) between images
    '''
    for i in range(len(imgs)):
        cv2.namedWindow('resized_window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('resized_window', 600, 600)
        cv2.imshow('resized_window', imgs[i])
        cv2.waitKey(tdelay)
    cv2.waitKey(2*tdelay)
    cv2.destroyAllWindows()
    


# Part 1 Camera calibration. Get calibration coefficients and save them in a pickle file
# create an instance to calibrate camera, and give nx and ny for each image.
# include input image names for camera calibration.
# included as an argument camera_cal folder path
w = CalibratorCamera(nxy=nxy,imgNames=imgNames, folder_path=folder_path )
# Do camera calibration and return camera matrix and distrotion coefficients. (ret, mtx, dist, rvecs, tvecs )
w.find_calibration_numbers()
# save coefficients in a pickle file to use them for calibration in the car project lane detection.
# Note: this pickle file is saevd in the same folder: camera_cal with the chess images provided to calibrate camera
w.save_coefficients(fileName='wide_dist_pickle.p')



# Part 2 Camera calibration. upload camera matrix and distortion coefficients.
# apply distortion correction to chess images.

w2 = CalibratorCamera(nxy=nxy,imgNames=imgNames, folder_path=folder_path )
w2.load_coefficients(fileName='wide_dist_pickle.p')


# draw inage corners
img_corners = []
for i,img_name in enumerate(imgNames):
    a = draw_corners(nxy=nxy, images=imgNames)
    plt.imshow(a[i])
    img_corners.append(a[i])



# keep  original and udistorted images in the list
original = []
undistorted = []
for i,img_name in enumerate(imgNames):
    in_img = mpimg.imread(img_name)
    out_img = w2.get_undistorted(in_img)
    
    original.append( in_img )
    undistorted.append( out_img )

    

# Plot  original and undistorted images
for i,img_name in enumerate(imgNames):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(original[i])
    plt.title('original image')
    plt.subplot(1,2,2)
    plt.imshow(undistorted[i])
    plt.title('undistorted image')
    plt.pause(1)
    