## Project 4 term1. Advanced lane Finding

### Note: I used template for my report.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Student Notes

Images for camera calibration are localted on folder **./output_images** . This folder contains:
* [a_chess_corners](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/a_chess_corners "a_chess_corners"): Folder with the camera calibration images 9x6 inner corners drawn on it.

* [b_undistorted_chess_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/b_undistorted_chess_images "b_undistorted_chess_images"): Folder comparing orginal and undistorted chess images in it.

* [c_undistorted_car_test_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/c_undistorted_car_test_images "c_undistorted_car_test_images"): Folder comparing orginal and undistorted test images from the driving car.

* [cx_car_gray_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/cx_car_gray_images "cx_car_gray_images"): Folder showing binary images from gray images.
* [d_car_s_channel_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/d_car_s_channel_binary_images "d_car_s_channel_binary_images"): Folder with binary images from s_channel of images in HLS color space.
* [e_car_L_channel_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/e_car_L_channel_binary_images "e_car_L_channel_binary_images"): Folder with binary images from L_channel of images in HLS color space.
* [f_car_sobelx_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/f_car_sobelx_binary_images "f_car_sobelx_binary_images"): Folder with binary images after applying gradients in x direction.
* [g_car_sobely_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/g_car_sobely_binary_images "g_car_sobely_binary_images"):
Folder with binary images after applying gradients in y direction.
* [h_car_combined_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/h_car_combined_binary_images "h_car_combined_binary_images"): Folder with combined binary images.
* [i_car_roi_combined_binary_images](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/i_car_roi_combined_binary_images "i_car_roi_combined_binary_images"): Folder with Region Of Interest selection applied to binary images.
* [j_car_lanelines_on_road](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/j_car_lanelines_on_road "j_car_lanelines_on_road"): Folder with lane lines dispayed on it.
* [k_car_bird_eye](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/k_car_bird_eye "k_car_bird_eye"): Folder with images after persepective transformation to get bird_eye images.
* [l_car_full_pipeline](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/l_car_full_pipeline "l_car_full_pipeline"): Folder with images after applying full process pipeline.
* [m_car_histograms_bird_eye](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/tree/master/output_images/m_car_histograms_bird_eye "m_car_histograms_bird_eye"): Folder with stimated lane lines from convolution between histogram vector and wavelets.

The images in **camera_cal** are for camera calibration.

The images in **test_images** are the test images provided for testing the pipeline.







### 1.  Camera Calibration
To get the camera matrix and distortion coefficients in file *camera_calibration_final.py (also in notebook cell 6).*  class *CalibratorCamera* was defined. It has the following funtions:
* 1.1 find_calibration_numbers(): Used to find chess board corners in grayscaled images and then to get camera matrix and distortion coefficients.
* 1.2. save_coefficients(): to save camera matrix and distortion coefficients in a pickle with  fileName='wide_dist_pickle.p'.
* 1.3. load_coefficients() to load the previously saved information to make process faster for the pipeline on the video instead of calculating them again.
* 1.4. get_coefficients(): Used to either load distortion coefficientes or to calculate them.
* * 1.5. get_undistorted(): To apply distortion correction to images.

##### Steps
1. Get inner corners.
1.1 draw detected corners for debbugging purposes.
2. find the chess inner corners using cv2.findChessboardCorners.
3. Get a list of object points and image points.
4. get camera matrix and distortion coeffients using cv2.calibrateCamera.
5. apply camera matrix and distortion correction to chess images.

These steps are ilustrated below:
<br/> ![alt text][image1]
<br/> ![alt text][image2]







The goals / steps of this project are the following:



-   Apply a perspective transform to rectify binary image (“birds-eye view”).
-   Detect lane pixels and fit to find the lane boundary.
-   Determine the curvature of the lane and vehicle position with respect to center.
-   Warp the detected lane boundaries back onto the original image.
-   Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Pipeline (single images)

#### 1. distortion-corrected image.

##### Provide an example of a distortion-corrected image.

After applying distortion correction to the chess board images I saved the camera matrix and distortion coefficients in a pickle file ([wide_dist_pickle.p](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/blob/master/wide_dist_pickle.p "wide_dist_pickle.p")). Then I loaded the same matrix and distortion coefficients and applied them to the raw test images provided in folder *./test_images*. Below there is an ilustration:
<br/> ![alt text][image3]






#### 2. Thresholded binary image

##### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Initially I tried to use a color combination using a yellow_white color mask between HSV and HSL color spaces, but then I noticed that the S channel and L channel can be used separately to generate binary images. Also I tested sobel gradients in x direction, y direction, magnitud threshold and angle direction thresholds but for simplicity I used gradients in x direction.

Note: I believe that x_treshold, y_threshold, magnitude_threshol and direction_threshold can be used altogether to get less noisy images. Even if I did not combine them. Those functions are in cells 32,33,34 from jupyter notebook and I was inspired on class exercises to write them.

The values of threshold used where tunned by trial and error looking to reduce all possible points/pixels that might introduce noise in the predictions.

All that said, I defined the following functions:
* hls_wyMask(). (cell 27 jupyter notebook). Used to apply a color white-yellow mask in HLS_color_space while keeping threshold values from project 1.
* hsv_wyMask(): (cell 28 jupyter notebook). Used to apply a color white-yellow mask in HSV_color_space.
* gray_binary(): (cell 29 jupyter notebook). Used to get a binary thresholded image from images im grayscale. I tried using global  histogram equalization but I did not included in this pipeline.
* hls_channel_binary(): (cell 30 jupyter notebook). Used to apply binary threshold to S or L channel of image in HSL color space.
* abs_sobel_thresh():  (cell 32 jupyter notebook). Used to apply sobel gradients in X or Y direction using cv2.Sobel to gray images. 
* mag_thresh():  (cell 33 jupyter notebook). Used to apply Sobel L2 norm using cv2.Sobel to gray images. 
* dir_threshold():  (cell 34 jupyter notebook). Used to apply Sobel directional threshold using cv2.Sobel to gray images.
* binary_threshold():  (cell 37 jupyter notebook). Used to generate the combined binary thresholded image.
An image to ilustrate the process is shown below:

<br/> ![alt text][image4]
<br/> ![alt text][image5]
<br/> ![alt text][image6]
<br/> ![alt text][image7]
<br/> ![alt text][image8]
<br/>**This a sample of a  final combined binary image**
<br/> ![alt text][image9]

#### 3. Perspective transform approach

##### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To get the perspective transform the vertices for the region_of_interest were defined and then warped images were obtained by using function cv2.warpPerspective to the images with hough lines after ROI was applied.

##### 3.1 ROI ( Region Of Interest)
3.1.1. Function generate_vertices(): (Cell 40 in Jupyter notebook). Was defined to set up the trapezoidal vertices of the region of interest containing road lane lines. This function was taken from project 1 term1.

3.1.2. Function region_of_interest(). (Cell 41 in Jupyter notebook). Was used to apply an image mask to the input image in such a way that only keeps region of the image defined by the polygon from from 'vertices'.
An ilustration is shown below:
<br/> ![alt text][image10]

##### 3.2 Perspective transform.

3.2.1. One image of ROI was selected. (Cell 42 in Jupyter notebook)

3.2.2. Hough transform is applied to binary image Region of Interest with function cv2.HoughLinesP(). (Cell 49 in Jupyter notebook)

3.2.3. to define the source and destination points the output heigh of points was defined as *0.4 * imghigh*  (cell 52 Jupyter notebook) for top_rigth and top_left vertices as destination points. 

This is a shot part of code to ilustrate it:

```
# get oringinal and destination points based on previously calculated hough lines
    # top righ, bottom rigth, top left, bottom left
    trs     = [line1[0], line1[1]]
    brs     = [line1[2], line1[3]]
    tls     = [line2[0], line2[1]]
    bls     = [line2[2], line2[3]]
    
    trd     = [line1[2], ylimit] 
    brd     = [line1[2], line1[3]]
    tld     = [line2[2], ylimit]
    bld     = [line2[2], line2[3]]
    
    
    source_img_pts = [trs, brs, tls, bls]
    destination_pts= [trd, brd, tld, bld]
```
3.2.4. Then direct and inverse perspective transform was computed using cv2.getPerspectiveTransform() .

3.2.5. A short pipeline to get direct nad inverse trasformations was defined. Function pipeline_get_tf_matrix(). (Cell 54 Jupyter notebook).

To ilustrate this process a sample image is shown below:
<br/> ![alt text][image11]
<br/> ![alt text][image12]
<br/> ![alt text][image13]



#### 4. Lane lines polinomial fit.
##### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

4.1. Using function scipy.find_peaks_cwt() to smooth a vector by convolving it with wavelet for each width, getting relative maximums as stimations for posible lane lines detections. (cell 57 from Jupyter notebook).

4.1.1.  Function getlanelinesbase(): (cell 57 Jupyter notebook). histogram vector is computed and then convolved with wavelet(width). 
4.1.2. The left line is assumed to be the first element returned by scipy.find_peaks_cwt()  and the rigth line is assumed to be the last element returned by the same function scipy.find_peaks_cwt().



Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



[//]: # (Image References)

[image1]: ./output_images/a_chess_corners/a14.png "chess corners img"

[image2]: ./output_images/b_undistorted_chess_images/b14.png " original and undistorted chess "

[image3]: ./output_images/c_undistorted_car_test_images/c6.png " original and undistorted img "

[image4]: ./output_images/cx_car_gray_images/cx6.png " gray binary img "

[image5]: ./output_images/d_car_s_channel_binary_images/d6.png "  schannel binary img"

[image6]: ./output_images/e_car_L_channel_binary_images/e6.png "  lchannel binary img"

[image7]: ./output_images/f_car_sobelx_binary_images/f6.png " x sobel img "

[image8]: ./output_images/g_car_sobely_binary_images/g6.png " y sobel img  "

[image9]: ./output_images/h_car_combined_binary_images/h6.png " combined thresholds  binary img "

[image10]: ./output_images/i_car_roi_combined_binary_images/i6.png " roi in binary img "

[image11]: ./output_images/j_car_lanelines_on_road/j2.png " lines on img "

[image12]: ./output_images/i_car_roi_combined_binary_images/i2.png " roi in binary img "

[image13]: ./output_images/k_car_bird_eye/k2.png " bird eye view img "

[image14]: ./output_images/l_car_full_pipeline/l2.png " full pipeline on img"





[image14]: ./output_images/i_car_roi_combined_binary_images/ "  "

[image15]: ./output_images/i_car_roi_combined_binary_images/ "  "







[video1]: ./project_video.mp4 "Video"
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MTI1NzE4NjUsMTU1MTM0ODI3OSwtMT
Y5NDE5MTg5OCwtMTg3MDU5MzY3OCwtMTYzNzYzMzMzMSw1ODUy
MDAwNDgsLTE2MTc2ODQ0OTEsLTExMjA5NzMwNjQsNDI3MDEwMj
c2LDEyMTMwNDg1NTUsMTA3MzE0MzkzNSwxMDE2MzE1NDMyLC03
NDMwODY3NCwxOTc3NDQ1MDcsOTc1NjgyMTAxLDExNjg4NTYyOD
QsLTEwMzE1NzM1NzUsLTE1NDk3MTk4NywxNTI1NTM5NzYyLDIx
MzA2MTUzMl19
-->