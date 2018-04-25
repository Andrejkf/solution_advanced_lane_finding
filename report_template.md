## Project 4 term1. Advanced lane Finding

### Note: I used the template provided for my report.

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

The output videos are located in the folder named **output_videos**.





### 1.  Camera Calibration
To get the camera matrix and distortion coefficients in file *camera_calibration_final.py (also in notebook cell 6)* the  class *CalibratorCamera* was defined. It has the following funtions:
 
* 1.1 find_calibration_numbers(): Used to find chess board *(9x6)* corners in grayscaled images and then to get camera matrix and distortion coefficients.
 
 * 1.2. save_coefficients(): to save camera matrix and distortion coefficients in a pickle with  *fileName='wide_dist_pickle.p'*.
 
* 1.3. load_coefficients(): Used to load the previously saved information(camera matrix and distortion coefficients) to make process faster for the pipeline on the video instead of calculating them again.

* 1.4. get_coefficients(): Used to either load distortion coefficients from the pickle file ([wide_dist_pickle.p](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/blob/master/wide_dist_pickle.p "wide_dist_pickle.p")) or to calculate them.
* 1.5. get_undistorted(): To apply distortion correction to images.

##### Steps
1. Get inner corners.

	1.1 draw detected corners for debbugging purposes.

2. Find the chess inner corners using *cv2.findChessboardCorners()*.
3. Get a list of object points and image points.
4. Get camera matrix and distortion coeffients using *cv2.calibrateCamera()*.
5. Apply camera matrix and distortion correction to chess images.

These steps are ilustrated below:
<br/> ![alt text][image1]
<br/> ![alt text][image2]








### Pipeline (single images)

#### 1. distortion-corrected image.

##### Provide an example of a distortion-corrected image.

After applying distortion correction to the chess board images I saved the camera matrix and distortion coefficients in a pickle file ([wide_dist_pickle.p](https://github.com/Andrejkf/solution_advanced_lane_finding_p4t1/blob/master/wide_dist_pickle.p "wide_dist_pickle.p")). Then I loaded the same matrix and distortion coefficients and applied them to the raw test images provided in folder *./test_images*. Below is shown an illustration:
<br/> ![alt text][image3]






#### 2. Thresholded binary image

##### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Initially I tried to use a color combination using a *yellow_white color mask* between HSV and HSL color spaces, but then I noticed that the *S channel* and *L channel* could be used separately to generate binary images. Also, I tested sobel gradients in *x direction*, *y direction*, *magnitud threshold* and *angle direction* thresholds but for simplicity I used gradients in x direction.

Note: I believe that x_treshold, y_threshold, magnitude_threshol and direction_threshold can be used altogether to get less noisy images. Even if I did not combine them. Those functions are in cells 32,33,34 from the Jupyter notebook and I was inspired on class exercises to write them.

The values of threshold used were tunned by trial and error *looking to reduce all possible points/pixels that might introduce noise in the predictions*.

All that said, I defined the following functions:
* *hls_wyMask()*. (cell 27 Jupyter notebook). Used to apply a color white-yellow mask in HLS_color_space while keeping threshold values from project 1.
* *hsv_wyMask()*: (cell 28 Jupyter notebook). Used to apply a color white-yellow mask in HSV_color_space.
* *gray_binary()*: (cell 29 Jupyter notebook). Used to get a binary thresholded image from images im grayscale. I tried using global  histogram equalization but I did not included in this pipeline.
* *hls_channel_binary()*: (cell 30 Jupyter notebook). Used to apply binary threshold to S or L channel of image in HSL color space.
* *abs_sobel_thresh()*:  (cell 32 Jupyter notebook). Used to apply sobel gradients in X or Y direction using cv2.Sobel to gray images. 
* *mag_thresh()*:  (cell 33 Jupyter notebook). Used to apply Sobel L2 norm using cv2.Sobel to gray images. 
* *dir_threshold()*:  (cell 34 Jupyter notebook). Used to apply Sobel directional threshold using cv2.Sobel to gray images.
* *binary_threshold()*:  (cell 37 Jupyter notebook). Used to generate the combined binary thresholded image.

Some images to illustrate the process are shown below:

<br/> ![alt text][image4]
<br/> ![alt text][image5]
<br/> ![alt text][image6]
<br/> ![alt text][image7]
<br/> ![alt text][image8]
<br/>**This a sample of a  final combined binary image**
<br/> ![alt text][image9]

#### 3. Perspective transform approach

##### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To get the perspective transform the *vertices* for the *region_of_interest* were defined and then *warped images* were obtained by applying the function *cv2.warpPerspective()* to the *images with hough lines detected after ROI was done*.

##### 3.1 ROI ( Region Of Interest)
To apply ROI to images a poligon vertices was generated for image masking as follows.

3.1.1. *Function generate_vertices()*: (Cell 40 in Jupyter notebook). Was defined to set up the trapezoidal vertices of the region of interest containing road lane lines. This function was taken from project 1 term1.

3.1.2. *Function region_of_interest()*. (Cell 41 in Jupyter notebook). Was used to apply an image mask to the input image in such a way that only kept the region of the image defined by the polygon from *'vertices'.*

An ilustration is shown below:
<br/> ![alt text][image10]

##### 3.2 Perspective transform.

3.2.1. First, image of ROI was selected. (Cell 42 in Jupyter notebook).

3.2.2. Then, Hough transform was applied to binary image Region of Interest with function *cv2.HoughLinesP().* (Cell 49 in Jupyter notebook).

3.2.3. In addition, to define the source and destination points the output heigh of points was defined as ***0.4 * imghigh***  (cell 52 Jupyter notebook) for *top_rigth* and *top_left* *vertices* as destination points. 

Bellow, you may find a short part of code to ilustrate it:

Here:
*t = top. b = bottom. r = rigth. l = left. s = source*, and *d = destination*. So, for source points we have ( *trs, brs, tls, and bls*) and for destination points (*trd, brd, tld, and bld*).
```
# get original and destination points based on previously calculated hough lines
    # top righ, bottom rigth, top left, bottom left
    trs     = [line1[0], line1[1]] # source
    brs     = [line1[2], line1[3]] # source
    tls     = [line2[0], line2[1]] # source
    bls     = [line2[2], line2[3]] # source
    
    trd     = [line1[2], ylimit]   # destination
    brd     = [line1[2], line1[3]] # destination
    tld     = [line2[2], ylimit]   # destination
    bld     = [line2[2], line2[3]] # destination
    
    
    source_img_pts = [trs, brs, tls, bls]
    destination_pts= [trd, brd, tld, bld]
```
3.2.4. Then direct and inverse perspective transform was computed using *cv2.getPerspectiveTransform()* .

3.2.5. A short pipeline to get direct (M) and inverse (Minv) transformations  was defined. Function *pipeline_get_tf_matrix()*. (Cell 54 Jupyter notebook).

Illustration of this process on sample image is shown below:
<br/> ![alt text][image11]
<br/> ![alt text][image12]
<br/> ![alt text][image13]



#### 4. Lane lines polinomial fit.
##### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

4.1. Using function *scipy.find_peaks_cwt()* to smooth a vector by convolving it with wavelet for each width, getting relative maximums as stimations for posible lane lines detections. (cell 57 from Jupyter notebook).

4.1.1.  Function *getlanelinesbase()*: (cell 57 Jupyter notebook). The histogram vector is computed and then convolved with wavelet(width). 

4.1.2. The left line is assumed to be the first element returned by *scipy.find_peaks_cwt()*  and the rigth line is assumed to be the last element returned by the same function *scipy.find_peaks_cwt()*.

This is illustrated below:
<br/> ![alt text][image14]
<br/>(In red: histogram used for convolution with wavelet(width))

4.1.3. Then pixels from lane lines were searched using a window of 120 pixels. Function *getlanepixels()* (cell 57 Jupyter notebook).

4.1.4. After, a *2nd* order polinomial function is aproximated to the curved lines described by lane lines using *numpy.polyfit()* function. Function *getcurvedlaneline()*. (cell 61 Jupyter notebook).


#### 5.1 Radius of curvature of the lane
##### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

5.1.1. Function *drawcurvedline()* was defined. (cell 62 Jupyter notebook). Used to estimate curvature radius for lanes. 

A conversion *from pixel units  to meters* is done. Then a seccond order polinomial regression estimation is done using *numpy.polyfit()*.

To ilustrate process, a short piece of code is shown below:
```
ymbypixel = 30/imghigh # aprox meters by pixel
    xmbypixel = 3.7/700 # aprox meters by pixel
    
    yeval = np.max(yp)

    polifit = np.polyfit(yp*ymbypixel, xp*xmbypixel, 2)
    
    estimation= int(((1 + (2*polifit[0]*yeval + polifit[1])**2)**1.5)/np.absolute(2*polifit[0]))
```
#### 5.2 Position of the vehicle with respect to center.
In here, an assumption that the images correspond to a camera located on the center of the car was made. So *pixel units* are *transformed to meters* and then the difference to the center of the image was computed.

5.2.1. A Function named *estimatedistancefromlane()* was defined. (Cell 64 Jupyter notebook). This was the funtion used to estimate vehicle distance from center.

To ilustrate it a piece of the code is shown below:
```
imghhalf = imgh/2    
    
    xmbypixel = 3.7/700 # aprox meters by pixel in x
    
    imgcenter = (imghhalf, imgw)
    
    carmiddlepixel = int((leftbase[0] + rightbase[0])/2)
    
    estimation = float("{0:.2f}".format((carmiddlepixel - imgcenter[0]) * xmbypixel))
```


#### 6. Lane area identification.

##### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In cell 28 from jupyter notebook was defined a *full pipeline* where a distorted images in RGB color space is fed and the predicted lane area is overlapped to the image in order to check and visualize the road lane prediction.

An ilustration is shown below:
<br/> ![alt text][image15]

---

### 7. Pipeline (video)

##### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Output videos are located in this repository on the folder *./output_videos*.

In addition,  you may want to watch them in your browser using the following links:
* [Project video solution](https://youtu.be/lyNLjPRfZ_8).
* [Testing initial solution on challenge video](https://youtu.be/RNq73SDdllc) (This is a video for testing purposes).
* [Testing initial solution on a harder challenge video](https://youtu.be/FhY_4tScF7M) (This is a seccond video for testing video).

---

### Discussion

##### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have noticed that sometimes the model returns slope estimations dividing by zero, so this is something related to the slope calculation and might be improved. That is why you get the perception in some miliseconds that the lane lines disapear *(blinking lane area)*. 

But, for the project video in general, lane lines are detected all the way along. 

Even if this solution is not ready for production on a real car, is a first aproximation about how to use computer vission techniques to the inference system in the Self driving Car.

One possible scenario where the solution is failing in, is summarized in this statement: *"the slower the radius of curvature, the more probability to have lane lines out of the perspective transformation"*, resulting in missing lane lines. This can be observed on the third video *(harder challenge video)*.

I also have noticed that for some images *noisy binary images* during the image preprocessing so further image preprocessing would end up with a more robust model to changes in brightness, temperature and fuzziness on lane l                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ines. This can be observed on the seccond video *(challenge video)* where the lane lines estimation is lost while the car is being driven under the bridge.

#### Possible improvements

* A nice way to go further on this project I would install a camera on a car and I would test this model again.
* To make this solution approach more robust *outliers filtering (in lane lines detection)* should be added. For example, a low pass filter. 
* It would be very good to try other aproaches for *low curvature radius* road lane lines values  like *splines* or *higher  order wavelets generation*.


***Note for the reader:  Thank you very much for taking part of your time reading this document, I appreciate it.***

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

[image14]: ./output_images/m_car_histograms_bird_eye/m2.png " histogram for wavelets"

[image15]: ./output_images/l_car_full_pipeline/l2.png " full pipeline img "





<!--stackedit_data:
eyJoaXN0b3J5IjpbNjI5ODQwOTM0LDMyMTcwMzM1MSwtMTQxOD
E5NDY0MywtMTIzNzY5OTYwMywtMTg5NTUzMTA2MywxNTE5NzA5
MTMyLC0yMDA5NTg1MTk4LDEwMzU1MTg3MzMsLTE4Mjg5NzgwNz
YsNTY5OTUzMDMsMjgyMzcyNzcxLDQzODU3MTUxOCwtMTA3Mjc3
NjkyNSwtMTk2Mzg3MDgyOCw3MDQzNjYzMjAsLTMzMTIxMTgwMC
wyODc2MTY3MjQsLTE5MzI2MzI0ODUsMTkyMTk4MTIxNywxMTYw
NjA5Njg1XX0=
-->