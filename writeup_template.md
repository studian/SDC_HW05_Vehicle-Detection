
[//]: # (Image References)
[image1]: ./output_images/HOG_train_datasets.JPG
[image2]: ./output_images/HOG_train_datasets_detail.JPG
[image3]: ./output_images/HOG_features.JPG
[image4]: ./output_images/HOG_vehicle_detect_pipline1.JPG
[image5]: ./output_images/HOG_vehicle_detect_pipline2.JPG
[image6]: ./output_images/YoloModel.JPG
[image7]: ./output_images/YoloDetection.JPG

[video1]: ./output_video/hog_svm_pipline.wmv
[video2]: ./output_video/project_video_output_HOGSVM.mp4
[video3]: ./output_video/project_video_output_Yolo.mp4


# **Vehicle Detection Project**

## I implement vehicle detection using HOG & SVM and Yolo v1.

### First, HOG & SVM algoritm is:
* Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
* Color transform and binned color features, as well as histograms of color, to combine the HOG feature vector with other classical computer vision approaches.
* Sliding-window technique to search for cars with the trained SVM.
* Creating a heatmap of recurring detections in subsequent framens of a video stream to reject outliers and follow detected vehicles.
* Decribed in: VehicleDetectionByHOGSVM.ipynb
* Decribed in: VehicleDetectionByHOGSVM.html

### Second, Yolo v1 algorithm is:
* Use tiny-YOLO v1, since it's easy to implement.
* Use Keras to construct the YOLO model.
* Decribed in: VehicleDetectionByYOLO.ipynb
* Decribed in: VehicleDetectionByYOLO.html

### The result videos of my whole implemented algorigm pipline is:
* ./output_video/project_video_output_HOGSVM.mp4
* ./output_video/project_video_output_Yolo.mp4

---

## 1. HOG & SVM based Vehicle Detection

### Data Exploration
* Labeled images were taken from the GTI vehicle image database GTI.
* All images are 64x64 pixels. 
* Images of the GTI data set are taken from video sequences which needed to be addressed in the separation into training and test set. 
* Due to the temporal correlation in the video sequences, the training set was divided as follows: the first 70% of any folder containing images was assigned to be the training set, the next 20% the validation set and the last 10% the test set. 
![alt text][image1]
* Shown below is an example of each class (vehicle, non-vehicle) of the data set.
* In the process of generating HOG features all training, validation and test images were normalized together and subsequently split again into training, test and validation set. 
* Each set was shuffled individually. 
![alt text][image2]

### Extraction of HOG, color and spatial features
* I experimented with a number of different combinations of color spaces and HOG parameters and trained a linear SVM using different combinations of HOG features extracted from the color channels. 
* I trained a linear SVM using all channels of images converted to HLS space. 
* I included spatial features color features as well as all three HLS channels, because using less than all three channels reduced the accuracy considerably. 
* The final feature vector has a length of 6156 elements, most of which are HOG features. 
![alt text][image3]

### Sliding window search
* I segmented the image into 4 partially overlapping zones with different sliding window sizes to account for different distances. 
* The window sizes are 240,180,120 and 70 pixels for each zone. 
* Within each zone adjacent windows have an ovelap of 75%, as illustrated below. 
* The search over all zones is implemented in the `search_all_scales(image)` function. 
* Using even slightly less than 75% overlap resulted in an unacceptably large number of false negatives.
* The false positives were filtered out by using a heatmap approach as described below. 
* Here are some typical examples of detections:
![alt text][image5]

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

