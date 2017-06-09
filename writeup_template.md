
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
* Decribed in: [VehicleDetectionByHOGSVM.ipynb](./VehicleDetectionByHOGSVM.ipynb)
* Decribed in: [VehicleDetectionByHOGSVM.html](./VehicleDetectionByHOGSVM.html)

### Second, Yolo v1 algorithm is:
* Use tiny-YOLO v1, since it's easy to implement.
* Use Keras to construct the YOLO model.
* Decribed in: [VehicleDetectionByYOLO.ipynb](./VehicleDetectionByYOLO.ipynb)
* Decribed in: [VehicleDetectionByYOLO.html](./VehicleDetectionByYOLO.html)

### The result videos of my whole implemented algorigm pipline is:
* [./output_video/project_video_output_HOGSVM.mp4](./output_video/project_video_output_HOGSVM.mp4)
* [./output_video/project_video_output_Yolo.mp4](./output_video/project_video_output_Yolo.mp4)

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

### My whole implemented Hog & SVM algorigm pipline is:
![alt text][image4]
* Here's a [link to my video (project_video) HOG & SVM pipline](./output_video/hog_svm_pipline.wmv)
* Here's a [link to my video (project_video) HOG & SVM result](./output_video/project_video_output_HOGSVM.mp4)

---

## 2. Yolo v1 based Vehicle Detection

### The tiny YOLO v1
* The tiny YOLO v1 is consist of 9 convolution layers and 3 full connected layers. 
* Each convolution layer consists of convolution, leaky relu and max pooling operations. 
* The first 9 convolution layers can be understood as the feature extractor, whereas the last three full connected layers can be understood as the "regression head" that predicts the bounding boxes.
* There are a total of 45,089,374 parameters in the model.
* The detail of the architecture is in list in this table:
![alt text][image6]
* The output of this network is a 1470 vector, which contains the information for the predicted bounding boxes. 
* The 1470 vector output is divided into three parts, giving the probability, confidence and box coordinates. 
* Each of these three parts is also further divided into 49 small regions, corresponding to the predictions at each cell. 
* In postprocessing steps, I take this 1470 vector output from the network to generate the boxes that with a probability higher than a certain threshold. 
* Training the YOLO network is time consuming. 
* I will download the pretrained weights from [here](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection) and load them into our Keras model. 
* The following shows the results for several test images with a threshold of 0.17. 
* I can see that the vehicles are detected:
![alt text][image7]


### My whole implemented Yolo v1 algorigm pipline is:
* Here's a [link to my video (project_video) Yolo result](./output_video/project_video_output_Yolo.mp4)

---

###Discussion

####1. Speed Up Problem
* I started out with a linear SVM due to its fast evaluation. 
* Nonlinear kernels such as rbf take not only longer to train, but also much longer to evaluate. 
* Using the linear SVM, I obtained execution speeds slower than Yolo. 

####2. Convert another programing languages
* I implemented by python
* I will try implement C/C++ with OpenCV.

