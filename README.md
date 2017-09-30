# Traffic Sign Recognition



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: training_set_count_image.PNG "Data Visualization 1"
[image2]: validation_set_count_image.PNG "Data Visualization 2"
[image3]: ./prediction_examples/example_1_70.jpg "Traffic Sign 1 Cropped"
[image4]: ./prediction_examples_raw/example_1_70.jpg "Traffic Sign 1 Raw"
[image5]: ./prediction_examples/example_2_yield.jpg "Traffic Sign 2 Cropped"
[image6]: ./prediction_examples_raw/example_2_yield.jpg "Traffic Sign 2 Raw"
[image7]: ./prediction_examples/example_3_stop.jpg "Traffic Sign 3 Cropped"
[image8]: ./prediction_examples_raw/example_3_stop.jpg "Traffic Sign 3 Raw"
[image9]: ./prediction_examples/example_4_ahead.jpg "Traffic Sign 4 Cropped"
[image10]: ./prediction_examples_raw/example_4_ahead.jpg "Traffic Sign 4 Raw"
[image11]: ./prediction_examples/example_5_right.jpg "Traffic Sign 5 Cropped"
[image12]: ./prediction_examples_raw/example_5_right.jpg "Traffic Sign 5 Raw"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used built-in python functions ( len() and set() ) and numpy.shape to calculate the summary statistics of the traffic signs data set:

* The size of training set is 34799 (32,32,3) images.
* The size of the validation set is 4410 (32,32,3) images.
* The size of test set is 12630 (32,32,3) images.
* The shape of a traffic sign image is (32,32,3) or a 32x32 image in RGB.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the frequency of the 43 different classes fro the training set, followed by the validation set.

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I approached the problem by first just normalizing the data to a range of -1 to 1. I would then try other preprocessing methods if the 93% accuracy could not be achieved; however, my pipeling was sufficient and no other preprocessing steps were necessary.

Data normalization was pretty straightforward: Cast as a float and then divide by that max value (255) and subtract 0.5.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I began by trying to implement LeNet's convolutional neural network in Keras and then go from there.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image							    | 
| Convolution 5x5     	| 1x1 stride, depth of 6                        |
| RELU					|												|
| Max pooling	      	| 1x1 stride                       				|
| Convolution 5x5     	| 1x1 stride, depth of 16                       |
| RELU					|												|
| Max pooling	      	| 1x1 stride                       				|
| Flatten       		|            									|
| Fully Connected		| 120 neurons        							|
| Dropout				| 50%											|
| Fully Connected		| 84 neurons									|
| Fully Connected		| 43 neurons									|
| Softmax       		| 								            	|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Since I used Keras for this project rather than Tensorflow, many of hyperparameters were preset; and I decided to use those first and then tweak as needed. I started with these parameters for my Keras network: optimizer = adam, batch_size = 128, epochs = 5, loss = mse). I would only need to tweak the epochs and loss. THe other hyperparameters were not changed since I was able to achieve over the necessary 93% using other means.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with a Keras implementation of LeNet's CNN, and was able to achieve about 89% accuracy with my starting parameters. I decided to use this architecture becuase it has been used for this type of classification problem before. The training loss and validation loss were very similar. So from here, I decided to use some of the tricks used in class since it seemed the model was not overfitting or underfitting just yet. It seemed the network architecture itself needed some tweaking to get higher accuracy. I decided to first change the epoch to 10 for more training, but that only increased the validation accuracy to 90%.

I decided to add dropout since I was using three fully connected layers at the end and dropouts are useful between fully connected layers. A dropout of 50% increased the accuracy to about 92%. At this point, I noticed I had set the loss to mse, which is more for regression type problems and not classification. I also noticed I did not have a softmax actiavtion from my last layer which would also be useful for this type of problem. So changing the loss to categorical_crossentropy and adding a softmax activation for the last layer increased the validation accuracy to about 94%. I finally decided to change the epochs to 15 just to see if I could get 95% since the loss was still steadily going down; meaning it was not overfitting or underfitting. I was finally able to achieve 95% this way.

Keras model.fit() function comes with the necessary code to produce the accuracy metrics. None of this was impelemented by me. Since the 93% accuracy minimum was achieved pretty easily with this architecture, it supports the notion that LeNet's architecture was a good choice.

My final model results were:
* training set accuracy of 98.4%
* validation set accuracy of 95.1%
* test set accuracy of 93.9%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web with cropped on left and raw on right:

![alt text][image3] ![alt text][image4] 

![alt text][image5] ![alt text][image6] 

![alt text][image7] ![alt text][image8] 

![alt text][image9] ![alt text][image10] 

![alt text][image11] ![alt text][image12] 

I first tried to classify all the images I found without cropping. My network was only able to get 20% accuracy. But once I cropped them, I was able to get 100% accuracy. I thought my network might have issues with the stop sign since there is a small watercolor in the image, but it didn't seem to mind. My guess is that the watercolor was small enough to not affect it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)   | Speed limit (70km/h)					        | 
| Yield     			| Yield 										|
| Priority Road			| Priority Road									|
| Keep right	        | Keep Right					 				|
| Ahead Only			| Ahead Only          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares to the accuracy in the test and validation sets since I got about 94%. As I said, this only worked after cropping the images. Running my netowrk on the raw images just resized produced only 20% accuracy with the top 5 results for each not being close. My guess is that without cropping, the network did not have enough information to go on.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Looking at the softmax values, my network was very positive about the predictions since they were 1.0. All other predictions were very small comparatively.

Example 1

* Sign Value: [4, 0, 1, 39, 37]
* Sign Name: ['Speed limit (70km/h)', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Keep left', 'Go straight or left']
* Probs: [1.0, 2.4243165e-11, 9.3124787e-12, 5.0900419e-13, 7.6936698e-22] 
* Correct Value:  4 
* Correct Sign: Speed limit (70km/h) 

Example 2

* Sign Value: [13, 35, 34, 9, 15]
* Sign Name: ['Yield', 'Ahead only', 'Turn left ahead', 'No passing', 'No vehicles']
* Probs: [1.0, 6.5337173e-30, 3.1067414e-32, 7.1209266e-33, 4.4466865e-33] 
* Correct Value:  13 
* Correct Sign: Yield 

Example 3

* Sign Value: [14, 17, 29, 0, 3]
* Sign Name: ['Stop', 'No entry', 'Bicycles crossing', 'Speed limit (20km/h)', 'Speed limit (60km/h)']
* Probs: [1.0, 3.0906172e-10, 9.2755408e-11, 1.2837635e-11, 8.7946733e-12] 
* Correct Value:  14 
* Correct Sign: Stop 

Example 4

* Sign Value: [35, 36, 34, 33, 38]
* Sign Name: ['Ahead only', 'Go straight or right', 'Turn left ahead', 'Turn right ahead', 'Keep right']
* Probs: [1.0, 3.0585937e-13, 2.752024e-15, 4.4884789e-16, 9.6793438e-20] 
* Correct Value:  35 
* Correct Sign: Ahead only 

Example 5

* Sign Value: [33, 39, 35, 11, 37]
* Sign Name: ['Turn right ahead', 'Keep left', 'Ahead only', 'Right-of-way at the next intersection', 'Go straight or left']
* Probs: [1.0, 2.4337135e-10, 1.5769221e-15, 8.5808726e-16, 3.3698941e-16] 
* Correct Value:  33 
* Correct Sign: Turn right ahead 