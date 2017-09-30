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

I approached the problem by first just normalizing the data to a range of 0 to 1 and then centering it to 0. I would then try other preprocessing methods if the 93% accuracy could not be achieved; however, my pipeline was sufficient and no other preprocessing steps were necessary. 

I used normalization and 0 centering for a couple reasons. Since the output layer is a softmax layer, it has an output of [0,1] so normalizing to [0,1] allows better mapping. Another reason is that having it centered at 0 allows gradient descent to converge quicker; and, since my optimizer is 'adam', which is a variation of gradient descent, center to 0 allows quicker convergence.

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

I decided to add dropout since I was using three fully connected layers at the end and dropouts are useful between fully connected layers. A dropout of 50% increased the accuracy to about 92%. At this point, I noticed I had set the loss to mse, which is more for regression type problems and not classification. I also noticed I did not have a softmax actiavtion from my last layer which would also be useful for this type of problem. So changing the loss to categorical_crossentropy and adding a softmax activation for the last layer increased the validation accuracy to about 93%. Finally, I changed the epochs to 15 to stabilize the loss; it stayed around 94%.

Keras model.fit() function comes with the necessary code to produce the accuracy metrics. None of this was impelemented by me. Since the 93% accuracy minimum was achieved pretty easily with this architecture, it supports the notion that LeNet's architecture was a good choice.

My final model results were:
* training set accuracy of 98.2%
* validation set accuracy of 93.6%
* test set accuracy of 92.1%

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
| Speed limit (70km/h)  | Speed limit (70km/h)					        | 
| Yield     			| Yield 										|
| Priority Road			| Priority Road									|
| Keep right	        | Keep Right					 				|
| Ahead Only			| Ahead Only          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares to the accuracy in the test and validation sets since I got about 94%. As I said, this only worked after cropping the images. Running my netowrk on the raw images just resized produced only 20% accuracy with the top 5 results for each not being close. My guess is that without cropping, the network did not have enough information to go on.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Looking at the softmax values, my network was very positive about the predictions since they were 1.0 or .999. All other predictions were very small comparatively. It seems that some of the predictions are greater than 1.0 since the top prediction is 1 and the rest would add up to greathan than 1. But, this is due to casting to float 32 and the way numbers are stored in the model. The others have an effective probability of 0 when the numbers are really small (10E-20 for example).

Example 1
* Sign Value: [4, 0, 1, 39, 33]
* Sign Name: ['Speed limit (70km/h)', 'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Keep left', 'Turn right ahead']
* Probs: [1.0, 3.0035403e-09, 3.1933085e-13, 1.8953176e-20, 6.5966291e-22] 
* Correct Value:  4 
* Correct Sign: Speed limit (70km/h) 

Example 2
* Sign Value: [13, 15, 9, 35, 38]
* Sign Name: ['Yield', 'No vehicles', 'No passing', 'Ahead only', 'Keep right']
* Probs: [1.0, 1.0978565e-25, 1.5936695e-27, 4.021732e-29, 5.1977345e-32] 
* Correct Value:  13 
* Correct Sign: Yield 

Example 3
* Sign Value: [14, 17, 29, 6, 11]
* Sign Name: ['Stop', 'No entry', 'Bicycles crossing', 'End of speed limit (80km/h)', 'Right-of-way at the next intersection']
* Probs: [0.9999429, 5.7102854e-05, 3.0667435e-09, 1.0783408e-09, 6.4386863e-10] 
* Correct Value:  14 
* Correct Sign: Stop 

Example 4
* Sign Value: [35, 36, 34, 3, 6]
* Sign Name: ['Ahead only', 'Go straight or right', 'Turn left ahead', 'Speed limit (60km/h)', 'End of speed limit (80km/h)']
* Probs: [1.0, 7.6385103e-16, 2.1744614e-19, 5.6551123e-21, 6.847924e-24] 
* Correct Value:  35 
* Correct Sign: Ahead only 

Example 5
* Sign Value: [33, 39, 24, 37, 40]
* Sign Name: ['Turn right ahead', 'Keep left', 'Road narrows on the right', 'Go straight or left', 'Roundabout mandatory']
* Probs: [0.99999487, 5.1807724e-06, 1.6847434e-14, 4.4204892e-15, 2.9713672e-15] 
* Correct Value:  33 
* Correct Sign: Turn right ahead 