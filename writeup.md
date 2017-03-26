# ** Build a Traffic Sign Recognition Project Writeup**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Support_Files_For_Writeup/ViewRawTrainingData.png "View Raw Training Data"
[image2]: ./Support_Files_For_Writeup/PlotNumberOfSamplesPerClass.png "Number Of Samples Per Class"
[image3]: ./Support_Files_For_Writeup/PlotNumberOfSamplesPerClassAfterTransformation.png "Number Of Samples Per Class After Transformation"
[image4]: ./Support_Files_For_Writeup/VisualizeAugmentation.png "Visualize Augmentation"
[image5]: ./German_Traffic_Sign_Images/bumpyRoad.jpg "Bumpy Road"
[image6]: ./German_Traffic_Sign_Images/pedestrians.jpg "Pedestrians"
[image7]: ./German_Traffic_Sign_Images/roadWorks.jpg "Road Works"
[image8]: ./German_Traffic_Sign_Images/speedLimit70.jpg "Speed Limit 70"
[image9]: ./German_Traffic_Sign_Images/stop.jpg "Stop"
[image10]: ./Support_Files_For_Writeup/RoadWorksWebConvLayer1.png "Road Works Web Convolutional Layer 1"
[image11]: ./Support_Files_For_Writeup/Softmax_BumpyRoad.png "Softmax Probability for Bumpy Road"
[image12]: ./Support_Files_For_Writeup/Softmax_Pedestrians.png "Softmax Probability for Pedestrians"
[image13]: ./Support_Files_For_Writeup/Softmax_RoadWork.png "Softmax Probability for Road Works"
[image14]: ./Support_Files_For_Writeup/SoftMax_70kmphr.png "Softmax Probability for 70 km/h Limit"
[image15]: ./Support_Files_For_Writeup/Softmax_Stop.png "Softmax Probability for Stop"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaxanc/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the first code cell of the IPython notebook under the heading **Step 1: Dataset Summary & Exploration**  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the first code cell of the IPython notebook under the heading **Include an exploratory visualization of the dataset**

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing the number of samples per class

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for the first step is contained in the first code cell of the IPython notebook under the heading **Image Transform**

As a first step, I decided to generate "fake" data for classes with low number of counts because some classes have only 200 counts where others have close to 2000. I added extra samples to the training set by applying random rotation, x and y translation to the original image.

Here is a bar chart showing the number of samples per class after transformation

![alt text][image3]

The second, I normalized the image data similar to tensorflow's per image normalization to remove differences of contrast and brightness between images.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for normalizing the training, validation and test data is contained in the second code cell of the IPython notebook under the heading **Pre-process the Data Set (normalization, grayscale, etc.)**

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

The first code cell of the IPython notebook under the heading **Image Transform** contains the code for augmenting the data set. I decided to generate additional data because some classes have very low number of samples. To add more data to the data set, I used the following techniques because resolution is quite low, zooming/scaling and shear might blur the samples too much.

Here is an example of 5 original image and an augmented image:

![alt text][image4]

The difference between the original data set and the augmented data set is the following:
* Random image rotation in the range of +/- 10 degrees
* Random image x and y translation in the range of +/- 2 pixels


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the first cell of the IPython notebook under the heading **Model Architecture**.

My final model consisted of the following layers:

| Layer | Description |
|:---------------------:|:---------------------------------------------:|
| Input | 32x32x3 RGB image |
| Convolutional Layer 1 (5x5) | 1x1 stride, Valid padding, Output = 28x28x6 |
| Activation 1 | RELU |
| Max Pooling 1 | 2x2 stride, Output = 14x14x6 |
| Convolutional Layer 2 (5x5) | 1x1 stride, Valid padding, Output = 10x10x16 |
| Activation 2 | RELU |
| Max Pooling 2 | 2x2 stride, Output = 5x5x16 |
| Flatten	| Output = 400 |
| Fully connected Layer 1 | Output = 120 |
| Activation 3 | RELU |
| Dropout 1 | Keep Probability = 0.5 |
| Fully connected Layer 2 | Output = 84 |
| Activation 4 | RELU |
| Dropout 2 | Keep Probability = 0.5 |
| Fully connected Layer 3 | Output = 43 |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cells of the IPython notebook under the heading **Train, Validate and Test the Model**

To train the model, I used an Adam optimizer. While searching and experimenting for different optimizers, I found this blog to be quite useful: [Optimizing Gradient Decent](http://sebastianruder.com/optimizing-gradient-descent/)

The batch size is 128 and epochs is 50. After playing around with different batch sizes and epochs number, I was not able to come up with an optimal approach with the time I allowed myself. I didn't find increasing batch size to significantly increase memory usage.

I left the learning rate to be as low as possible (learning rate = 0.0009) and run for longer time to hopefully achieve the most accurate result.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the first cell of the IPython notebook under the heading **Evaluate the Model**

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 96.9
* test set accuracy of 95.1%

**If an iterative approach was chosen:**
**What was the first architecture that was tried and why was it chosen?**

The LeNet solution from the Convolutional Neural Networks lesson was chosen as the first architecture. After adjusting the input sizes and colour depth, it was able to achieve about 85% validation accuracy.

**What were some problems with the initial architecture?**

It was over-fitting the data. The training data was able to achieve reasonable accuracy but validation data flatten at about 85%.

**How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc.), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

To overcome over-fitting, I have added dropouts to the model at various layer and experimented different keep probabilities. The best result seem to be placing two dropouts sandwiched between the three fully connected layer and keep probability lower than 0.5. The results were still over fitting, but accuracy was reasonable (validation accuracy around 94%)

Later on I changed strategy and looked into training data itself. After experimentation, the following was implemented:
* Per image normalization the same as the implementation in tensorflow. However I had issues getting the function working properly, so I wrote my own function.
* Some classes have very low counts, around 200 compared to 2000 in other classes. So I added random rotation and translation to generate fake data for classes with low counts.

**Which parameters were tuned? How were they adjusted and why?**

The learning rate was initially increased to help speed up the training but was creating worse results. Thus it is lowered to as small as possible to create accurate data.

The keep rate was adjusted to help reduce over-fitting, but below 0.5, it did not seem to improve results any further.

To generate fake data, I have tried rotation and translation. There are other techniques I could also try, such as brightness adjustment, shear, scaling etc.. After trying to tune the amount of rotation and translation I have decided to keep this simple.

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**

Adding the dropout layers was a good decision. It helped to reduce over-fitting of the data and also create fake data on classes with low number of samples.

**If a well known architecture was chosen:**
**What architecture was chosen?**

The LeNet model with dropout layers were chosen.

**Why did you believe it would be relevant to the traffic sign application?**

It showed promising results predicting 32x32 letters and numbers. It has enough layers to recognise objects in similar size street signs.

**How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**

It can predict test data set to 95% accuracy. Higher than the requirement of 93% for the validation set. However, the model is still over-fitting. For future work, I would remove more layers from the model to help remove over-fitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the first cell of the IPython notebook under the heading **Predict the Sign Type for Each Image**

Here are the results of the prediction:

| Image | Prediction |
|:-------------:|:-------------:|
| Bumpy Road | Bumpy Road |
| Pedestrians | Pedestrians |
| Road Works | Bicycle Crossing |
| Speed limit (70km/h) | Speed limit (70km/h) |
| Stop Sign | Stop Sign |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. It did not correctly predict the road works sign and confused it with a bicycle.

Further analysis was done looking into the feature maps below while processing the Road Works street sign.

![alt text][image10]

It is clear to see the bright spot between the legs of the man and the pile of dirt has been recognised as features. Even from looking into let's say Feature Map 1 or Feature Map 2, I would think it is a bicycle. It is interesting, and a bit strange that a white spot and a black spot has been recognised as similar features.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the first cell of the Ipython notebook under the heading **Visualizing Softmax probabilities**

For the first image, the model is sure that this is a Bumpy Road sign (probability of 1.0). The correct result is Bumpy Road sign. The top five soft max probabilities were:

![alt text][image11]

For the second image is sure that this is a Pedestrians sign (probability of 0.99999). The correct result is Pedestrians sign. The top five soft max probabilities were:

![alt text][image12]

For the third image is relatively sure that this is a Road Works sign (probability of 0.972), however the correct result is Road Work. The top five soft max probabilities were:

![alt text][image13]

For the forth image is sure that this is a Speed Limit (70 km/h) sign (probability of 0.992). The correct result is a Speed Limit (70 km/h) sign. The second prediction is the Speed Limit (20 km/h) sign which seems sensible. The top five soft max probabilities were:

![alt text][image14]

For the fifth image is sure that this is a Stop sign (probability of 1.0). The correct result is a Stop sign. The top five soft max probabilities were:

![alt text][image15]
