# Distracted-Driver-using-Deep-Neural-Networks
Real time driver distraction using state farm dataset
In this notebook, I'll use the dataset which includes images of drivers while performing a number of tasks including drinking, texting etc. The aim is to correctly identify if the driver is distracted from driving. We might also like to check what activity the person is performing.

The notebook will be broken into the following steps:

1. Import the Libraries.
2. Import the Datasets.
3. Create a vanilla CNN model.
4. Create a vanilla CNN model with data augmentation.
5. Train a CNN with Transfer Learning (VGG16).
6. Results.

Import the Libraries
I'll use Keras and Tensorflow libraries to create a Convolutional Neural Network. So, I'll import the necessary libraries to do the same.

DATASET

We use State Farm Dataset for training and testing the model. State farm, an insurance company released dataset which contains driver images for Kaggle competition in 2016 for image based driver poster classification. The dataset consist of 22,450 labelled images of 26 subjects which includes different colour, ethnicity, action, age, size etc. these subjects were used to perform classification of the subjects into 10 classes such as normal/safe driving,  talking in the phone by both left and right hands, controlling the radio, text messaging by left hand, test messaging by right hand,  talking to the co passenger, drinking while driving, hair and makeup, reaching behind, etc. each image is labelled with their action class.

“State farm distracted driver detection - data,”
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data,

Images overview
Let's take a look at the various images in the dataset. I'll plot an image for each of the 10 classes. As the directory names are not descriptive, I'll use a map to define the title for each image that is more descriptive.

                'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'
                
Create a vanilla CNN model
Building the model
I'll develop the model with a total of 4 Convolutional layers, then a Flatten layer and then 2 Dense layers. I'll use the optimizer as rmsprop, and loss as categorical_crossentropy.


Create a vanilla CNN model with data augmentation
Here I'm augmenting the previous model classifier, I'll use the data on which I want to train the model. The folder train includes the images I need. I'll generate more images using ImageDataGenerator and split the training data into 80% train and 20% validation split.


Train a CNN with Transfer Learning (VGG, MobileNet)
To reduce training time without sacrificing accuracy, I'll train a CNN using transfer learning.



