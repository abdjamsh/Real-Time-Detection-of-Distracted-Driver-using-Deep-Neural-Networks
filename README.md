# Distracted-Driver-using-Deep-Neural-Networks
Real time driver distraction using state farm dataset
In this notebook, I'll use the dataset which includes images of drivers while performing a number of tasks including drinking, texting etc. The aim is to correctly identify if the driver is distracted from driving. We might also like to check what activity the person is performing.

The notebook will be borken into the following steps:

1. Import the Libraries.
2. Import the Datasets.
3. Create a vanilla CNN model.
4. Create a vanilla CNN model with data augmentation.
5. Train a CNN with Transfer Learning (VGG16).
6. Results.

Import the Libraries
I'll use Keras and Tensorflow libraries to create a Convolutional Neural Network. So, I'll import the necessary libraries to do the same.

from glob import glob
import random
import time
import tensorflow
import datetime
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

from tqdm import tqdm

import numpy as np
import pandas as pd
from IPython.display import FileLink
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
%matplotlib inline
from IPython.display import display, Image
import matplotlib.image as mpimg
import cv2

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files       
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
