import numpy as np
import seaborn as sns
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel
from sklearn import preprocessing
from skimage.feature import graycomatrix,graycoprops
#from co_mat import cal_comat, cal_feat 
from skimage.measure import shannon_entropy 

#print(os.listdir("images/pc_parts"))

#extracting train images from the files using glob
images = []
labels = []

for dir_path in glob.glob("images/train/*"):
  label = dir_path.split("\\")[-1]
 # print(label)
  for img_path in glob.glob(os.path.join(dir_path,"*.jpg")):
    #print(img_path)
    img = cv2.imread(img_path,0)
    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

le = preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels)

labels = labels_encoded


def feat_extraction(dataset):
  image_dataset = pd.DataFrame()
  for image in range(dataset.shape[0]):
    df = pd.DataFrame()
    img = dataset[image,:,:]

    GLCM = graycomatrix(img,[1],[0])
    GLCM_Energy = graycoprops(GLCM,'energy')[0]
    df['energy'] = GLCM_Energy
    GLCM_Corr = graycoprops(GLCM,'correlation')[0]
    df['correlation'] = GLCM_Corr
    GLCM_Diss = graycoprops(GLCM,'dissimilarity')[0]
    df['dissimilarity'] = GLCM_Diss
    GLCM_Homo = graycoprops(GLCM,'homogeneity')[0]
    df['homogeneity'] = GLCM_Homo
    GLCM_Cont = graycoprops(GLCM,'contrast')[0]
    df['contrast'] = GLCM_Cont

    GLCM2 = graycomatrix(img,[3],[0])
    GLCM_Energy2 = graycoprops(GLCM2,'energy')[0]
    df['energy2'] = GLCM_Energy2
    GLCM_Corr2 = graycoprops(GLCM2,'correlation')[0]
    df['correlation2'] = GLCM_Corr2
    GLCM_Diss2 = graycoprops(GLCM2,'dissimilarity')[0]
    df['dissimilarity2'] = GLCM_Diss2
    GLCM_Homo2 = graycoprops(GLCM2,'homogeneity')[0]
    df['homogeneity2'] = GLCM_Homo2
    GLCM_Cont2 = graycoprops(GLCM2,'contrast')[0]
    df['contrast2'] = GLCM_Cont2

    
    GLCM3 = graycomatrix(img,[5],[0])
    GLCM_Energy3 = graycoprops(GLCM3,'energy')[0]
    df['energy3'] = GLCM_Energy3
    GLCM_Corr3 = graycoprops(GLCM3,'correlation')[0]
    df['correlation3'] = GLCM_Corr3
    GLCM_Diss3 = graycoprops(GLCM3,'dissimilarity')[0]
    df['dissimilarity3'] = GLCM_Diss3
    GLCM_Homo3 = graycoprops(GLCM3,'homogeneity')[0]
    df['homogeneity3'] = GLCM_Homo3
    GLCM_Cont3 = graycoprops(GLCM3,'contrast')[0]
    df['contrast3'] = GLCM_Cont3

    
    GLCM4 = graycomatrix(img,[0],[np.pi/4])
    GLCM_Energy4 = graycoprops(GLCM4,'energy')[0]
    df['energy4'] = GLCM_Energy4
    GLCM_Corr4 = graycoprops(GLCM4,'correlation')[0]
    df['correlation4'] = GLCM_Corr4
    GLCM_Diss4 = graycoprops(GLCM4,'dissimilarity')[0]
    df['dissimilarity4'] = GLCM_Diss4
    GLCM_Homo4 = graycoprops(GLCM4,'homogeneity')[0]
    df['homogeneity4'] = GLCM_Homo4
    GLCM_Cont4 = graycoprops(GLCM4,'contrast')[0]
    df['contrast4'] = GLCM_Cont4

    
    GLCM5 = graycomatrix(img,[0],[np.pi/2])
    GLCM_Energy5 = graycoprops(GLCM5,'energy')[0]
    df['energy5'] = GLCM_Energy5
    GLCM_Corr5 = graycoprops(GLCM5,'correlation')[0]
    df['correlation5'] = GLCM_Corr5
    GLCM_Diss5 = graycoprops(GLCM5,'dissimilarity')[0]
    df['dissimilarity5'] = GLCM_Diss5
    GLCM_Homo5 = graycoprops(GLCM5,'homogeneity')[0]
    df['homogeneity5'] = GLCM_Homo5
    GLCM_Cont5 = graycoprops(GLCM5,'contrast')[0]
    df['contrast5'] = GLCM_Cont5

    image_dataset = image_dataset._append(df)

  return image_dataset

image_features = feat_extraction(images)
image_features["labels"] = labels

image_features.to_csv("image_dataset.csv",index=False)

''''
#extracting test images from the files using glob
test_images = []
test_labels = []

for dir_path in glob.glob("images/validation/*"):
  label = dir_path.split("\\")[-1]
 # print(label)
  for img_path in glob.glob(os.path.join(dir_path,"*.jpg")):
    #print(img_path)
    img = cv2.imread(img_path,0)
    test_images.append(img)
    test_labels.append(label)

test_images = np.array(train_images)
test_labels = np.array(train_labels)
'''
#print(test_images.shape)

#encoding the labels to digits
'''
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#reassigning labels to the encoded
x_train, y_train, x_test, y_test = train_images,train_labels_encoded,test_images,test_labels_encoded

#feature extraction function



def feat_extraction(dataset):
  image_dataset = pd.DataFrame()
  for image in range(dataset.shape[0]):
    df = pd.DataFrame()
    img = dataset[image,:,:]

    GLCM = graycomatrix(img,[1],[0])
    GLCM_Energy = graycoprops(GLCM,'energy')[0]
    df['energy'] = GLCM_Energy
    GLCM_Corr = graycoprops(GLCM,'correlation')[0]
    df['correlation'] = GLCM_Corr
    GLCM_Diss = graycoprops(GLCM,'dissimilarity')[0]
    df['dissimilarity'] = GLCM_Diss
    GLCM_Homo = graycoprops(GLCM,'homogeneity')[0]
    df['homogeneity'] = GLCM_Homo
    GLCM_Cont = graycoprops(GLCM,'contrast')[0]
    df['contrast'] = GLCM_Cont

    GLCM2 = graycomatrix(img,[3],[0])
    GLCM_Energy2 = graycoprops(GLCM2,'energy')[0]
    df['energy2'] = GLCM_Energy2
    GLCM_Corr2 = graycoprops(GLCM2,'correlation')[0]
    df['correlation2'] = GLCM_Corr2
    GLCM_Diss2 = graycoprops(GLCM2,'dissimilarity')[0]
    df['dissimilarity2'] = GLCM_Diss2
    GLCM_Homo2 = graycoprops(GLCM2,'homogeneity')[0]
    df['homogeneity2'] = GLCM_Homo2
    GLCM_Cont2 = graycoprops(GLCM2,'contrast')[0]
    df['contrast2'] = GLCM_Cont2

    
    GLCM3 = graycomatrix(img,[5],[0])
    GLCM_Energy3 = graycoprops(GLCM3,'energy')[0]
    df['energy3'] = GLCM_Energy3
    GLCM_Corr3 = graycoprops(GLCM3,'correlation')[0]
    df['correlation3'] = GLCM_Corr3
    GLCM_Diss3 = graycoprops(GLCM3,'dissimilarity')[0]
    df['dissimilarity3'] = GLCM_Diss3
    GLCM_Homo3 = graycoprops(GLCM3,'homogeneity')[0]
    df['homogeneity3'] = GLCM_Homo3
    GLCM_Cont3 = graycoprops(GLCM3,'contrast')[0]
    df['contrast3'] = GLCM_Cont3

    
    GLCM4 = graycomatrix(img,[0],[np.pi/4])
    GLCM_Energy4 = graycoprops(GLCM4,'energy')[0]
    df['energy4'] = GLCM_Energy4
    GLCM_Corr4 = graycoprops(GLCM4,'correlation')[0]
    df['correlation4'] = GLCM_Corr4
    GLCM_Diss4 = graycoprops(GLCM4,'dissimilarity')[0]
    df['dissimilarity4'] = GLCM_Diss4
    GLCM_Homo4 = graycoprops(GLCM4,'homogeneity')[0]
    df['homogeneity4'] = GLCM_Homo4
    GLCM_Cont4 = graycoprops(GLCM4,'contrast')[0]
    df['contrast4'] = GLCM_Cont4

    
    GLCM5 = graycomatrix(img,[0],[np.pi/2])
    GLCM_Energy5 = graycoprops(GLCM5,'energy')[0]
    df['energy5'] = GLCM_Energy5
    GLCM_Corr5 = graycoprops(GLCM5,'correlation')[0]
    df['correlation5'] = GLCM_Corr5
    GLCM_Diss5 = graycoprops(GLCM5,'dissimilarity')[0]
    df['dissimilarity5'] = GLCM_Diss5
    GLCM_Homo5 = graycoprops(GLCM5,'homogeneity')[0]
    df['homogeneity5'] = GLCM_Homo5
    GLCM_Cont5 = graycoprops(GLCM5,'contrast')[0]
    df['contrast5'] = GLCM_Cont5

    image_dataset = image_dataset._append(df)

  return image_dataset


image_features = feat_extraction(x_train)
test_image_features = feat_extraction(x_test)

image_features["labels"] = y_train
test_image_features["labels"] = y_test

frames = [image_features,test_image_features]
image_data = pd.concat(frames)
image_data.to_csv("image_dataset.csv",index=False)

'''