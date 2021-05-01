from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import cv2 
from PIL import Image
import pickle
#Loading the face_recognition cascade


face_cascade = cv2.CascadeClassifier('./upload/haarcascade_frontalface_default.xml')


def crop_faces_first():	
	img = cv2.imread('test.jpg')
	im = Image.open(r"./upload/test.jpg")
	#detect faces
	 
	faces = face_cascade.detectMultiScale(img,1.5,5)
	cropped_images = []
	 #Draw Rectangles around the faces
	idx = 0
	for (x,y,w,h) in faces:
		idx +=1
		cv2.rectangle(img,(x,y),(x+w,y+h),(50,205,50),20)
		im1 = im.crop((x,y,x+w,y+h))
		im1 = im1.resize((96,96))
		im1.save('images/cropped_'+str(idx)+'.JPG')
	#Exporting the result
	cv2.imwrite("face_detected.png",img)
	print("succesfully saved reference images")


def crop_faces_test(test_img):	
	img = cv2.imread(test_img)
	im = Image.open(r""+test_img)
	#detect faces
	 
	faces = face_cascade.detectMultiScale(img,1.3,4)
	cropped_images = []
	 #Draw Rectangles arpund the faces
	idx = 0
	for (x,y,w,h) in faces:
		idx +=1
		cv2.rectangle(img,(x,y),(x+w,y+h),(50,205,50),20)
		im1 = im.crop((x,y,x+w,y+h))
		im1 = im1.resize((96,96))
		im1.save('./upload/test_images/cropped_'+str(idx)+'.JPG')
	#Exporting the result
	cv2.imwrite("face_detected.png",img)
	print("succesfully saved test images")
	
	FRmodel = faceRecoModel(input_shape=(3, 96, 96))
	print("Total Params:", FRmodel.count_params())

	
	FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
	load_weights_from_FaceNet(FRmodel)
	
	students_name_encodings = {}
	
	for filename in os.listdir('./upload/test_images'):
		students_name_encodings[filename] = img_to_encoding("./upload/test_images/"+filename,FRmodel)
		print(filename+' encoding is stored. ')
	with open('student_name_encodings.pkl','wb') as stupickle:
		pickle.dump(students_name_encodings,stupickle)
	return idx


	

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor-positive),axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor-negative),axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    ### END CODE HERE ###
    
    return loss


def not_req():
	FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
	load_weights_from_FaceNet(FRmodel)

	database = {}
	database["scarjo"] = img_to_encoding("images/scarjo.JPG",FRmodel)
	database["downey"] = img_to_encoding("images/downey.JPG",FRmodel)
	database["hemsworth"] = img_to_encoding("images/hemsworth.JPG",FRmodel)
	database["evans"] = img_to_encoding("images/evans.JPG",FRmodel)
	database["renner"] = img_to_encoding("images/renner.JPG",FRmodel)



	test_img = "test.jpg"
	no_of_faces_detected = crop_faces_test(test_img)
	for i in range(1,no_of_faces_detected+1):
		test_encoding = img_to_encoding("test_images/cropped_"+str(i)+".JPG",FRmodel)
		min_dist = 100
		for (name,encoding) in database.items():
			dist = np.linalg.norm(test_encoding - encoding)
			if dist<min_dist:
				identity = name
				min_dist = dist
		if min_dist>0.7 :
			print("not in database")
		else:
			print(str(identity)+" has been detected and the distance is "+str(min_dist)+"in image"+str(i))
	
			


