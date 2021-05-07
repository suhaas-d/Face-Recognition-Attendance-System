from keras_facenet import FaceNet
import cv2
embedder = FaceNet()
from PIL import Image, ImageFont, ImageDraw
import pickle
import numpy as np
import itertools
import datetime
from array import *
import os
import shutil


now=datetime.datetime.now()
#detections = embedder.extract('test.jpg', threshold = 0.95)

folder = './media'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#****First Computation Heavy part of the project****
def individual_encodings(file_names, class_name):
    encoding_dict = {}
    for i in file_names:
        encoding_dict[i[:-4]] = embedder.extract('./media/'+i, threshold = 0.95)[0]['embedding']
        os.remove('./media/'+i)
	
    pickle_file = class_name+'.pickle'
    with open(pickle_file,'wb') as stupickle:
    	pickle.dump(encoding_dict,stupickle)


def show_recognised_faces(test_image, boxes, n, keys):
    img = Image.open(r""+test_image)
    draw = ImageDraw.Draw(img)
    
     #detect faces

    #Draw Rectangles around the faces
    idx = 0    
    for ([x,y,w,h],key) in zip(boxes, keys):
        idx+=1
        draw.rectangle(((x,y),(x+w,y+h)), outline =(106, 153, 73), width = 10 )
        font = ImageFont.truetype("arial.ttf", 32)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((x, y+h+20),key,(255, 255, 255),font=font)
    #Exporting the result
    img.save('face_detected.jpg')
    print("succesfully saved cropped images from uploaded photo")

def get_attendance(test_image, class_name):
    encodings = {}
    database_encodings = {}
    detections = {}
    msgs = []
    file_loc = ""
    boxes = []
    keys = []

    #get encodings from stored pickle file/ database
    with open(class_name+'.pickle','rb') as stupickle:
        database_encodings = pickle.load(stupickle)
    
    val=[['DATE: '+ now.strftime("%x"),'','','STUDENTS','' ],['TIME ' + now.strftime("%X"),'','', 'ATTENDANCE','']]
    r=5
    c=0
    for key in database_encodings.keys():
        val[0].insert(r, key)
        r+=1
    for i in range(5, r):
        val[1].insert(i, 'Absent')
     
    

    #****second computation Heavy element of the project****
    #****most computation Heavy part of the project****
    #crop and get encodings of faces present in uploaded class image
    detections = embedder.extract(test_image)
    n = len(detections)
    for i in range(n):
        encodings['cropped_'+str(i)] = detections[i]['embedding']
        boxes.append(detections[i]['box'])
    
    
    #compare generatedd encodings with encodings in database to get attendance
    for key, value in encodings.items():
        min_diff = 100	
        remember_key = ""
        for keydb, valuedb in database_encodings.items():
            diff = np.linalg.norm(value - valuedb)
            if diff < min_diff:
                min_diff = diff
                remember_key = keydb
        if min_diff < 0.8:
            for i in range(6, r+1):
                if val[0][i-1] == remember_key:
                    val[1][i-1] = 'Present'
    
    
        keys.append(key)
    for i in range(5, r):
        if val[1][i] ==  'Absent' :
            msgs.append(val[0][i]+' is Absent')
    show_recognised_faces(test_image, boxes, n, keys)
    return msgs, val


