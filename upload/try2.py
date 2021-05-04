from keras_facenet import FaceNet
import cv2
embedder = FaceNet()
from PIL import Image, ImageFont, ImageDraw
import pickle
import numpy as np
import itertools
#detections = embedder.extract('test.jpg', threshold = 0.95)

'''print(detections)

boxes = []

for i in range(len(detections)):
    boxes.append(detections[i]['box'])

img = cv2.imread('test.jpg')
im = Image.open(r"test.jpg")
 #detect faces

#Draw Rectangles around the faces
idx = 0    
for [x,y,w,h] in boxes:
    idx+=1
    cv2.rectangle(img,(x,y),(x+w,y+h),(50,205,50),20)
    im1 = im.crop((x,y,x+w,y+h))
#Exporting the result
cv2.imwrite("face_detected.png",img)
print("succesfully saved cropped images from uploaded photo")'''

#****First Computation Heavy part of the project****
def individual_encodings(file_names, class_name):
    encoding_dict = {}
    for i in file_names:
    		encoding_dict[i[:-4]] = embedder.extract('./media/'+i, threshold = 0.95)[0]['embedding']
		#os.remove('./media/'+i)
	
    print('I am here')
    print(encoding_dict)
    pickle_file = class_name+'.pickle'
    with open(pickle_file,'wb') as stupickle:
    	pickle.dump(encoding_dict,stupickle)
    print('I am done here')


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
    msg_dict = {}
    file_loc = ""
    boxes = []
    keys = []

    #get encodings from stored pickle file/ database
    with open(class_name+'.pickle','rb') as stupickle:
        database_encodings = pickle.load(stupickle)
    
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
            print(key, keydb, diff)
            if diff < min_diff:
                min_diff = diff
                remember_key = keydb
        if min_diff < 0.8:
            msg_dict[key] = remember_key+' is present in the class and the distance is '+ str(min_diff)+' cropped image is'+ key
        else:
            msg_dict[key] = key+' has not been identified as anyone present in the database, please check again and min dist is' + str(min_diff)+'with '+remember_key

        '''for filename in os.listdir('./upload/images'):
            fille_loc = './upload/images'+filename
            os.remove(file_loc)'''
        keys.append(key)
    print(msg_dict)
    show_recognised_faces(test_image, boxes, n, keys)
    return msg_dict