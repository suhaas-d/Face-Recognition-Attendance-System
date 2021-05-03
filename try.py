from keras_facenet import FaceNet
from PIL import Image
embedder = FaceNet()

detections = embedder.extract('test.jpg', threshold = 0.95)



def crop_for_attendance(test_img):
    print(detections)

    boxes = []
    for i in range(len(detections)):
        box = detections[i]['box']

    img = cv2.imread(test_img)
    im = Image.open(r""+test_img)
            #detect faces

    #Draw Rectangles around the faces
    #idx = 0
    for (x,y,w,h) in boxes:
        idx +=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(50,205,50),20)
        im1 = im.crop((x,y,x+w,y+h))
        im1 = im1.resize((96,96))
        im1.save('./upload/images/cropped_'+str(idx)+'.JPG')
    #Exporting the result
    cv2.imwrite("face_detected.png",img)
    print("succesfully saved cropped images from uploaded photo")

crop_for_attendance('test.jpg')