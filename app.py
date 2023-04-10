# install opencv, pyttsx3
import cv2
import pyttsx3
import time
# creating pyttsx3 object
engine = pyttsx3.init()
import cv2
import matplotlib.pyplot as plt
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels=[]
file_name='Labels.txt'

with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
# distance from camera to object(face) measured
# centimeter
Known_distance = 76.2
 
# width of face in the real world or Object Plane
# centimeter
Known_width = 14.3
 
# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
 
# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX
 
# face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
 
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length
 
# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
 
    # return the distance
    return distance
 
 
def face_data(image):
 
    face_width = 0  # making face width to zero
 
    # converting color image ot gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
 
    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
 
        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
 
        # getting face width in the pixels
        face_width = w    
 
    # return the face width in pixel
    return face_width

# reading reference_image from directory
ref_image_face = cv2.imread("Ref_image_face.jpg")
 
# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image_face)
 
# get the focal by calling "Focal_Length_Finder"
# face width in reference(pixels),
# Known_distance(centimeters),
# known_width(centimeters)
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)   
 
print(Focal_length_found)

# initialize the camera object so that we
# can get frame from it
#cap = cv2.VideoCapture(0)
 
# looping through frame, incoming from
# camera/video
cap= cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")
font_scale=3
font =cv2.FONT_HERSHEY_PLAIN
while True:
 
    # reading the frame from camera
    _, frame = cap.read()
 
    # calling face_data function to find
    # the width of face(pixels) in the frame
    face_width_in_frame = face_data(frame)
 
    # check if the face is zero then not
    # find the distance
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[1]+40,boxes[0]+10),font,fontScale=font_scale,color=(0,255,0),thickness=3)
    if face_width_in_frame != 0:
       
        # finding the distance by calling function
        # Distance distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)

        Distance= round(Distance,0)

        text = "Man detected at a distance of "+ str(Distance)+" centimetres "
        engine.say(text)
        engine.runAndWait()

        time.sleep(5)        

        # draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
 
        # Drawing Text on the screen
        cv2.putText(frame, f"Distance: {round(Distance,2)} CM", (30, 35), fonts, 0.6, GREEN, 2)
 
 
    # show the frame on the screen
    cv2.imshow("frame", frame)
 
    # quit the program if you press 'q' on keyboard
    if cv2.waitKey(1) == ord("q"):
        break
 
# closing the camera
cap.release()
 
# closing the the windows that are opened
cv2.destroyAllWindows()
