import cv2
import numpy as np
import face_recognition

img1 = face_recognition.load_image_file('images/ppbinus.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = face_recognition.load_image_file('images/martin1.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

facelocation = face_recognition.face_locations(img1)[0]
encodeimg1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]),(255,0,255),2)

facelocation2 = face_recognition.face_locations(img2)[0]
encodeimg2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(facelocation2[3],facelocation2[0]),(facelocation2[1],facelocation2[2]),(255,0,255),2)

res = face_recognition.compare_faces([encodeimg1], encodeimg2)
faceDistance = face_recognition.face_distance([encodeimg1],encodeimg2)
print("Hasil: ", res,faceDistance)
cv2.putText(img2,f'{res}{round(faceDistance[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Martin',img1)
cv2.imshow('Martin_binus',img2)
cv2.waitKey(0)