""" face and eye detection"""
# importing  open cv2 and datetime files
import cv2
import datetime

# reading the haarcascade xml files
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# camera capturing
cap=cv2.VideoCapture(0)

# while loop for capturing frames from camera
while cap.isOpened():
    _, frame = cap.read() # capturing frames
    date=str(datetime.datetime.now()) # data and time setting
    font=cv2.FONT_HERSHEY_COMPLEX
    frame = cv2.putText(frame,date,(10,25),font,1,(204,88,88),2,cv2.LINE_AA)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # converting color frame to gray frame
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)  # forming rectangle for face
        font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL
        text_face='FACE'
        frame = cv2.putText(frame,text_face,(x,y-5),font_face,1,(0,0,0),1,cv2.LINE_AA)
        eye_gray=gray[y:y+h, x:x+w]
        eye_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(eye_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eye_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)  # forming rectangle for eyes
            text_eye='EYES'
            frame_eye=cv2.putText(eye_color,text_eye,(ex,ey-5),font_face,1,(0,0,0),1,cv2.LINE_AA)



    cv2.imshow('frame',frame)

    # quit the task when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
