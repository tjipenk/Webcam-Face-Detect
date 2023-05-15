import cv2
import sys
import logging as log
import datetime as dt
import base64, requests, json
from time import sleep


# cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

def base64img(img):
    # Convert the image to base64 format
    with open(img, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')
    # print(encoded_image)
    return str(encoded_image)
        

def get_face(b64img):
    try:
        url = "http://10.10.1.79:8011//api/v1/recognition/recognize?limit=0&det_prob_threshold=0.0&prediction_count=1&status=true"
        payload = json.dumps({
            "file": b64img
            })
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': '2ce78139-36c0-4109-8df6-7dd3ce41edaa'
            }
        response = requests.request("POST", url, headers=headers, data=payload)
        #print(payload)
        hasil = json.loads(response.text)
        
        return hasil['result'][0]['subjects'][0]['subject'], hasil['result'][0]['subjects'][0]['similarity']
    except:
        return 'unrecognize', '0'


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(faces)
        crop_face = gray[y-10:y + h+10, x-10:x + w +10]
        retval, buffer_img= cv2.imencode('.jpg', crop_face)
        b64img = base64.b64encode(buffer_img).decode('ascii')
        # cv2.putText(frame, 'base64: '+b64img, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        npp, sim = get_face(b64img)
        # cv2.putText(frame, 'NPP: '+npp+' Similarity: '+ sim)
        cv2.putText(frame,'NPP: '+str(npp)+' Similarity: '+ str(sim), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
