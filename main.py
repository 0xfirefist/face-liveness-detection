import cv2
from liveness_model.model import LivenessNet
from keras.models import load_model
import face_recognition
from keras.preprocessing.image import img_to_array
import numpy as np

def extract_faces(image):
    face_locations = face_recognition.face_locations(image)
    face_images = []
    locs = []
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        locs.append([top, right, bottom, left])
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)
    
    return face_images, locs

# loading models
model = load_model("liveness_model/model.h5")

cap = cv2.VideoCapture(0)

checker = 0
while True:

    ret, frame = cap.read()
    if checker < 50:
        checker += 1
        continue

    a, locs = extract_faces(frame)
    if len(a) < 1:
        continue
    face_image = a[0]
    face_image = cv2.resize(face_image, (64,64))
    face = img_to_array(face_image)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face)
    predicted_class_indices=np.argmax(preds,axis=1)
    if(predicted_class_indices[0] == 1):
        cv2.putText(frame, 'REAL', (frame.shape[1]//2, frame.shape[0]//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)
    else:
        cv2.putText(frame, 'FAKE', (frame.shape[1]//2, frame.shape[0]//2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)


    top,right,bottom,left = locs[0]
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imshow('testing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()