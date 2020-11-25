import face_recognition
import cv2
import os
import glob

def extract_faces(image):
    face_locations = face_recognition.face_locations(image)
    face_images = []
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        face_images.append(face_image)
    
    return face_images

folders = ["test","train"]
subfolders = ["real","imposter"]
destination_prefix = "face-dataset"

for folder in folders:
    for subfolder in subfolders:
        # get all imagenames
        path = os.path.join(folder,subfolder,"*")
        images = glob.glob(path)
        # read one image extract face and save the face
        for image_name in images:
            img = cv2.imread(image_name)
            face_images = extract_faces(img)
            new_image_path = os.path.join(destination_prefix,folder,subfolder,image_name.split("/")[-1])
            print(new_image_path)
            cv2.imwrite(new_image_path, face_images[0])
