import cv2
from os import walk

imagePathInput = "F:/project_python/ClassificationFace/MyTam/" # duong dan anh input
cascPath = "haarcascade_frontalface_default.xml"
imagePathOutput = "F:/project_python/ClassificationFace/MyTamOutPut/" #duong dan output

def detect_face(filepath):
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        return x , y , w , h , image


def crop_image(x , y , w , h , image,filename):
    crop_img = image[y:y+h, x:x+w]
    cv2.imwrite(imagePathOutput+filename, crop_img)

for (dirpath, dirnames, filenames) in walk(imagePathInput):
    for filename in filenames:
        print(filename)
        filepath = imagePathInput+filename
        x , y , w , h , image = detect_face(filepath)
        crop_image(x , y , w , h , image,filename)


