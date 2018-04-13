import cv2
import os
import numpy as np

subjects = ["", "Vishal", "Elvis Presley", "Test case 3", "Test_case 4"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
    return faces
    
def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
     #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            faces1 = detect_face(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #for now ignoring the faces that are not detected
            if faces1[0] is not None:
                (x, y, w, h) = faces1[0]
                face = gray[y:y+w, x:x+h]
                rect = faces1[0]
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    #make a copy of the original image
    img = test_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces1 = detect_face(img)

    for i in faces1:
        (x, y, w, h) = i
        face = gray[y:y+w, x:x+h]
        rect = i
        #predict the image using our face recognizer 
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
#test_img4 = cv2.imread("test-data/test4.jpg")
#test_img5 = cv2.imread("test-data/test5.jpg")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
#predicted_img4 = predict(test_img4)
#predicted_img5 = predict(test_img5)
print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400,500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
#cv2.imshow(subjects[4], cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





