from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from mtcnn.mtcnn import MTCNN 
import cv2

from keras.models import model_from_json
from os import listdir

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))


model.load_weights('D:/DS/face_detect//vgg_face_weights.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


vgg_face_representer = Model(inputs = model.layers[0].input, outputs = model.layers[-2].output)

def findCosineSimilarity(source_representation, test_representation):
 	a = np.matmul(np.transpose(source_representation), test_representation)
 	b = np.sum(np.multiply(source_representation, source_representation))
 	c = np.sum(np.multiply(test_representation, test_representation))
 	return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

epsilon = 0.40 #cosine similarity
#epsilon = 120 #euclidean distance


 
employee_pictures = "D:/DS/face_detect/images/"

employees = dict()

for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    employees[employee] = vgg_face_representer.predict(preprocess_image('D:/DS/face_detect/images//%s.jpg' % (employee)))[0,:]

color = (155,0,155)
detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    __, frame = cap.read()

    result = detector.detect_faces(frame)

    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]
            
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv2.resize(detected_face, (224,224))

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = preprocess_input(img_pixels)

            captured_representation = vgg_face_representer.predict(img_pixels)[0,:]

            found = 0

            for i in employees:
                employee_name = i
                representation = employees[i]

                similarity = findEuclideanDistance(representation, captured_representation)
                
                if(similarity < 90):
                    cv2.putText(frame, employee_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    found = 1
                    break


                cv2.line(frame,(int((x+x+w)/2), y+15), (x+w, y-20), color, 1)
                cv2.line(frame,(x+w, y-20), (x+w+10, y-20), color, 1)

#                if(found==0):
#                    cv2.putText(frame, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.rectangle(frame, 
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),
                            (0,155,255),
                            2)
 
            cv2.circle(frame, (keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['mouth_right']), 2, (0,155,255), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()


