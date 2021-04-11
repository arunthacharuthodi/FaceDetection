import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from PIL import Image
import base64
from io import BytesIO
import json






# video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# count = 0
#
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# load function
def face_extractor(img):
    faces = faceCascade.detectMultiScale(img, 1.1, 4)

    if faces is ():
        return  None

    for (x, y, w, h) in faces:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return  cropped_face
# #
#
#
#
#
#
# while True:
#
#     success,img = video.read()
#     # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if face_extractor(img) is not None:
#         count += 1
#         face = cv2.resize(face_extractor(img), (400, 400))
#         file_name_path = 'dataset/train/aswathy/' + str(count) + '.jpg'
#         cv2.imwrite(file_name_path, face)
#
#         #live count on images
#         cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow('face cropper', face)
#
#     else:
#         print("face not found")
#         pass
#     if cv2.waitKey(1) == 13 or count == 100:
#         break
#
# video.release()
# cv2.destroyAllWindows()
# print("collecting samples completed")
#
#
#
#
#
#
#
#
# IMAGE_SIZE = [224, 224]
#
# tarin_path = 'dataset/train'
# valid_path = 'dataset/test'
#
# vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)
#
# for layers in vgg.layers:
#     layers.trainable = False
#
#
#
# folders = glob('dataset/train/*')
#
# x = Flatten()(vgg.output)
#
# prediction = Dense(len(folders), activation='softmax')(x)
#
# model = Model(inputs=vgg.input, outputs=prediction)
#
# model.summary()
#
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# train_datagen = ImageDataGenerator(rescale= 1./255,
#                                    shear_range=0.2,
#                                    zoom_range= 0.2,
#                                    horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# training_set = train_datagen.flow_from_directory('dataset/train',
#                                                  target_size=(224, 224),
#                                                  batch_size= 32,
#                                                  class_mode= 'categorical')
#
# test_set = test_datagen.flow_from_directory('dataset/test',
#                                                  target_size=(224, 224),
#                                                  batch_size= 32,
#                                                  class_mode= 'categorical')
#
# r = model.fit_generator(
#     training_set,
#     validation_data=test_set,
#     epochs=5,
#     steps_per_epoch=len(training_set),
#     validation_steps=len(test_set)
# )
# model.save('facefeatures_model1.h5')







model_ld = load_model('facefeatures_model1.h5')
print(type(model_ld))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    sucess, frame = cap.read()

    face_re = face_extractor(frame)
    cv2.imshow("detect", face_re)
    if type(face_re) is np.ndarray:
        face_re = cv2.resize(face_re, (224, 224))
        im = Image.fromarray(face_re, 'RGB')
        im_array = np.array(im)
        im_array = np.expand_dims(im_array, axis=0)
        pred = model_ld.predict(im_array)
        print(pred)



# from PyPDF2 import PdfFileMerger
# import os
#
#
# list1 = os.listdir('c:\\Users\\hp\\work1')
# print(list1)
#
# pdfs = ['Basic Integral Formula.pdf', 'Basic Trigonometric Formula.pdf']
#
# merger = PdfFileMerger()
#
# for pdf in pdfs:
#     merger.append(pdf)
#
# merger.write("result.pdf")
# merger.close()
