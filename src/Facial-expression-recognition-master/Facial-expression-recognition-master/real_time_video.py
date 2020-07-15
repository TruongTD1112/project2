
import imutils
import cv2
from keras.models import load_model
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras import models
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np


def VGG19(num_classes, input_shape=(48, 48, 3), dropout=None, block5=True, batch_norm=True):
    img_input = layers.Input(shape=input_shape)

    # Block1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1')(img_input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv3')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv4')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block4
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv1')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv2')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv4')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block5
    if block5:
        x = layers.Conv2D(512, (3, 3),
                          padding='same',
                          name='block5_conv1')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(512, (3, 3),
                          padding='same',
                          name='block5_conv2')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(512, (3, 3),
                          padding='same',
                          name='block5_conv4')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = layers.AveragePooling2D((1, 1), strides=(1, 1), name='block6_pool')(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = models.Model(img_input, x, name='vgg19')
    return model


lr = 1e-2
momentum = 0.9
epochs = 100
batch_size = 128
# tham số để load data và ảnh
detection_model_path = '/content/drive/My Drive/fer2013/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '/content/drive/My Drive/fer2013/model/VGG19_nonBN.h5'

# khơỉ tạo siêu tham số
# load models
sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=lr / epochs)
adam_ = optimizers.Adam(learning_rate=1e-4) # chọn learning rate = 0.0001

model = VGG19(num_classes=7, input_shape=(48, 48, 3), batch_norm=False)
model.compile(optimizer=adam_, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('/content/drive/My Drive/fer2013/model/VGG19_nonBN.h5')

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = model
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    # reading the frame
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.repeat(roi[..., np.newaxis], 3, -1)
        roi = np.reshape(roi, (1, 48, 48, 3))

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        # draw the label + probability bar on the canvas
        # emoji_face = feelings_faces[np.argmax(preds)]
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()