# Download and preprocess the dataset
#use to preprocess data
import numpy as np
import csv
from args import get
import tensorflow as tf
import cv2

image_size = (48, 48)
num_classes = 7

# chuyển ảnh sang dl ma trận
def pixel_preprocess(pixels):
    face = [int(pixel) for pixel in pixels.split(" ")]#chia anh thanh tung nhung pixel
    face = np.asarray(face).reshape(48, 48) #thay doi hinh dang cua anh nhung van giu nguyen gia tri data cua anh
    face = cv2.resize(face.astype('uint8'), image_size) #chuyen mang ve dang unit8, nghia la moi phan tu cua mang co gia tri 0-255
    face = face.astype('float32') #chuyen mang ve dang float 32
    return face

# Preprocess data
def preprocess(args): # đây là hàm phân loai ảnh vào trong các tập train, val và test
    X_train, y_train, X_dev, y_dev, X_test, y_test, weight = [], [], [], [], [], [], []
    with open(args.dataset_path, 'r') as file:
        data = csv.reader(file)
        for row in data:
            if row[-1] == 'Training':
                X_train.append(pixel_preprocess(row[1]))
                emotion = tf.keras.utils.to_categorical(row[0], num_classes)
                y_train.append(emotion)
                weight.append(int(row[0]))

            elif row[-1] == 'PublicTest':
                X_dev.append(pixel_preprocess(row[1]))
                emotion = tf.keras.utils.to_categorical(row[0], num_classes)
                y_dev.append(emotion)

            elif row[-1] == 'PrivateTest':
                X_test.append(pixel_preprocess(row[1]))
                emotion = tf.keras.utils.to_categorical(row[0], num_classes)
                y_test.append(emotion)

        X_train = np.asarray(X_train).astype('float32') / 255.0
        X_dev = np.asarray(X_dev).astype('float32') / 255.0
        X_test = np.asarray(X_test).astype('float32') / 255.0

        X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
        X_dev = np.repeat(X_dev[..., np.newaxis], 3, -1)
        X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

        y_train = np.asarray(y_train).astype('float32')
        y_dev = np.asarray(y_dev).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        return (X_train, y_train, X_dev, y_dev, X_test, y_test, weight)

#đây là hàm lưu lại tất cả sau khi phân loai ảnh sau
def save(args, X_train, y_train, X_dev, y_dev, X_test, y_test, weight):
    np.save(args.train_faces, X_train)
    np.save(args.train_labels, y_train)
    np.save(args.dev_faces, X_dev)
    np.save(args.dev_labels, y_dev)
    np.save(args.test_faces, X_test)
    np.save(args.test_labels, y_test)
    np.save('./data/weight.npy', weight)

if __name__ == "__main__":
    args_ = get_setup_args()
    X_train, y_train, X_dev, y_dev, X_test, y_test, weight = preprocess(args_)
    if args_.save:
        save(args_, X_train, y_train, X_dev, y_dev, X_test, y_test, weight)
        print("File save succesfully")





