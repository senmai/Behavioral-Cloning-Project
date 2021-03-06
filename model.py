import csv
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

samples = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images = []
measurements = []
for line in samples:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split("/")[-1]
        current_path = "./data/IMG/" + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

augumented_images,augumented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measurement * -1.0)

X_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,MaxPooling2D,Dropout,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) # 160 * 320 * 3
model.add(Conv2D(96,(5,5),strides=(2,2))) #78 * 158 * 32
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(128,(5,5),strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation("relu")) #37 * 77 * 48
model.add(MaxPooling2D(2,2)) # 18 * 38 * 48
model.add(Conv2D(128,(3,3),strides=(1,1))) #16 * 36 * 64
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(256,(3,3),strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation("relu")) #14 * 34 * 128
model.add(MaxPooling2D(2,2)) # 7 * 17 * 128
model.add(Flatten()) # 15232
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(1))
model.compile(loss="mse",optimizer="adam")

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotaion_range = 20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)
model.fit_generator(data.gen.flow(X_train,y_train,batch_size=32),
        steps_per_epochs=len(X_train)/32,epochs=epochs)

print("model saved")
model.save("model.h5")
print("model summary")
print(model.summary())
