import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    Cropping2D,
    Lambda,
    Convolution2D,
    Dropout,
    MaxPooling2D,
    Activation,
)
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

lines = []

# Data Provided By Udacity
folder_name = "data3/"
with open(folder_name + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0] = folder_name + line[0]
        line[1] = folder_name + line[1].strip()
        line[2] = folder_name + line[2].strip()
        lines.append(line)

# My training data, Append with lines
folder_name = "test/"
with open(folder_name + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0] = folder_name + "IMG/" + line[0].split("\\")[-1]
        line[1] = folder_name + "IMG/" + line[1].split("\\")[-1]
        line[2] = folder_name + "IMG/" + line[2].split("\\")[-1]
        lines.append(line)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])

                left_image = cv2.imread(batch_sample[1])
                left_angle = center_angle + 0.25
                images.append(left_image)
                angles.append(left_angle)

                right_image = cv2.imread(batch_sample[2])
                right_angle = center_angle - 0.25
                images.append(right_image)
                angles.append(right_angle)

                flip_image = np.fliplr(cv2.imread(batch_sample[0]))
                flip_angle = -1 * center_angle
                images.append(flip_image)
                angles.append(flip_angle)

                center_image = cv2.imread(batch_sample[0])
                center_angle = center_angle
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Split the data between training and validation set (80% vs 20%)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Training Model Architecture
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda X: X / 127.5 - 1))

model.add(Convolution2D(24, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(48, 5, 5))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=5,
    verbose=1,
)

### print the keys contained in the history object
model.save("model.h5")
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history["loss"])
plt.plot(history_object.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()
