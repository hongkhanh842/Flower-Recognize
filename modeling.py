#1 Install Dependencies and Setup
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import dataframe_image as dfi

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

#2 Load & Scale data
data_dir = 'data'

data = []
label = []

SIZE = 128 # Crop the image to 128x128

for folder in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, folder)):
        if file.endswith("jpeg"):
            label.append(folder)
            img = cv2.imread(os.path.join(data_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (SIZE, SIZE))
            data.append(im)
            break
        else:
            continue
# convert the data into numerical values:
data_arr = np.array(data)
label_arr = np.array(label)
# use the Label encoder and normalize the data:
encoder = LabelEncoder()
y = encoder.fit_transform(label_arr)
y = to_categorical(y, 10)
X = data_arr/255
#3 Split data 70-18-12: train-val-test
X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, y, test_size=0.30, random_state=10)
X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test, test_size=0.4, random_state=10)
#4 Build a neural network model
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation ='relu', input_shape = (SIZE, SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation = "softmax"))
# create more training images to prevent overfitting:
datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range = 0.20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)
datagen.fit(X_train)
#5 compile the neural network model:
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
batch_size = 32
epochs = 64
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs = epochs,
                              validation_data = (X_validation, y_validation),
                              verbose = 1)
#6 Plot Performance
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.savefig("lost.jpeg")

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.savefig("accuracy.jpeg")

predicted = []
for i in range(len(X_test)):
    predicted.append(np.argmax(model.predict(np.expand_dims(X_test[i], 0))))

original = []
for i in range(len(y_test)):
    original.append(np.argmax(y_test[i]))

categories = np.sort(os.listdir(data_dir))
cm = confusion_matrix(original, predicted)
plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues, colorbar=True)
plt.xticks(range(len(categories)), categories, fontsize=10)
plt.yticks(range(len(categories)), categories, fontsize=16)
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.savefig("confusion.jpeg")


report = classification_report(original, predicted,
                               output_dict=True,
                               target_names=categories)
report_df = pd.DataFrame(report).transpose()

pd.set_option("display.max_rows", None)
dfi.export(report_df.style.background_gradient(), 'report.jpeg')
#7 Save the model
model.save(os.path.join('models', 'flowerclassifier.h5'))
#8 Test
fig, ax = plt.subplots(6, 6, figsize=(25, 40))

for i in range(6):
    for j in range(6):
        k = int(np.random.random_sample() * len(X_test))
        if(categories[np.argmax(y_test[k])] == categories[np.argmax(model.predict(X_test)[k])]):
            ax[i, j].set_title("TRUE: " + categories[np.argmax(y_test[k])], color='green')
            ax[i, j].set_xlabel("PREDICTED: " + categories[np.argmax(model.predict(X_test)[k])], color='green')
            ax[i, j].imshow(np.array(X_test)[k].reshape(SIZE, SIZE, 3), cmap='gray')
        else:
            ax[i, j].set_title("TRUE: " + categories[np.argmax(y_test[k])], color='red')
            ax[i, j].set_xlabel("PREDICTED: " + categories[np.argmax(model.predict(X_test)[k])], color='red')
            ax[i, j].imshow(np.array(X_test)[k].reshape(SIZE, SIZE, 3), cmap='gray')

plt.savefig("test.jpeg")
plt.show()