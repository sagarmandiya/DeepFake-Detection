import os
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7, ResNet50V2, VGG19
from tensorflow.keras.applications import Xception, InceptionResNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from scipy.interpolate import make_interp_spline, BSpline


# Initializing the Common Variables.
input_shape = (128, 128, 3)
batch_size = 64
epochs = 10
epoch_list = list(range(1, epochs+1))

# Path to training & testing set.
train_dir = './Datasets/filtered-dataset-full/training'
test_dir = './Datasets/filtered-dataset-full/testing'

train_dir_fake, test_dir_fake = os.path.join(train_dir, 'fake'), os.path.join(test_dir, 'fake')
train_dir_real, test_dir_real = os.path.join(train_dir, 'real'), os.path.join(test_dir, 'real')

train_fake_fnames, test_fake_fnames = os.listdir(train_dir_fake), os.listdir(test_dir_fake)
train_real_fnames, test_real_fnames = os.listdir(train_dir_real), os.listdir(test_dir_real)

# Training Data Generator.
train_datagen = ImageDataGenerator(rescale=1./255.)

# Testing Data Generator.
test_datagen = ImageDataGenerator(rescale=1./255.)

# Flow training images in batches of 64 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(128, 128),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

# Flow test images in batches of 64 using test_datagen generator
test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(128, 128),
                                                    batch_size=batch_size,
                                                    class_mode='binary')


# Loading the Neural Network & Pre-Trained Weights.
ResNet50V2_model = ResNet50V2(input_shape=input_shape, include_top=False, weights="imagenet", classes=2)

for i in range(50):
    l = ResNet50V2_model.get_layer(index=i)
    l.trainable = True

# Extending the Neural Network Sequentially.
model = Sequential()
model.add(ResNet50V2_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the Model.
model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0),
                metrics=['accuracy'])


# Early Stopping mechanism to stop training if the loss rate is unchanged for certain epochs.
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')

# Starting the Training.
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator)


# Plotting the graphs
network_name = "ResNet50V2"

try:
    os.mkdir("./Reference_Data")
    os.mkdir("./Reference_Data/Graphs")
    os.mkdir("./Reference_Data/Summary")
    os.mkdir("./Reference_Data/Model")
except OSError:
    pass

try:
    os.mkdir(os.path.join("./Reference_Data/Graphs", network_name))
except OSError:
    pass

#define x as 200 equally spaced values between the min and max of original x 
acc = np.linspace(min(epoch_list), max(epoch_list), 200) 
val_acc = np.linspace(min(epoch_list), max(epoch_list), 200) 

#define spline for accuracy
spl1 = make_interp_spline(epoch_list, history.history['accuracy'], k=3)
y_smooth1 = spl1(acc)
#define spline accuracy
spl2 = make_interp_spline(epoch_list, history.history['val_accuracy'], k=3)
y_smooth2 = spl2(val_acc)

#create smooth line chart 
graph_1 = plt.subplots(1, 1)
plt.suptitle('Training & Testing Accuracy v/s Number of Epochs.', fontsize=10)
plt.plot(acc, y_smooth1, label='Train Accuracy')
plt.plot(val_acc, y_smooth2, label='Validation Accuracy')
plt.xticks(np.arange(1, epochs + 1, 1))
plt.ylabel('Accuracy Value')
plt.xlabel('Epoch')
plt.title('Accuracy')
plt.legend(loc="best")
plt.savefig(os.path.join("./Reference_Data/Graphs", network_name, "AccuracyVEpochs.png"), dpi=300,
            bbox_inches='tight')

#define x as 200 equally spaced values between the min and max of original x 
loss = np.linspace(min(epoch_list), max(epoch_list), 200) 
val_loss = np.linspace(min(epoch_list), max(epoch_list), 200) 

#define spline for accuracy
spl3 = make_interp_spline(epoch_list, history.history['loss'], k=3)
y_smooth3 = spl3(loss)
#define spline accuracy
spl4 = make_interp_spline(epoch_list, history.history['val_loss'], k=3)
y_smooth4 = spl4(val_loss)

#create smooth line chart 
graph_2 = plt.subplots(1, 1)
plt.suptitle('Training & Testing Loss v/s Number of Epochs.', fontsize=10)
plt.plot(loss, y_smooth3, label='Train Loss')
plt.plot(val_loss, y_smooth4, label='Validation Loss')
plt.xticks(np.arange(1, epochs + 1, 1))
plt.ylabel('Loss Value')
plt.xlabel('Epoch')
plt.title('Loss')
plt.legend(loc="best")
plt.savefig(os.path.join("./Reference_Data/Graphs", network_name, "LossVEpochs.png"), dpi=300, 
            bbox_inches='tight')
plt.show()


# Saving model summary
with open("./Reference_Data/Summary/" + network_name + "summary.txt", 'w+') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Saving the Model for Inference Purpose.
model.save('./Reference_Data/Model/' + network_name + '/')
model.save('./Reference_Data/Model/' + network_name + '/' + network_name + '.h5')
