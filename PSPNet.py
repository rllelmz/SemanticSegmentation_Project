"""  Import our files from Google Drive in Google Colab
from google.colab import drive
drive.mount('/content/gdrive')
"""
import os
import segmentation_models as sm  # !pip install -U segmentation-models in Google Colab 
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

sm.set_framework('tf.keras')
sm.framework()

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Prepare our data
TRAINING_DIR = '/Users/reuellelk/Desktop/PROJET/training_images/'
MASKTRAIN_DIR = '/Users/reuellelk/Desktop/PROJET/training_masks/'
TESTING_DIR = '/Users/reuellelk/Desktop/Images/test_images/'
MASKTEST_DIR = '/Users/reuellelk/Desktop/Masks/test_masks/'

""" Prepare our data in Google Colab 
TRAINING_DIR = 'gdrive/MyDrive/training_images/'
MASKTRAIN_DIR = 'gdrive/MyDrive/training_masks/'
TESTING_DIR = 'gdrive/MyDrive/test_images/'
MASKTEST_DIR = 'gdrive/MyDrive/test_masks/'
"""

img_train = [os.path.join(TRAINING_DIR, filename) for filename in os.listdir(TRAINING_DIR) if filename.endswith('.jpg')]
mask_train = [os.path.join(MASKTRAIN_DIR, filename) for filename in os.listdir(MASKTRAIN_DIR) if
              filename.endswith('.png')]
img_test = [os.path.join(TESTING_DIR, filename) for filename in os.listdir(TESTING_DIR) if filename.endswith('.jpg')]
mask_test = [os.path.join(MASKTEST_DIR, filename) for filename in os.listdir(MASKTEST_DIR) if filename.endswith('.png')]

print("Length training images dataset : " + str(len(img_train)))
print("Length training masks dataset : " + str(len(mask_train)))
print("Length testing images dataset : " + str(len(img_test)))
print("Length testing masks dataset : " + str(len(mask_test)))

# Load our data
x_train = []
for f in img_train:
    image = PIL.Image.open(f)
    x_train.append(image)

y_train = []
for f1 in mask_train:
    image = PIL.Image.open(f1)
    y_train.append(image)

x_test = []
for f2 in img_test:
    image = PIL.Image.open(f2)
    x_test.append(image)

y_test = []
for f3 in mask_test:
    image = PIL.Image.open(f3)
    y_test.append(image)

# Resize images & masks
for i in range(len(x_train)):
    x_train[i] = np.resize(x_train[i], (480, 240, 3))

for j in range(len(y_train)):
    y_train[j] = np.resize(y_train[j], (480, 240, 3))

for k in range(len(x_test)):
    x_test[k] = np.resize(x_test[k], (480, 240, 3))

for l in range(len(y_test)):
    y_test[l] = np.resize(y_test[l], (480, 240, 3))

# Create Numpy arrays
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

print(str(x_train), str(y_train), str(x_test), str(y_test))
print(str(x_train.shape), str(y_train.shape), str(x_test.shape), str(y_test.shape))

# Preprocess input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Define model
model = sm.PSPNet(BACKBONE, input_shape=(480, 240, 3), classes=3, activation='softmax')
model.compile(
    'Adam',
    loss=sm.losses.categorical_focal_loss,
    metrics=[sm.metrics.f1_score],
)

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 5

# Fit model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    validation_split=0.2, )

# Plot training & validation f1 score
plt.figure(figsize=(30, 10))
plt.plot(history.history['f1-score'])
plt.plot(history.history['val_f1-score'])
plt.title('Model F1 SCORE')
plt.xlabel('EPOCHS')
plt.ylabel('F1_SCORE')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show() # In PyCharm

# Plot training & validation loss
plt.figure(figsize=(30, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show() # In PyCharm 

# Model evaluation 
model.evaluate(
    x=x_test,
    y=y_test,
    batch_size=2, )
