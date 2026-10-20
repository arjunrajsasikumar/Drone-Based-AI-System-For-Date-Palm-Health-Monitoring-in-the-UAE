## Data Pre Processing

import numpy as np
import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

# Getting images and labels from the dataset folder
dataset_path = os.listdir('dataset')
print()
print("=" * 50)
print("Datasets:")
print (dataset_path)  #what kinds of classes are in this dataset
print("Types of classes labels found: ", len(dataset_path))
print("=" * 50)

class_labels = []
for item in dataset_path:
 # Get all the file names
 all_classes = os.listdir('dataset' + '/' +item)
 # print(all_classes)
 # Add them to the list
 for room in all_classes:
    class_labels.append((item, str('dataset_path' + '/' +item) + '/' + room))
    #print(class_labels[:5])

# Build a dataframe        
df = pd.DataFrame(data=class_labels, columns=['Labels', 'image'])
print()
print("=" * 50)
print("Putting everything together:")
print(df.head())
print(df.tail())
print("=" * 50)

# Let's check how many samples for each category are present
print()
print("=" * 50)
print("Details of the list:")
print("Total number of images in the dataset: ", len(df))
label_count = df['Labels'].value_counts()
print(label_count)
print("=" * 50)

import cv2
path = 'dataset/'
dataset_path = os.listdir('dataset')
im_size = 224
images = []
labels = []
for i in dataset_path:
    data_path = path + str(i)  
    filenames = [i for i in os.listdir(data_path) ]
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)


#This model takes input images of shape (224, 224, 3), and the input data should range [0, 255]. 
images = np.array(images)
images = images.astype('float32') / 255.0
print()
print("Images shape: ")
images.shape

print()
print("=" * 50)
print("One Hot Encoding the labels")
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
y=df['Labels'].values
print(y)
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)


y=y.reshape(-1,1)
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
Y = ct.fit_transform(y).toarray()
print(Y[:5])
print(Y[35:])
print("=" * 50)


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
images, Y = shuffle(images, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.2, random_state=415)
#inpect the shape of the training and testing.
print()
print("=" * 50)
print("Shape of the training and testing data")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print("=" * 50)

## EfficientNetB0 model for multi-class classification

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
NUM_CLASSES = 4
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# Using model without transfer learning
outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )
model.summary()
hist = model.fit(train_x, train_y, epochs=5, verbose=1)

import matplotlib.pyplot as plt
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
plot_hist(hist)

print()
print("=" * 50)
preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print("=" * 50)

from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
img_path = 'unseen.png'
#img = image.load_img(img_path, target_size=(224, 224))
#x = img.img_to_array(img)
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = imread(img_path)
imshow(my_image)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names = ["Healthy", "Magnesium Deficiency", "Manganese Deficiency", "Potassium Deficiency"]

y_pred = np.argmax(model.predict(test_x, verbose=0), axis=1)
y_true = np.argmax(test_y, axis=1)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", xticks_rotation=30)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print()
print("=" * 50)
print("Prediction for unseen image:")
preds=model.predict(x)
print(preds)     # probabilities for being in each of the 4 classes
preds = model.predict(x)[0]
classes = ["Healthy", "Magnesium Deficiency", "Manganese Deficiency", "Potassium Deficiency"]
print("\nPrediction for unseen image:")
for name, p in zip(classes, preds):
    print(f"  {name:<22} {p * 100:5.1f}%")
print(f"\nPredicted: {classes[preds.argmax()]}")
print("=" * 50)
