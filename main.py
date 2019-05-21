import os
import cv2
import numpy as np

from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

from cnn.convolution import Convolution

LETTERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'буквы/')
CLASSES = {i: n for n, i in enumerate('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')}
R_CLASSES = {n: i for n, i in enumerate('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')}
data, labels = [], []


for folder in os.listdir(LETTERS_PATH):
    label = folder.replace('буква ', '').upper()
    label = CLASSES.get(label)
    cur = os.path.join(LETTERS_PATH, folder)

    for letter in os.listdir(cur):
        path = cur + '/' + letter
        im1 = cv2.imread(path, 1)
        im1 = cv2.resize(im1, (28, 28), interpolation=cv2.INTER_AREA)
        im_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        ret1, roi = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        data.append(roi)
        labels.append(label)

classes = list(set(labels))
_classes = len(classes)
_labels = np.asarray(labels).reshape((len(data), -1))
data = np.asarray(data).reshape((len(data), -1))

data = data.reshape((data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(train_data, test_data, train_labels, test_labels) = train_test_split(
    data / 255.0, _labels, test_size=0.2)

train_labels = np_utils.to_categorical(train_labels, _classes + 1)
test_labels = np_utils.to_categorical(test_labels, _classes + 1)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = Convolution.build(width=28,
                          height=28,
                          depth=1,
                          classes=_classes + 1,
                          weights_path=None)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training...")
model.fit(
    train_data,
    train_labels,
    batch_size=128,
    nb_epoch=130,
    verbose=1
)

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(
    test_data,
    test_labels,
    batch_size=128,
    verbose=1
)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

loss, accuracy = model.evaluate(test_data, test_labels, batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


for i in np.random.choice(np.arange(0, len(test_labels)), size=(10,)):
    probs = model.predict(test_data[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    image = (test_data[i][0] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    print("[INFO] Predicted: {}, Actual: {}".format(
        R_CLASSES.get(prediction[0]),
        R_CLASSES.get(np.argmax(test_labels[i])))
    )

    cv2.imshow("Digit", image)
    cv2.waitKey(0)
