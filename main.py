import numpy as np
import pandas as pd
import keras.backend as K

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score, f1_score

from utils import getFilesInDir, create_labels
from rnn_backend import RnnClassifier, preprocess_input_rnn
from utils import one_hot_to_integer
from utils import one_hot
from utils import missclassification_rate
from utils import test_train_split

n_time_steps = 32
n_inputs = 64
fraction = 25


def img_to_tensor(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    tensor = img_to_array(img)
    # print(tensor.shape)
    tensor = np.expand_dims(tensor, axis=0)
    # tensor = preprocess_input(tensor)
    print("Image """ + str(image_path) +
          " "" converted to tensor with shape " + str(tensor.shape))
    return tensor


def getfeatures(folder, target_size, model='Resnet50'):
    model = ResNet50(weights="imagenet", include_top=False)
    images = getFilesInDir(folder)
    images_train, images_test = test_train_split(images, fraction)
    tensors_train = []
    tensors_test = []
    for i, j in images_train.items():
        for k in j:
            tensors_train.append(img_to_tensor(k, target_size=(target_size)))
    for i, j in images_test.items():
        for k in j:
            tensors_test.append(img_to_tensor(k, target_size=(target_size)))
            # print()
    preprocessed_tensors_train = [preprocess_input(i) for i in tensors_train]
    preprocessed_tensors_test = [preprocess_input(i) for i in tensors_test]
    print("Total Training Tensors created:" + str(len(tensors_train)))
    print("Total Testing Tensors created:" + str(len(tensors_test)))
    labels_train = create_labels(images_train, output_classes=21)
    labels_test = create_labels(images_test, output_classes=21)
    print("Total Training Lables created:" + str(len(labels_train)))
    print("Total Testing Lables created:" + str(len(labels_test)))
    features_list_train = []
    features_list_test = []
    features_list_train = [model.predict(x)
                           for x in preprocessed_tensors_train]
    features_list_test = [model.predict(x)
                          for x in preprocessed_tensors_test]
    return (features_list_train, labels_train, features_list_test, labels_test)
    # (1,1,1,2048)


if __name__ == '__main__':

    # (img_to_tensor(getFilesInDir('Dataset')[0], target_size=(224, 224)))

    extracted_features_train, labels_train, extracted_features_test, labels_test = getfeatures(
        folder="Dataset", target_size=(224, 224))
    print("Number of Images used for Extracted Features :" +
          str(len(extracted_features_train)))

    print("Shape of Each Extracted Feature " +
          str(extracted_features_train[0].shape))

    x_train = [preprocess_input_rnn(i, n_time_steps, n_inputs)
               for i in extracted_features_train]
    y_train = labels_train

    x_test = [preprocess_input_rnn(i, n_time_steps, n_inputs)
              for i in extracted_features_test]
    y_test = labels_test

    r1 = RnnClassifier(input_size=(
        n_time_steps, n_inputs))

    # x_train,y_train,

    r1.train_model(input_tensors=x_train, output_tensors=y_train)

    outputs = [i[0] for i in r1.predict(input_tensors=x_test)]
    # print("Predicted Output Labels::" +
    # str(r1.predict(input_tensors=x_train)))

    # print("\nActual Output Labels::" + str([one_hot_to_integer(i)
    # for i in labels]))
    # (one_hot_to_integer(labels[167]))
    print(outputs)
    print([one_hot_to_integer(i) for i in labels_test])
    test_acc = missclassification_rate(
        outputs, [one_hot_to_integer(i) for i in labels_test])

    print("with missclassification score:" + str(test_acc))
    print("with f1 score:" + str(f1_score(outputs,
                                          [one_hot_to_integer(i) for i in labels_test], average=None)))

    r1.parameterPlot(['acc', 'loss'])
