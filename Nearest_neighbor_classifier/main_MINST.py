#!/usr/bin/env python3
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import neighbors, datasets
import random as random
import math


def load_t10k_images():
    mndata = MNIST('/Users/pmh/Desktop/classification_scheme/Attached_files/MNIST')

    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    return images_training, labels_training, images_testing, labels_testing, mndata


def fetch_NNC_training_set(wanted_training_set_by_label, images_training, labels_training):
    traing_list = []

    for i in range(len(images_training)):
        if(wanted_training_set_by_label == labels_training[i]):
            traing_list.append((images_training[i], labels_training[i]))
    return traing_list


def fetch_NNC_test_set(wanted_test_set_by_label, images_test, labels_test):
    test_list = []

    for i in range(len(images_test)):
        if(wanted_test_set_by_label == labels_test[i]):
            test_list.append((images_test[i], labels_test[i]))
    return test_list


def nearest_neighbor_class_centroid(images_training, labels_training, images_testing, labels_testing):
    pca = PCA(n_components=(2))
    training_images_pca = pca.fit_transform(images_training)

    pca = PCA(n_components=(2))
    test_images_pca = pca.fit_transform(images_testing)

    clf = neighbors.KNeighborsClassifier(2, weights='uniform')
    clf.fit(training_images_pca, labels_training)

    return (clf.predict(test_images_pca),
            training_images_pca,
            test_images_pca)


def calculate_success_rate(labels_testing):
    predicted_test_image_labels, training_images_pca, test_images_pca = nearest_neighbor_class_centroid(images_training, labels_training, images_testing, labels_testing)

    counter = 0
    success = 0
    for i,label in enumerate(labels_testing):
        if(label == predicted_test_image_labels[i]):
            success = success + 1
        counter = counter + 1

    total = len(predicted_test_image_labels)
    percentage = (success/total)*100

    print("Total image labels: ", counter)
    print("Succeful matched image labels: ", success)
    print(f"Percentage: {percentage}%")


def plot_data(predicted_images_pca, pca_images_training, pca_images_test):
    
    pca_images_training_X = [pca_images_training[i][0] for i in range(len(pca_images_training))]
    pca_images_training_Y = [pca_images_training[i][1] for i in range(len(pca_images_training))]

    pca_images_test_X = [pca_images_test[i][0] for i in range(len(pca_images_test))]
    pca_images_test_Y = [pca_images_test[i][1] for i in range(len(pca_images_test))]


    plt.figure(figsize=(10,10))

    training_list = []
    for i,elem in enumerate(predicted_images_pca):
            training_list.append((pca_images_training_X[i], pca_images_training_Y[i]))

    plt.scatter([training_list[i][0] for i in range(len(training_list))],
                [training_list[i][1] for i in range(len(training_list))],
                s=80, c="orange", label="Training images")


    test_list = []
    for i,elem in enumerate(predicted_images_pca):
            test_list.append((pca_images_test_X[i], pca_images_test_Y[i]))

    plt.scatter([test_list[i][0] for i in range(len(test_list))],
                [test_list[i][1] for i in range(len(test_list))],
                s=80, marker="v", c="purple", label="Test images")


    # plt.title(f"NSC, k={len(pca_centers)}, training-images={len(pca_images_training)}, test-images={len(pca_images_test)}")
    i = plt.legend()


if __name__ == "__main__": 

    images_training, labels_training, images_testing, labels_testing, mndata = load_t10k_images()

    predicted_images_pca, pca_images_training, pca_images_test = nearest_neighbor_class_centroid(images_training, labels_training, images_testing, labels_testing)
      
    calculate_success_rate(labels_testing)
    plot_data(predicted_images_pca, pca_images_training, pca_images_test)
    plt.show()