#!/usr/bin/env python3
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random as random
import math


def load_t10k_images():
    mndata = MNIST('/Users/pmh/Desktop/classification_scheme/Attached_files/MNIST')

    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    return images_training, labels_training, images_testing, labels_testing, mndata


def fetch_NSC_training_set(wanted_training_set_by_label, images_training, labels_training):
    traing_list = []

    for i in range(len(images_training)):
        if(wanted_training_set_by_label == labels_training[i]):
            traing_list.append((images_training[i], labels_training[i]))
    return traing_list


def fetch_NSC_test_set(wanted_test_set_by_label, images_test, labels_test):
    test_list = []

    for i in range(len(images_test)):
        if(wanted_test_set_by_label == labels_test[i]):
            test_list.append((images_test[i], labels_test[i]))
    return test_list


def nearest_sub_class_centroid(wanted_training_set_by_label, images_training, labels_training, images_testing, labels_testing):
    training_data = fetch_NSC_training_set(wanted_training_set_by_label, images_training, labels_training)
    training_images = [training_data[i][0] for i in range(len(training_data))]

    test_data = fetch_NSC_test_set(wanted_training_set_by_label, images_testing, labels_testing)
    test_images = [test_data[i][0] for i in range(len(test_data))]
    chosen_label = wanted_training_set_by_label
    
    pca = PCA(n_components=2)
    training_images_pca = pca.fit_transform(training_images)
    print("training images pca transformed: \n", pca.fit_transform(training_images_pca))

    pca = PCA(n_components=2)
    test_images_pca = pca.fit_transform(test_images)
    print("test images pca transformed: \n", pca.fit_transform(test_images_pca))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(training_images_pca)
    print("KMeans labels: \n", kmeans.labels_)
    print("KMeans predicted test images placement: \n",kmeans.predict(test_images_pca))
    print("KMeans cluster centers: \n",kmeans.cluster_centers_)


    return (kmeans.labels_,
            kmeans.predict(test_images_pca),
            kmeans.cluster_centers_,
            training_images_pca,
            test_images_pca,
            chosen_label)


def plot_data(chosen_label, kmeans_test_images_predict, pca_centroids, pca_images_training, pca_images_test):

    pca_images_test_X = [pca_images_test[i][0] for i in range(len(pca_images_test))]
    pca_images_test_Y = [pca_images_test[i][1] for i in range(len(pca_images_test))]

    print(len(pca_images_test_X))
    print(len(pca_images_test_Y))

    plt.figure(figsize=(10,10))
    number_of_colors = len(pca_centroids)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    

    test_list_0 = []
    test_list_1 = []
    for i,elem in enumerate(kmeans_test_images_predict):
        if(elem == 0):
            test_list_0.append((pca_images_test_X[i], pca_images_test_Y[i]))
        else:
            test_list_1.append((pca_images_test_X[i], pca_images_test_Y[i]))


    plt.scatter([test_list_0[i][0] for i in range(len(test_list_0))],
                [test_list_0[i][1] for i in range(len(test_list_0))],
                s=100, c=color[0], alpha=0.4, label="Test image(s) assigned to label 0")

    plt.scatter([test_list_1[i][0] for i in range(len(test_list_1))],
                [test_list_1[i][1] for i in range(len(test_list_1))],
                s=100, c=color[1], alpha=0.4, label="Test image(s) assigned to label 1")


    for i,centroid in enumerate(pca_centroids):
        plt.scatter(pca_centroids[i][0], pca_centroids[i][1], s=250, c="black")
        plt.scatter(pca_centroids[i][0], pca_centroids[i][1], s=200, c=color[i], label=f"Centroid {i}")

    plt.title(f"NSC, k={len(pca_centroids)}, number of subclasses in set {chosen_label} = {len(pca_images_test)}")
    i = plt.legend()

if __name__ == "__main__": 

    images_training, labels_training, images_testing, labels_testing, mndata = load_t10k_images()

    kmeans_labels, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca, chosen_label = nearest_sub_class_centroid(2, images_training, labels_training, images_testing, labels_testing)
    plot_data(chosen_label, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca)

    kmeans_labels, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca, chosen_label = nearest_sub_class_centroid(3, images_training, labels_training, images_testing, labels_testing)
    plot_data(chosen_label, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca)

    kmeans_labels, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca, chosen_label = nearest_sub_class_centroid(5, images_training, labels_training, images_testing, labels_testing)
    plot_data(chosen_label, kmeans_predicted_test_images, pca_centroids, training_images_pca, test_images_pca)
    plt.show()