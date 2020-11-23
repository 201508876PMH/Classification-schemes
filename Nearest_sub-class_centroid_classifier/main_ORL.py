#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append("/Users/pmh/Desktop/classification_scheme/Prerequisites") 
from LoadFiles import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import random as random

def fetch_label_by_image_id(image_id, loaded_labels):
    return loaded_labels[image_id]


def fetch_specific_image_in_binary(imageNumber, loaded_images):
    matrix = loaded_images
    return matrix[:,imageNumber]


def fetch_NSC_training_set(wanted_training_set_by_label,loaded_images, loaded_labels):
    traing_list = []

    counter = 0
    x = 0
    total_images = 400
    while(x < total_images):
        if(counter == 7):
            x += 2
            counter = 0
        else:
            label_x = fetch_label_by_image_id(x, loaded_labels)
            if(label_x == str(wanted_training_set_by_label)):
                image_x = fetch_specific_image_in_binary(x, loaded_images)
                traing_list.append((image_x, label_x))
            counter = counter + 1    
        x += 1
    return traing_list


def fetch_NSC_test_set(wanted_test_set_by_label, loaded_images, loaded_labels):
    test_list = []

    counter = 0
    x = 0
    total_images = 400
    while(x < total_images):
        if(counter < 7):
            counter = counter + 1 
        elif(counter == 10):
            counter = 1
        else:
            label_x = fetch_label_by_image_id(x, loaded_labels)
            if(label_x == str(wanted_test_set_by_label)):
                image_x = fetch_specific_image_in_binary(x, loaded_images)
                test_list.append((image_x, label_x))
            counter = counter + 1    
        x += 1
    return test_list


def nearest_sub_class_centroid(wanted_training_set_by_label, loaded_images, loaded_labels):
    training_data = fetch_NSC_training_set(wanted_training_set_by_label, loaded_images, loaded_labels)
    training_images = [training_data[i][0] for i in range(len(training_data))]

    test_data = fetch_NSC_test_set(wanted_training_set_by_label, loaded_images, loaded_labels)
    test_images = [test_data[i][0] for i in range(len(test_data))]

    pca = PCA(n_components=2)
    training_images_pca = pca.fit_transform(training_images)
    # print("training images pca transformed: \n", pca.fit_transform(training_images_pca))

    pca = PCA(n_components=2)
    test_images_pca = pca.fit_transform(test_images)
    # print("test images pca transformed: \n", pca.fit_transform(test_images_pca))

    kmeans = KMeans(n_clusters=3, random_state=0).fit(training_images_pca)
    # print("KMeans labels: \n", kmeans.labels_)
    # print("KMeans predicted test images placement: \n",kmeans.predict(test_images_pca))
    # print("KMeans cluster centers: \n",kmeans.cluster_centers_)


    return (kmeans.labels_,
            kmeans.predict(test_images_pca),
            kmeans.cluster_centers_,
            training_images_pca,
            test_images_pca)


def plot_data(kmeans_labels, kmeans_test_images_predict, pca_centers, pca_images_training, pca_images_test):

    pca_images_test_X = [pca_images_test[i][0] for i in range(len(pca_images_test))]
    pca_images_test_Y = [pca_images_test[i][1] for i in range(len(pca_images_test))]

    plt.figure(figsize=(10,10))
    number_of_colors = len(pca_centers)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
     
    test_list_0 = []
    test_list_1 = []
    test_list_2 = []
    for i,elem in enumerate(kmeans_test_images_predict):
        if(elem == 0):
            test_list_0.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 1):
            test_list_1.append((pca_images_test_X[i], pca_images_test_Y[i]))
        else:
            test_list_2.append((pca_images_test_X[i], pca_images_test_Y[i]))


    plt.scatter([test_list_0[i][0] for i in range(len(test_list_0))],
                [test_list_0[i][1] for i in range(len(test_list_0))],
                s=100, c=color[0], alpha=0.4, label="Test image(s) assigned to label 0")

    plt.scatter([test_list_1[i][0] for i in range(len(test_list_1))],
                [test_list_1[i][1] for i in range(len(test_list_1))],
                s=100, c=color[1], alpha=0.4, label="Test image(s) assigned to label 1")

    plt.scatter([test_list_2[i][0] for i in range(len(test_list_2))],
                [test_list_2[i][1] for i in range(len(test_list_2))],
                s=100, c=color[2], alpha=0.4, label="Test image(s) assigned to label 2")

    for i,centroid in enumerate(pca_centers):
            plt.scatter(pca_centers[i][0], pca_centers[i][1], s=250, c="black")
            plt.scatter(pca_centers[i][0], pca_centers[i][1], s=200, c=color[i], label=f"Centroid {i}")

    plt.title(f"NSC, k={len(pca_centers)}, training-images={len(pca_images_training)}, test-images={len(pca_images_test)}")
    i = plt.legend()

# Code from: https://predictivehacks.com/k-means-elbow-method-code-for-python/
def plot_elbow_graph(elbow_data):

    pca = PCA(n_components=2)
    training_images_pca = pca.fit_transform(elbow_data)
    distortions = []
    K = range(1,8)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(training_images_pca)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,7))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def calculate_success_rate(kmeans_predicted_test_images):
    
    sub_class_0 = 0
    sub_class_1 = 0
    sub_class_2 = 0

    for elem in kmeans_predicted_test_images:
        if(elem == 0):
            sub_class_0 = sub_class_0 + 1
        elif(elem == 1):
            sub_class_1 = sub_class_1 + 1
        else:
            sub_class_2 = sub_class_2 + 1

    total = len(kmeans_predicted_test_images)

    print("Total image labels: ", total)
    print("Probability for class 0: ", (sub_class_0/total)*100)
    print("Probability for class 1: ", (sub_class_1/total)*100)
    print("Probability for class 2: ", (sub_class_2/total)*100)


if __name__ == "__main__": 
    # prerequisites
    file_loader = LoadFiles()
    loaded_images = file_loader.load_ORL_face_data_set_40x30()
    loaded_labels = file_loader.load_ORL_labels()

    kmeans_labels, kmeans_predicted, pca_centers, pca_images_training, test_images_pca = nearest_sub_class_centroid(5,loaded_images,loaded_labels)
    calculate_success_rate(kmeans_predicted)
    plot_data(kmeans_labels,kmeans_predicted, pca_centers, pca_images_training, test_images_pca)

    training_images = fetch_NSC_training_set(3,loaded_images, loaded_labels)
    elbow_data = [training_images[i][0] for i in range(len(training_images))]
    plot_elbow_graph(elbow_data)

    kmean_labels3, kmeans_predicted3, pca_centers3, pca_images3_training, test_images_pca3 = nearest_sub_class_centroid(3,loaded_images,loaded_labels)
    plot_data(kmean_labels3, kmeans_predicted3, pca_centers3, pca_images3_training, test_images_pca3)

    kmean_labels5, kmeans_predicted5, pca_centers5, pca_images5_training, test_images_pca5 = nearest_sub_class_centroid(5,loaded_images,loaded_labels)
    plot_data(kmean_labels5, kmeans_predicted5, pca_centers5, pca_images5_training, test_images_pca5)
    plt.show()

    