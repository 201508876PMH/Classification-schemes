#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append("/Users/pmh/Desktop/classification_scheme/Prerequisites") 
from LoadFiles import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


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
            training_images_pca)

def plot_data(kmeans_labels, pca_centers, pca_images):

    cluster_centers_X = [pca_centers[i][0] for i in range(len(pca_centers))]
    cluster_centers_Y = [pca_centers[i][1] for i in range(len(pca_centers))]

    pca_images_X = [pca_images[i][0] for i in range(len(pca_images))]
    pca_images_Y = [pca_images[i][1] for i in range(len(pca_images))]


    plt.figure(figsize=(8,8))
    plt.scatter(cluster_centers_X, cluster_centers_Y, s=150, c="green")
    plt.annotate("  One of two Centroids",(cluster_centers_X[0],cluster_centers_Y[0]))

    for i,elem in enumerate(kmeans_labels):
        if(elem == 1):
            plt.scatter(pca_images_X[i], pca_images_Y[i], s=50, c="orange")
        else:
            plt.scatter(pca_images_X[i], pca_images_Y[i], s=50, c="blue")

    plt.annotate("  One of seven PCA training images",(pca_images_X[0],pca_images_Y[0]))
    plt.title(f"NSC, k={len(pca_centers)}, test-images={len(pca_images)}")



if __name__ == "__main__": 
    # prerequisites
    file_loader = LoadFiles()
    loaded_images = file_loader.load_ORL_face_data_set_40x30()
    loaded_labels = file_loader.load_ORL_labels()

    kmeans_labels, kmeans_predicted, pca_centers, pca_images = nearest_sub_class_centroid(2,loaded_images,loaded_labels)
    plot_data(kmeans_labels, pca_centers, pca_images)

    kmean_labels3, kmeans_predicted3, pca_centers3, pca_images3 = nearest_sub_class_centroid(3,loaded_images,loaded_labels)
    plot_data(kmean_labels3, pca_centers3, pca_images3)

    kmean_labels5, kmeans_predicted5, pca_centers5, pca_images5 = nearest_sub_class_centroid(5,loaded_images,loaded_labels)
    plot_data(kmean_labels5, pca_centers5, pca_images5)
    plt.show()

    # print(len(fetch_NSC_training_set(40, loaded_images, loaded_labels)))
    # print(fetch_NSC_training_set(40, loaded_images, loaded_labels))

    # print(len(fetch_NSC_test_set(40, loaded_images, loaded_labels)))
    # print(fetch_NSC_test_set(40, loaded_images, loaded_labels))
    