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
            training_images_pca,
            test_images_pca)


def plot_data(kmeans_labels, kmeans_test_images_predict, pca_centers, pca_images_training, pca_images_test):

    cluster_centers_X = [pca_centers[i][0] for i in range(len(pca_centers))]
    cluster_centers_Y = [pca_centers[i][1] for i in range(len(pca_centers))]

    pca_images_training_X = [pca_images_training[i][0] for i in range(len(pca_images_training))]
    pca_images_training_Y = [pca_images_training[i][1] for i in range(len(pca_images_training))]

    pca_images_test_X = [pca_images_test[i][0] for i in range(len(pca_images_test))]
    pca_images_test_Y = [pca_images_test[i][1] for i in range(len(pca_images_test))]

    print(len(pca_images_test_X))
    print(len(pca_images_test_Y))

    plt.figure(figsize=(10,10))
    plt.scatter(cluster_centers_X, cluster_centers_Y, s=200, c="green", label="Centroids")
    plt.annotate("Centroid 0",(cluster_centers_X[0],cluster_centers_Y[0]))
    plt.annotate("Centroid 1",(cluster_centers_X[1],cluster_centers_Y[1]))


    training_list_0 = []
    training_list_1 = []
    for i,elem in enumerate(kmeans_labels):
        if(elem == 0):
            training_list_0.append((pca_images_training_X[i], pca_images_training_Y[i]))
        else:
            training_list_1.append((pca_images_training_X[i], pca_images_training_Y[i]))


    plt.scatter([training_list_0[i][0] for i in range(len(training_list_0))],
                [training_list_0[i][1] for i in range(len(training_list_0))],
                s=150, c="blue", label="Training image(s) assigned to label 0")

    plt.scatter([training_list_1[i][0] for i in range(len(training_list_1))],
                [training_list_1[i][1] for i in range(len(training_list_1))],
                s=150, c="orange", label="Training image(s) assigned to label 1")


    test_list_0 = []
    test_list_1 = []
    for i,elem in enumerate(kmeans_test_images_predict):
        if(elem == 1):
            test_list_1.append((pca_images_test_X[i], pca_images_test_Y[i]))
        else:
            test_list_0.append((pca_images_test_X[i], pca_images_test_Y[i]))


    plt.scatter([test_list_0[i][0] for i in range(len(test_list_0))],
                [test_list_0[i][1] for i in range(len(test_list_0))],
                s=150, marker="v", c="purple", label="Test image(s) assigned to label 0")

    plt.scatter([test_list_1[i][0] for i in range(len(test_list_1))],
                [test_list_1[i][1] for i in range(len(test_list_1))],
                s=150, marker="v", c="red", label="Test image(s) assigned to label 1")

    plt.title(f"NSC, k={len(pca_centers)}, training-images={len(pca_images_training)}, test-images={len(pca_images_test)}")
    i = plt.legend()



if __name__ == "__main__": 
    # prerequisites
    file_loader = LoadFiles()
    loaded_images = file_loader.load_ORL_face_data_set_40x30()
    loaded_labels = file_loader.load_ORL_labels()

    kmeans_labels, kmeans_predicted, pca_centers, pca_images_training, test_images_pca = nearest_sub_class_centroid(2,loaded_images,loaded_labels)
    plot_data(kmeans_labels,kmeans_predicted, pca_centers, pca_images_training, test_images_pca)

    kmean_labels3, kmeans_predicted3, pca_centers3, pca_images3_training, test_images_pca3 = nearest_sub_class_centroid(3,loaded_images,loaded_labels)
    plot_data(kmean_labels3, kmeans_predicted3, pca_centers3, pca_images3_training, test_images_pca3)

    kmean_labels5, kmeans_predicted5, pca_centers5, pca_images5_training, test_images_pca5 = nearest_sub_class_centroid(5,loaded_images,loaded_labels)
    plot_data(kmean_labels5, kmeans_predicted5, pca_centers5, pca_images5_training, test_images_pca5)
    plt.show()


    # print(len(fetch_NSC_training_set(40, loaded_images, loaded_labels)))
    # print(fetch_NSC_training_set(40, loaded_images, loaded_labels))

    # print(len(fetch_NSC_test_set(40, loaded_images, loaded_labels)))
    # print(fetch_NSC_test_set(40, loaded_images, loaded_labels))
    