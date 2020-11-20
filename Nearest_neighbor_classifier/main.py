#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append("/Users/pmh/Desktop/classification_scheme/Prerequisites") 
from LoadFiles import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn import neighbors, datasets


def fetch_label_by_image_id(image_id, loaded_labels):
    return loaded_labels[image_id]


def fetch_specific_image_in_binary(imageNumber, loaded_images):
    matrix = loaded_images
    return matrix[:,imageNumber]


def fetch_training_set(loaded_images, loaded_labels):
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
            image_x = fetch_specific_image_in_binary(x, loaded_images)
            traing_list.append((image_x, label_x))
            counter = counter + 1    
        x += 1
    return traing_list


def fetch_testing_set(loaded_images, loaded_labels):
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
            image_x = fetch_specific_image_in_binary(x, loaded_images)
            test_list.append((image_x, label_x))
            counter = counter + 1    
        x += 1
    return test_list


def nearest_neighbor_class_centroid(loaded_images, loaded_labels):
    training_data = fetch_training_set(loaded_images, loaded_labels)
    training_images = [training_data[i][0] for i in range(len(training_data))]
    training_labels = [training_data[i][1] for i in range(len(training_data))]

    test_data = fetch_testing_set(loaded_images, loaded_labels)
    test_images = [test_data[i][0] for i in range(len(test_data))]

    pca = PCA(n_components=(2))
    training_images_pca = pca.fit_transform(training_images)

    pca = PCA(n_components=(2))
    test_images_pca = pca.fit_transform(test_images)

    kmeans = KMeans(n_clusters=(280), random_state=0).fit(training_images_pca)
    
    # From library, but same as above
    clf = neighbors.KNeighborsClassifier(1, weights='uniform')
    clf.fit(training_images_pca, training_labels)

    # print("KMeans labels: \n", kmeans.labels_)
    # print("KMeans predicted test images placement: \n",kmeans.predict(test_images_pca))
    # print("KMeans cluster centers: \n",kmeans.cluster_centers_)


    return (kmeans.labels_,
            kmeans.predict(test_images_pca),
            kmeans.cluster_centers_,
            training_images_pca,
            test_images_pca,
            clf.predict(test_images_pca))


def calculate_success_rate(loaded_images, loaded_labels, predicted_images_pca):
    means_labels, kmeans_predicted, pca_centers, pca_images_training, test_images_pca, predicted_images_pca = nearest_neighbor_class_centroid(loaded_images, loaded_labels)
    print(kmeans_predicted)
    print(predicted_images_pca)


    test_data = fetch_training_set(loaded_images, loaded_labels)
    test_labels = [test_data[i][1] for i in range(len(test_data))]

    training_data = fetch_training_set(loaded_images, loaded_labels)
    training_labels = [training_data[i][1] for i in range(len(training_data))]

    counter = 0
    success = 0
    for i,label in enumerate(kmeans_predicted):
        if(test_labels[i] == training_labels[label]):
            success = success + 1
        counter = counter + 1

    total = len(kmeans_predicted)
    percentage = (success/total)*100

    print("Total image labels: ", counter)
    print("Succeful matched image labels: ", success)
    print(f"Percentage: {percentage}%")


    counter_library = 0
    success_library = 0

    for i,label in enumerate(predicted_images_pca):
        if(label == test_labels[i]):
            success_library = success_library + 1
        counter_library = counter_library + 1

    total = counter_library
    percentage = (success_library/total)*100

    print("Total image labels: ", counter_library)
    print("Succeful matched image labels: ", success_library)
    print(f"Percentage: {percentage}%")



def plot_data(kmeans_labels, kmeans_test_images_predict, pca_centers, pca_images_training, pca_images_test):

    cluster_centers_X = [pca_centers[i][0] for i in range(len(pca_centers))]
    cluster_centers_Y = [pca_centers[i][1] for i in range(len(pca_centers))]

    pca_images_training_X = [pca_images_training[i][0] for i in range(len(pca_images_training))]
    pca_images_training_Y = [pca_images_training[i][1] for i in range(len(pca_images_training))]

    pca_images_test_X = [pca_images_test[i][0] for i in range(len(pca_images_test))]
    pca_images_test_Y = [pca_images_test[i][1] for i in range(len(pca_images_test))]

    plt.figure(figsize=(10,10))
    plt.scatter(cluster_centers_X, cluster_centers_Y, s=200, c="green", label="Centroids")

    training_list = []
    for i,elem in enumerate(kmeans_labels):
            training_list.append((pca_images_training_X[i], pca_images_training_Y[i]))

    plt.scatter([training_list[i][0] for i in range(len(training_list))],
                [training_list[i][1] for i in range(len(training_list))],
                s=80, c="orange", label="Training images")


    test_list = []
    for i,elem in enumerate(kmeans_test_images_predict):
            test_list.append((pca_images_test_X[i], pca_images_test_Y[i]))

    plt.scatter([test_list[i][0] for i in range(len(test_list))],
                [test_list[i][1] for i in range(len(test_list))],
                s=80, marker="v", c="purple", label="Test images")


    plt.title(f"NSC, k={len(pca_centers)}, training-images={len(pca_images_training)}, test-images={len(pca_images_test)}")
    i = plt.legend()



if __name__ == "__main__": 
    # prerequisites
    file_loader = LoadFiles()
    loaded_images = file_loader.load_ORL_face_data_set_40x30()
    loaded_labels = file_loader.load_ORL_labels()

    kmeans_labels, kmeans_predicted, pca_centers, pca_images_training, test_images_pca, predicted_images_pca = nearest_neighbor_class_centroid(loaded_images,loaded_labels)
    calculate_success_rate(loaded_images, loaded_labels, predicted_images_pca)
    plot_data(kmeans_labels,kmeans_predicted, pca_centers, pca_images_training, test_images_pca)
    plt.show()

    