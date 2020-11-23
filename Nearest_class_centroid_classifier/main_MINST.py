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


def diplay_image(mnist, image):
    print(mnist.display(image))


def nearest_class_centroid(images_training, labels_training, images_testing, labels_testing):

    pca = PCA(n_components=(2))
    training_images_pca = pca.fit_transform(images_training)
    test_images_pca = pca.fit_transform(images_testing)
   
    clf = NearestCentroid()
    clf.fit(training_images_pca, labels_training)
    print("Centoids: \n", clf.centroids_)

    return (clf.predict(test_images_pca),
            clf.centroids_,
            training_images_pca,
            test_images_pca)


def calculate_success_rate(images_training, labels_training, images_testing, labels_testing):
    predict_image_testing, centroids, training_images_pca, test_images_pca = nearest_class_centroid(images_training, labels_training, images_testing, labels_testing)
    
    print("Predicted testing labels: \n", (len(predict_image_testing)))
    print("Test labels: \n", labels_testing)

    counter = 0
    success = 0
    for label in labels_testing:
        if(label == predict_image_testing[counter]):
            success = success + 1
        counter = counter + 1

    percentage = (success/counter)*100

    print("Total image labels: ", counter)
    print("Successfully matched image labels: ", success)
    print(f"Percentage: {percentage}%")


def plot_data(kmeans_labels, predicted_test_image_labels, pca_centroids, training_images_pca, test_images_pca):

    pca_images_test_X = [test_images_pca[i][0] for i in range(len(test_images_pca))]
    pca_images_test_Y = [test_images_pca[i][1] for i in range(len(test_images_pca))]

    plt.figure(figsize=(10,10))

    number_of_colors = 10
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

    test_list_0 = []
    test_list_1 = []
    test_list_2 = []
    test_list_3 = []
    test_list_4 = []
    test_list_5 = []
    test_list_6 = []
    test_list_7 = []
    test_list_8 = []
    test_list_9 = []

    for i,elem in enumerate(predicted_test_image_labels):
        if(elem == 0):
            test_list_0.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 1):
            test_list_1.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 2):
            test_list_2.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 3):
            test_list_3.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 4):
            test_list_4.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 5):
            test_list_5.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 6):
            test_list_6.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 7):
            test_list_7.append((pca_images_test_X[i], pca_images_test_Y[i]))
        elif(elem == 8):
            test_list_8.append((pca_images_test_X[i], pca_images_test_Y[i]))
        else:
            test_list_9.append((pca_images_test_X[i], pca_images_test_Y[i]))

    # print("list_0: ", test_list_0[1])
    plt.scatter([test_list_0[i][0] for i in range(len(test_list_0))],[test_list_0[i][1] for i in range(len(test_list_0))], s=30, c=color[0], label="Test images for label 0", alpha=0.4)
    plt.scatter([test_list_1[i][0] for i in range(len(test_list_1))],[test_list_1[i][1] for i in range(len(test_list_1))], s=30, c=color[1], label="Test images for label 1", alpha=0.4)
    plt.scatter([test_list_2[i][0] for i in range(len(test_list_2))],[test_list_2[i][1] for i in range(len(test_list_2))], s=30, c=color[2], label="Test images for label 2", alpha=0.4)
    plt.scatter([test_list_3[i][0] for i in range(len(test_list_3))],[test_list_3[i][1] for i in range(len(test_list_3))], s=30, c=color[3], label="Test images for label 3", alpha=0.4)
    plt.scatter([test_list_4[i][0] for i in range(len(test_list_4))],[test_list_4[i][1] for i in range(len(test_list_4))], s=30, c=color[4], label="Test images for label 4", alpha=0.4)
    plt.scatter([test_list_5[i][0] for i in range(len(test_list_5))],[test_list_5[i][1] for i in range(len(test_list_5))], s=30, c=color[5], label="Test images for label 5", alpha=0.4)
    plt.scatter([test_list_6[i][0] for i in range(len(test_list_6))],[test_list_6[i][1] for i in range(len(test_list_6))], s=30, c=color[6], label="Test images for label 6", alpha=0.4)
    plt.scatter([test_list_7[i][0] for i in range(len(test_list_7))],[test_list_7[i][1] for i in range(len(test_list_7))], s=30, c=color[7], label="Test images for label 7", alpha=0.4)
    plt.scatter([test_list_8[i][0] for i in range(len(test_list_8))],[test_list_8[i][1] for i in range(len(test_list_8))], s=30, c=color[8], label="Test images for label 8", alpha=0.4)
    plt.scatter([test_list_9[i][0] for i in range(len(test_list_9))],[test_list_9[i][1] for i in range(len(test_list_9))], s=30, c=color[9], label="Test images for label 9", alpha=0.4)  
   
    for i,centroid in enumerate(pca_centroids):
        plt.scatter(pca_centroids[i][0], pca_centroids[i][1], s=230, c="black")
        plt.scatter(pca_centroids[i][0], pca_centroids[i][1], s=180, c=color[i], label=f"Centroid {i}")

    plt.title(f"NSC, k={len(pca_centroids)}, training-images={len(training_images_pca)}, test-images={len(pca_images_test_X)}")
    i = plt.legend()


if __name__ == "__main__": 
    # Load data
    images_training, labels_training, images_testing, labels_testing, mndata = load_t10k_images()
    
    # Show one of images loaded
    diplay_image(mndata, images_training[1])
    
    # Train algorithm and apply test data
    data = nearest_class_centroid(images_training, labels_training,images_testing, labels_testing)
      
    # Calculate success rate
    calculate_success_rate(images_training, labels_training, images_testing, labels_testing)

    # Plot results
    plot_data(labels_training, data[0], data[1], data[2], data[3])
    plt.show()

    