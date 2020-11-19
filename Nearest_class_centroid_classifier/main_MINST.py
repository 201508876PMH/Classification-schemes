#!/usr/bin/env python3
from mnist import MNIST
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_t10k_images():
    mndata = MNIST('/Users/pmh/Desktop/classification_scheme/Attached_files/MNIST')

    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()

    return images_training, labels_training, images_testing, labels_testing, mndata


def diplay_image(mnist, image):
    print(mnist.display(image))


def nearest_class_centroid(images_training, labels_training, images_testing, labels_testing):
    clf = NearestCentroid()
    centroids = clf.fit(images_training, labels_training)
    print("Centroids: ", centroids.centroids_)

    pca = PCA(n_components=(2))
    training_images_pca = pca.fit_transform(images_training)
    pca.fit_transform(training_images_pca)

    pca = PCA(n_components=(2))
    test_images_pca = pca.fit_transform(images_testing)

    pca = PCA(n_components=(2))
    pca_centroids = pca.fit_transform(centroids.centroids_)
    print(pca_centroids)

    # kmeans = KMeans(n_clusters=(784), random_state=0).fit(training_images_pca)
    # print("LOLOL", kmeans.labels_)
    # kmeans.predict(test_images_pca)
    # kmeans.cluster_centers_

    return (clf.predict(images_testing),
            # kmeans.labels_,
            # kmeans.predict(test_images_pca),
            pca_centroids,
            training_images_pca,
            test_images_pca)



def calculate_success_rate(images_training, labels_training, images_testing, labels_testing):
    predict_image_testing, centroids, training_images_pca, test_images_pca = nearest_class_centroid(images_training, labels_training, images_testing, labels_testing)
    
    counter = 0
    success = 0
    for label in labels_testing:
        if(label == predict_image_testing[counter]):
            success = success + 1
        counter = counter + 1

    percentage = (success/counter)*100

    print("Total image labels: ", counter)
    print("Succeful matched image labels: ", success)
    print(f"Percentage: {percentage}%")


def plot_data(kmeans_labels, predict_image_testing, pca_centroids, training_images_pca, test_images_pca):

    cluster_centers_X = [pca_centroids[i][0] for i in range(len(pca_centroids))]
    cluster_centers_Y = [pca_centroids[i][1] for i in range(len(pca_centroids))]

    pca_images_training_X = [training_images_pca[i][0] for i in range(len(training_images_pca))]
    pca_images_training_Y = [training_images_pca[i][1] for i in range(len(training_images_pca))]

    pca_images_test_X = [test_images_pca[i][0] for i in range(len(test_images_pca))]
    pca_images_test_Y = [test_images_pca[i][1] for i in range(len(test_images_pca))]

    plt.figure(figsize=(10,10))
    plt.scatter(cluster_centers_X, cluster_centers_Y, s=500, c="green", label="Centroids")

    training_list = []
    for i,elem in enumerate(kmeans_labels):
            training_list.append((pca_images_training_X[i], pca_images_training_Y[i]))

    plt.scatter([training_list[i][0] for i in range(len(training_list))],
                [training_list[i][1] for i in range(len(training_list))],
                s=80, c="orange", label="Training images")


    test_list = []
    for i,elem in enumerate(predict_image_testing):
            test_list.append((pca_images_test_X[i], pca_images_test_Y[i]))

    plt.scatter([test_list[i][0] for i in range(len(test_list))],
                [test_list[i][1] for i in range(len(test_list))],
                s=80, marker="v", c="purple", label="Test images")


    # plt.title(f"NSC, k={len(pca_centers)}, training-images={len(pca_images_training)}, test-images={len(pca_images_test)}")
    i = plt.legend()

if __name__ == "__main__": 

    images_training, labels_training, images_testing, labels_testing, mndata = load_t10k_images()

    # diplay_image(mndata, images_training[1])

    predict_image_testing, pca_centroids, training_images_pca, test_images_pca = nearest_class_centroid(images_training, labels_training,images_testing, labels_testing)
    print("TIS")
    
    calculate_success_rate(images_training, labels_training, images_testing, labels_testing)
    plot_data(labels_training,predict_image_testing, pca_centroids, training_images_pca, test_images_pca)
    plt.show()

    