from dataset_prepare import get_notation_file_list, load_notation_file
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt


def clustering_anchor_boxes(n, train_data):
    """
    Create a instance for MiniBatchKMeans (Since our testing dataset has over 2M, the MinibatchKMeans will be better than KMeas.
    :param n: number of clusters
    :param train_data: training data in numpy array
    :return: MiniBatchKMeans instance
    """
    cluster = MiniBatchKMeans(n_clusters=n, init='k-means++', max_iter=100,
                              batch_size=500, verbose=0, compute_labels=True,
                              random_state=None, tol=0.0, max_no_improvement=10,
                              init_size=None, n_init=3, reassignment_ratio=0.01)
    cluster.fit(train_data)
    return cluster

def load_all_anchor_box_size(folder_path):
    """
    Loading all of the anchor box from notation file. Computer the width and height by using the Left, Top, Right, Bottom
    coordinates. Then append all of the width and height into a numpy array as (n_samples, 2) for training.
    :param folder_path: string: Path of the notation files
    :return: numpy array with multiple samples with 2 features.
    """
    anchor_box_w_h = None
    file_list = get_notation_file_list(folder_path)
    for file_name in file_list:
        # print("Processing {} ...".format(file_name))
        content_data_frame = load_notation_file(file_name)
        content_data_frame['width'] = content_data_frame['right'] - content_data_frame['left']
        content_data_frame['height'] = content_data_frame['bottom'] - content_data_frame['top']
        width_array = content_data_frame.width.values
        width_array = np.expand_dims(width_array, axis=1)
        height_array = content_data_frame.height.values
        height_array = np.expand_dims(height_array, axis=1)
        width_height_array = np.hstack((width_array, height_array))
        if anchor_box_w_h is None:
            anchor_box_w_h = np.copy(width_height_array)
        else:
            anchor_box_w_h = np.vstack((anchor_box_w_h, width_height_array))
    return anchor_box_w_h


def avg_iou_distance(box, kmeans_cluster):
    """
    Compute the average distance for the clustering mode
    Sum(d(box),centroid) = 1 - IOU(box, centroid)) / samples_#
    :param box: numpy array of all boxes' width and height
    :param kmeans_cluster: training KMeans model
    :return: float: average distance of all cluster
    """
    centroids = kmeans_cluster.cluster_centers_
    cluster_number = len(centroids)
    cluster_label = kmeans_cluster.labels_
    total_distance = 0
    for i in xrange(len(box)):
        total_distance += 1 - abs((box[i, 0] * box[i, 1] - centroids[cluster_label[i], 0] * centroids[cluster_label[i], 1])) / (box[i, 0] * box[i, 1] + centroids[cluster_label[i], 0] * centroids[cluster_label[i], 1])
    return total_distance / len(box)

def draw_avg_iou_plot(avg_iou_list):
    plt.figure(1)
    plt.xlabel('# of Clusters')
    plt.ylabel('Avg IOU')
    plt.title('Avg IOU based on different # of Clusters')
    plt.grid(True)
    plt.plot([x for x in xrange(1, len(avg_iou_list) + 1)], avg_iou_list, 'bo',
             [x for x in xrange(1, len(avg_iou_list) + 1)], avg_iou_list, 'k')
    plt.show()


if __name__ == "__main__":
    bbox_WH = load_all_anchor_box_size("./datasets/vgg_face_dataset/files/")
    print("There are {} anchor boxes found!".format(len(bbox_WH)))
    avg_iou_dist_list = []
    for n_of_cluster in xrange (1, 16):
        cluster = clustering_anchor_boxes(n_of_cluster, bbox_WH)
        print("Centroid of {} clusters: ".format(n_of_cluster))
        print(cluster.cluster_centers_)
        avg_iou_dist = avg_iou_distance(bbox_WH, cluster)
        avg_iou_dist_list.append(avg_iou_dist)
        print("Average IOU Distance for K-Means #{} clusters is {}".format(n_of_cluster, avg_iou_dist))

    draw_avg_iou_plot(avg_iou_dist_list)
    # Per displayed in the picture of "AvgIOU_analyze.png", I plan to choose '7' as the number of cluster for KMeans



