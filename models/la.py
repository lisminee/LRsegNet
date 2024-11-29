import numpy as np

def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)
    sortedDistIndices = np.argsort(squaredDist)

    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k]


def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    Q, R = np.linalg.qr(MatX)
    feature_matrix = Q
    num_samples = feature_matrix.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)

    if kernel_type == 'rbf':
        if rbf_sigma is None:
            raise ValueError('You should input a sigma for the RBF kernel!')
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = feature_matrix[i, :] - feature_matrix[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors is None:
            raise ValueError('You should input a k for the KNN kernel!')
        for i in range(num_samples):
            k_neighbors = navie_knn(feature_matrix, feature_matrix[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError('Not supported kernel type! You can use knn or rbf!')

    return affinity_matrix


def Soft_Low_Rank_Approximation(s, b, t, tier, points, labels, num_classes, kernel_type='rbf', rbf_sigma=1.5, \
                                  knn_num_neighbors=8, max_iter=500, tol=1e-3):
    classified_index = np.where(labels != 0)[0]
    unclassified_index = np.where(labels == 0)[0]

    num_classified_samples = len(classified_index)
    num_unclassified_samples = len(unclassified_index)
    num_samples = num_classified_samples + num_unclassified_samples

    clamp_data_label = np.zeros((num_classified_samples, num_classes), np.float32)
    for i in range(num_classified_samples):
        clamp_data_label[i][int(labels[classified_index[i]] - 1)] = 1.0

    approximation_matrix = np.zeros((num_samples, num_classes), np.float32)
    approximation_matrix[classified_index] = clamp_data_label
    approximation_matrix[unclassified_index] = -1

    affinity_matrix = buildGraph(points, kernel_type, rbf_sigma, knn_num_neighbors)

    iter = 0
    previous_approximation_matrix = np.zeros((num_samples, num_classes), np.float32)
    change = np.abs(previous_approximation_matrix - approximation_matrix).sum()

    confident_labels_history = []
    confidence_threshold = 0.1

    while iter < max_iter:
        if iter % 1 == 0:
            confident_labels = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_samples):
                if approximation_matrix[i].max() > confidence_threshold:
                    confident_labels[i] = float(np.argmax(approximation_matrix[i]) + 1)

        filename = f"{s}_{b}_{tier}:confident_labels_iter_{iter:04}.txt"
        np.savetxt(filename, confident_labels, fmt='%.1f', newline="\n")  

        previous_approximation_matrix = approximation_matrix
        iter += 1

        approximation_matrix = np.dot(affinity_matrix, approximation_matrix)

        approximation_matrix[classified_index] = clamp_data_label

        change = np.abs(previous_approximation_matrix - approximation_matrix).sum()

    all_data_labels = np.zeros(num_samples)
    np.savetxt("confident_labels_history.txt", confident_labels_history, fmt='%d', delimiter=",")
    for i in range(num_samples):
        all_data_labels[i] = np.argmax(approximation_matrix[i]) + 1

    return all_data_labels
