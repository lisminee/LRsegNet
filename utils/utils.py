
import numpy as np

import open3d as o3d


def visualize_point_cloud(points, label_function, confidence_threshold):

    colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        max_confidence = label_function[i].max()
        if max_confidence < confidence_threshold:
            colors[i] = [0, 0, 0] 
        else:
            color = np.argmax(label_function[i]) + 1
            if color == 1:
                colors[i] = [1, 0, 0]
            elif color == 2:
                colors[i] = [0, 1, 0]
            elif color == 3:
                colors[i] = [0, 0, 1]
            elif color == 4:
                colors[i] = [1, 1, 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name='Open3D', width=800, height=600)


def sample_label(label, expt, m):
    unique_labels = np.unique(label)
    new_label = np.zeros_like(label)

    if not isinstance(expt, (list, np.ndarray)):
        expt = [expt]

    for lbl in unique_labels:
        if lbl not in expt:
            lbl_indices = np.where(label == lbl)[0]
            if len(lbl_indices) < m:
                raise ValueError(f"Label {lbl} has less than {m} samples")
            selected_indices = np.random.choice(lbl_indices, m, replace=False)
            new_label[selected_indices] = lbl

    return new_label


def sample_top_k_labels(label, k, m):
    unique_labels, counts = np.unique(label, return_counts=True)

    top_k_labels = unique_labels[np.argsort(-counts)[:k]]

    new_label = np.zeros_like(label)

    for lbl in top_k_labels:
        lbl_indices = np.where(label == lbl)[0]
        num_samples = min(len(lbl_indices), m)
        selected_indices = np.random.choice(lbl_indices, num_samples, replace=False)
        new_label[selected_indices] = lbl

    return new_label



def sample_k_classes(label, k, m):
    unique_labels = np.unique(label)

    selected_labels = np.random.choice(unique_labels, k, replace=True)

    new_label = np.zeros_like(label)

    for lbl in selected_labels:
        lbl_indices = np.where(label == lbl)[0]

        num_samples = min(len(lbl_indices), m)
        selected_indices = np.random.choice(lbl_indices, num_samples, replace=False)
        new_label[selected_indices] = lbl

    return new_label


def calculate_iou(predicted, y_all, n_classes):

    true_positives = np.zeros(n_classes)
    false_positives = np.zeros(n_classes)
    false_negatives = np.zeros(n_classes)

    for i in range(n_classes):
        true_positives = np.sum((predicted+1 == i+1) & (y_all == i+1))
        false_positives = np.sum((predicted+1 == i+1) & (y_all != i+1))
        false_negatives = np.sum((predicted+1 != i+1) & (y_all == i+1))

    iou = true_positives / (true_positives + false_positives + false_negatives)

    return iou
def calculate_mean_iou(predicted, true_labels, n_classes):
    true_positives = np.zeros(n_classes)
    false_positives = np.zeros(n_classes)
    false_negatives = np.zeros(n_classes)

    for i in range(1,n_classes + 1):
        true_positives[i-1] = np.sum((predicted == i) & (true_labels == i))
        false_positives[i-1] = np.sum((predicted == i) & (true_labels != i))
        false_negatives[i-1] = np.sum((predicted != i) & (true_labels == i))

    iou = true_positives / (true_positives + false_positives + false_negatives + 1e-6)
    mean_iou = np.nanmean(iou)
    return mean_iou

def calculate_classwise_mean_iou(predicted, true_labels, classes):
    classwise_iou = []
    for cls in classes:
        if cls not in predicted and cls not in true_labels:
            continue  
        true_positives = np.sum((predicted == cls) & (true_labels == cls))
        false_positives = np.sum((predicted == cls) & (true_labels != cls))
        false_negatives = np.sum((predicted != cls) & (true_labels == cls))

        iou = true_positives / (true_positives + false_positives + false_negatives + 1e-6)
        classwise_iou.append(iou)

    classwise_iou = np.nan_to_num(classwise_iou)


    if np.all(classwise_iou == 0):
        return 0 

    mean_iou = np.mean(classwise_iou)
    return mean_iou
