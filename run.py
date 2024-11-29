import os
import random
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models.la import *
from models.mlp import *
from utils.utils import *
import multiprocessing
from argparse import ArgumentParser


multiprocessing.set_start_method('spawn', force=True)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc4 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc5 = nn.Linear(2 * hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x



def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(params):
    try:
        set_points, num_epochs = params

        device = setup_device()


        config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "mse_weight": 0.2,
            "dist_weight": 5,
            "num_epochs": num_epochs,
            "arange": 1000,
            "set_classes": 2
        }


        all_acc, all_iou, l_iou, u_iou, count = 0, 0, 0, 0, 0
        for i in range(config["arange"]):

            if i in [107, 224, 131, 363]:
                continue


            file = f"data/ALL_data_{i}.txt"
            flabel = f"data/ALL_label_{i}.txt"
            ffeature = f"data/ALL_featues_{i}.txt"

            points = np.loadtxt(file)
            flabels = np.loadtxt(flabel)
            features = np.loadtxt(ffeature)


            new_label = sample_k_classes(flabels, config["set_classes"], set_points)
            labeled_classes = np.unique(new_label[new_label != 0])
            unlabeled_classes = np.setdiff1d(np.unique(flabels), labeled_classes)


            unlabel_data_labels = Soft_Low_Rank_Approximation(set_points, config["dist_weight"], i, features, new_label,
                                                            len(np.unique(flabels)), kernel_type='knn', knn_num_neighbors=10,
                                                            max_iter=400)
            clabel = f"{set_points}_{config['dist_weight']}_{i}:confident_labels_iter_0300.txt"
            clabels = np.loadtxt(clabel)


            X, y = prepare_data(features, flabels, clabels, device)


            model, criterion, optimizer = setup_model(X.shape[1], len(np.unique(flabels)), device, config)


            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

  
            trained_model, epoch_acc, epoch_iou = train_epochs(model, loader, criterion, optimizer, config["num_epochs"], device)
            

            predicted_labels, true_labels = evaluate_model(trained_model, features, flabels, device)


            dataset_accuracy = accuracy_score(true_labels, predicted_labels)
            dataset_mean_iou = calculate_mean_iou(predicted_labels, true_labels, len(np.unique(flabels)))


            all_acc += dataset_accuracy
            all_iou += dataset_mean_iou
            l_iou += calculate_classwise_mean_iou(predicted_labels, true_labels, labeled_classes)
            u_iou += calculate_classwise_mean_iou(predicted_labels, true_labels, unlabeled_classes)
            if u_iou == 0:
                count += 1


        log_results(all_acc, all_iou, l_iou, u_iou, count, config, num_epochs)

        return (set_classes, set_points, all_acc / config["arange"] * 100, all_iou / config["arange"] * 100,
                l_iou / config["arange"] * 100, u_iou / (config["arange"] - count) * 100)

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def setup_device():
    """Set up device for training"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    return device

def prepare_data(features, flabels, clabels, device):
    """Prepare and filter data"""
    mask = clabels != 0
    X_train_filtered = features[mask]
    y_train_filtered = clabels[mask] - 1 
    X_tensor = torch.tensor(X_train_filtered, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train_filtered, dtype=torch.long).to(device)
    return X_tensor, y_tensor

def setup_model(input_size, output_size, device, config):
    """Set up MLP model, criterion and optimizer"""
    model = MLP(input_size, 128, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    return model, criterion, optimizer

def train_epochs(model, loader, criterion, optimizer, num_epochs, device):
    """Train the model for a number of epochs"""
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_one_hot = torch.zeros(targets.size(0), outputs.size(1)).to(device)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets_one_hot)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, total_loss

def evaluate_model(model, features, flabels, device):
    """Evaluate model's accuracy"""
    with torch.no_grad():
        outputs = model(torch.tensor(features, dtype=torch.float32).to(device))
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy(), flabels

def log_results(all_acc, all_iou, l_iou, u_iou, count, config, num_epochs):
    """Log results"""
    logging.info(f"Total Accuracy: {all_acc / config['arange'] * 100:.2f}%")
    logging.info(f"Total Mean IOU: {all_iou / config['arange'] * 100:.2f}%")
    logging.info(f"Labeled Mean IOU: {l_iou / config['arange'] * 100:.2f}%")
    logging.info(f"Unlabeled Mean IOU: {u_iou / (config['arange'] - count) * 100:.2f}%")
    logging.info(f"Total Epochs: {num_epochs}")
