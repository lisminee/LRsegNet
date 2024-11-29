import numpy as np
import os

def correct_labels(labels):
    unique_labels = np.unique(labels)
    max_label = max(unique_labels)
    correct_labels = np.arange(1, max_label + 1)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
    corrected_labels = np.array([label_mapping[label] for label in labels])
    return corrected_labels

def process_label_files(file_path, num_files):
    for i in range( num_files):
        file_name = f"{file_path}/ALL_label_{i}.txt"

        flabels = np.loadtxt(file_name)

        corrected_labels = correct_labels(flabels)
        with open(f"{file_path}/corrected_label_{i}.txt", 'w', encoding='utf-8') as file:
            for label in corrected_labels:
                file.write(f"{label}\n")

process_label_files("/data/all_labels", 65)
