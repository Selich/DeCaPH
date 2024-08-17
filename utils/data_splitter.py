import os
import csv
import pandas as pd
from sklearn.utils import shuffle

def load_all_data(images_dir):
    labels = {"COVID": 0, "NORMAL": 1, "VIRAL_PNEUMONIA": 2}
    data = []
    
    for label_name, label_id in labels.items():
        class_dir = os.path.join(images_dir, label_name, "images") 
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')): 
                    data.append((os.path.join(label_name, "images", img_name), label_id))
        else:
            print(f"Directory {class_dir} does not exist.")
    
    data_df = pd.DataFrame(data, columns=["filename", "label"])
    return data_df

def split_data_heterogeneous(data, num_hospitals=3):
    splits = []
    total_samples = len(data)

    ratios = [
        {"COVID": 0.5, "NORMAL": 0.3, "VIRAL_PNEUMONIA": 0.2},
        {"COVID": 0.3, "NORMAL": 0.5, "VIRAL_PNEUMONIA": 0.2},
        {"COVID": 0.4, "NORMAL": 0.2, "VIRAL_PNEUMONIA": 0.4}
    ]
    
    for i in range(num_hospitals):
        hospital_data = pd.DataFrame(columns=["filename", "label"])
        for label_name, ratio in ratios[i].items():
            label_data = data[data["label"] == {"COVID": 0, "NORMAL": 1, "VIRAL_PNEUMONIA": 2}[label_name]]
            split_size = int(total_samples * ratio / num_hospitals)

            if split_size > len(label_data):
                split_size = len(label_data) 

            selected_data = label_data.sample(n=split_size, replace=False, random_state=42)
            hospital_data = pd.concat([hospital_data, selected_data], axis=0)

            data = data.drop(selected_data.index)

        splits.append(hospital_data)
    
    return splits

def save_hospital_data(hospital_splits, base_dir="data"):
    for i, split in enumerate(hospital_splits, 1):
        hospital_dir = os.path.join(base_dir, f"hospital_{i}")
        os.makedirs(hospital_dir, exist_ok=True)

        for class_name in ["COVID", "NORMAL", "VIRAL_PNEUMONIA"]:
            class_dir = os.path.join(hospital_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        for _, row in split.iterrows():
            src = os.path.join("data/original", row["filename"])
            dst = os.path.join(hospital_dir, row["filename"])
            if os.path.exists(src):
                os.rename(src, dst)
            else:
                print(f"{src} does not exist")

        generate_labels_file(hospital_dir, os.path.join(hospital_dir, "labels.csv"))

def generate_labels_file(images_dir, output_file):
    labels = {"COVID": 0, "NORMAL": 1, "VIRAL_PNEUMONIA": 2}
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        
        for label_name, label_id in labels.items():
            class_dir = os.path.join(images_dir, label_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):  
                        writer.writerow([os.path.join(label_name, img_name), label_id])

all_data = load_all_data("data/original/")
hospital_splits = split_data_heterogeneous(all_data, num_hospitals=3)
save_hospital_data(hospital_splits, base_dir="data")
