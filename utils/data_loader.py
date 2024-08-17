import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(labels_file)
        
        if self.labels.empty:
            raise ValueError(f"labels file {labels_file} is empty or not properly formatted")
        
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data(images_dir, labels_file, batch_size=32, shuffle=True):
    if not os.path.exists(images_dir):
        raise ValueError(f"image directory {images_dir} does not exist")
    
    if not os.path.isfile(labels_file):
        raise ValueError(f"labels file {labels_file} does not exist")
    
    print(f"{images_dir} with labels from {labels_file}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MedicalImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"no data found in dataset:  {images_dir} with labels {labels_file}.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
