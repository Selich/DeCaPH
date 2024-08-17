import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("")
    
    def train_model(self, dataloader, optimizer, criterion):
        self.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)
    
    def evaluate_model(self, dataloader, criterion):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return running_loss / len(dataloader), accuracy
