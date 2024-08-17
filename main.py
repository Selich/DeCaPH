from utils.data_loader import load_data
from utils.data_splitter import load_all_data, split_data_heterogeneous, save_hospital_data
from models.cnn_model import CNNModel
from utils.leader_selection import randomly_select_leader
from privacy.diff_privacy import clip_gradients, add_noise
from privacy.sec_aggr import secure_aggregate
import torch
from torch.utils.tensorboard import SummaryWriter

all_data = load_all_data("./data/original/")
hospital_splits = split_data_heterogeneous(all_data, num_hospitals=3)
save_hospital_data(hospital_splits, base_dir="data")

num_hospitals = 3
input_size = 224 * 224 * 3
output_size = 3
learning_rate = 0.001
max_norm = 1.0
noise_multiplier = 1.0
batch_size = 32
communication_rounds = 10

writer = SummaryWriter("runs/DeCaPH_experiment")

participants = []
for i in range(num_hospitals):
    model = CNNModel(num_classes=output_size)
    data_loader = load_data(f"data/hospital_{i+1}/", f"data/hospital_{i+1}/labels.csv", batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    participants.append((model, data_loader, optimizer, criterion))

for round in range(communication_rounds):
    leader_idx = randomly_select_leader(num_hospitals)
    leader_model, _, leader_optimizer, leader_criterion = participants[leader_idx]
    
    gradients_list = []
    
    for model, dataloader, optimizer, criterion in participants:
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            gradients = torch.cat([param.grad.view(-1) for param in model.parameters()])
            clipped_gradients = clip_gradients(gradients, max_norm)
            noisy_gradients = add_noise(clipped_gradients, noise_multiplier, len(dataloader.dataset))
            gradients_list.append(noisy_gradients)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        writer.add_scalar(f'Loss/Hospital_{participants.index((model, dataloader, optimizer, criterion)) + 1}', total_loss / len(dataloader), round)
        writer.add_scalar(f'Accuracy/Hospital_{participants.index((model, dataloader, optimizer, criterion)) + 1}', correct_predictions / total_samples, round)
    
    aggregated_gradients = secure_aggregate(gradients_list)
    
    leader_optimizer.zero_grad()
    offset = 0
    for param in leader_model.parameters():
        param.grad = aggregated_gradients[offset:offset+param.numel()].view(param.size())
        offset += param.numel()
    leader_optimizer.step()
    
    with torch.no_grad():
        for model, _, _, _ in participants:
            for leader_param, param in zip(leader_model.parameters(), model.parameters()):
                param.data.copy_(leader_param.data)

    print(f"{round + 1}")

for model, dataloader, _, criterion in participants:
    loss, accuracy = model.evaluate_model(dataloader, criterion)
    writer.add_scalar(f'Final_Loss/Hospital_{participants.index((model, dataloader, _, criterion)) + 1}', loss, round)
    writer.add_scalar(f'Final_Accuracy/Hospital_{participants.index((model, dataloader, _, criterion)) + 1}', accuracy, round)
    print(f"loss: {loss:.4f}, accuracy: {accuracy:.2f}%")

writer.close()
