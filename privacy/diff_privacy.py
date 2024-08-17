import torch

def clip_gradients(gradients, max_norm):
    total_norm = torch.norm(gradients)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        gradients = gradients * clip_coef
    return gradients

def add_noise(gradients, noise_multiplier, num_samples):
    noise = torch.randn_like(gradients) * noise_multiplier / num_samples
    return gradients + noise
