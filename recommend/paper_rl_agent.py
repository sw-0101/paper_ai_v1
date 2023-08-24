import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class TransformerDQNAgent:
    def __init__(self, input_dim, sequence_length, output_dim):
        self.model = self.build_model(input_dim, sequence_length, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.95

    def build_model(self, input_dim, sequence_length, output_dim):
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        model = nn.Sequential(
            transformer_encoder,
            nn.Flatten(),
            nn.Linear(input_dim , output_dim)
        )
        return model

    def choose_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.sample(range(len(state)), 5)
        else:
            q_values = self.model(torch.stack(state).unsqueeze(1))
            _, indices = q_values.topk(5)
            return indices.tolist()

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.stack(state).unsqueeze(1)
        next_state_tensor = torch.stack(next_state).unsqueeze(1)
        q_values = self.model(state_tensor).squeeze()
        next_q_values = self.model(next_state_tensor).squeeze()
        max_next_q_values, _ = next_q_values.max(dim=0)
        target_q_value = reward + (1 - done) * self.gamma * max_next_q_values
        q_values[action] = target_q_value
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_value)
        loss.backward()
        self.optimizer.step()
