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
        combined_dim = input_dim + input_dim  
        encoder_layer = nn.TransformerEncoderLayer(d_model=combined_dim, nhead=4)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        model = nn.Sequential(
            transformer_encoder,
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(combined_dim, output_dim)
        )
        return model

    def choose_action(self, state, user_input_embedding, epsilon):
        if np.random.rand() <= epsilon:
            return random.sample(range(1, len(state)), 5)
        else:
            repeated_user_input = user_input_embedding.repeat(len(state), 1)
            combined_input = torch.cat([torch.stack(state), repeated_user_input], dim=-1).unsqueeze(1)
            q_values = self.model(combined_input)
            _, indices = q_values.topk(5)
            return indices.tolist()

    def train(self, state, user_input_embedding, action, reward, next_state, next_user_input_embedding, done):
        repeated_user_input = user_input_embedding.repeat(len(state), 1)
        state_tensor = torch.cat([torch.stack(state), repeated_user_input], dim=-1).unsqueeze(1)
        
        repeated_next_user_input = next_user_input_embedding.repeat(len(next_state), 1)
        next_state_tensor = torch.cat([torch.stack(next_state), repeated_next_user_input], dim=-1).unsqueeze(1)

        q_values = self.model(state_tensor).squeeze()
        next_q_values = self.model(next_state_tensor).squeeze()
        max_next_q_values, _ = next_q_values.max(dim=0)
        target_q_value = reward + (1 - done) * self.gamma * max_next_q_values
        q_values[action] = target_q_value
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_value)
        loss.backward()
        self.optimizer.step()
