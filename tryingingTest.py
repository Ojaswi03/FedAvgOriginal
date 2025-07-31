"""
EBM Implementation

"""

import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import view_10, num_params
from data import fetch_dataset, iid_partition_loader, noniid_partition_loader

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("| using device:", device)

# Hyperparameters
bsz = 10                # Batch size for local training
SIGMA = 0.1             # Standard deviation for Gaussian noise
S = 5                   # Number of noise samples per client
num_clients = 100       # Total number of clients
num_rounds = 100        # Total number of communication rounds
clients_per_round = 10  # Number of clients selected per round
local_epochs = 1        # Number of local epochs per client
lr = 0.05               # Learning rate for local training

# Load Data
train_data, test_data = fetch_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
iid_client_train_loader = iid_partition_loader(train_data, bsz=bsz)
# set up noniid_client_train_loader when needed.

# Models
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

criterion = nn.CrossEntropyLoss()

def validate(model):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return correct / total

def train_client(client_loader, global_model, num_local_epochs, lr, sigma=0.0):
    # Returns a trained model (local update) and a list of loss values per epoch
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    sigma_squared = sigma ** 2
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    epoch_loss_list = []

    for _ in range(num_local_epochs):
        batch_loss = 0.0
        count = 0
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            if sigma_squared > 1e-6:  # Add noise only if sigma is significant
                grad = torch.autograd.grad(loss, local_model.parameters(), create_graph=True)
                grad_norm_squared = sum((g ** 2).sum() for g in grad)
                loss += sigma_squared * grad_norm_squared

            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            count += 1

        avg_epoch_loss = batch_loss / count if count > 0 else 0.0
        epoch_loss_list.append(avg_epoch_loss)

    return local_model.state_dict(), epoch_loss_list

def add_gaussian_noise(model, sigma):
    noisy_model = copy.deepcopy(model)
    for param in noisy_model.parameters():
        noise = torch.normal(0.0, sigma, size=param.data.size(), device=param.data.device)
        param.data += noise
    return noisy_model

def average_state_dicts(dicts):
    # Average a list of state_dicts
    avg_dict = copy.deepcopy(dicts[0])
    for key in avg_dict:
        for d in dicts[1:]:
            avg_dict[key] += d[key]
        avg_dict[key] /= len(dicts)
    return avg_dict
def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current

# Centralized Training (As per the paper)

def train_centralized(model, train_loader, test_loader, num_epochs, lr):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    acc_list = []
    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        acc_list.append(acc)
        loss_list.append(total_loss)
        print(f"Epoch {epoch}, Centralized Test Accuracy: {acc:.4f}, Loss: {total_loss:.4f}")

    return acc_list, loss_list

# Conventional FedAvg(Noise) Implementation
def fed_avg(global_model, client_loaders, num_rounds, clients_per_round, local_epochs, lr, sigma, filename=None):
    global_model = global_model.to(device)
    global_model.train()

    acc_list = []
    loss_list = []

    for rnd in range(num_rounds):
        sampled_indices = random.sample(range(len(client_loaders)), clients_per_round)
        weights_list = []
        round_loss = 0.0
        round_count = 0

        for idx in sampled_indices:
            client_loader = client_loaders[idx]
            noisy_model = copy.deepcopy(global_model)
            for name, param in noisy_model.named_parameters():
                noise = torch.normal(0, sigma, size=param.shape).to(device)
                param.data += noise

            updated_weights, local_loss_list = train_client(client_loader, noisy_model, local_epochs, lr, sigma)
            weights_list.append(updated_weights)
            round_loss += sum(local_loss_list)
            round_count += len(local_loss_list)

        global_dict = average_state_dicts(weights_list)
        global_model.load_state_dict(global_dict)

        acc = validate(global_model)
        avg_round_loss = round_loss / round_count if round_count > 0 else 0.0
        acc_list.append(acc)
        loss_list.append(avg_round_loss)

        print(f"Round {rnd+1}, FedAvg (noisy) Accuracy: {acc:.4f}, Loss: {avg_round_loss:.4f}")

        if filename and (rnd + 1) % 10 == 0:
            np.save(filename + '.npy', np.array(acc_list))
            np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    if filename:
        np.save(filename + '.npy', np.array(acc_list))
        np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    return acc_list, loss_list

# Conventional FedAvg (Clean) Implementation
def fed_avg_clean(global_model, client_loaders, num_rounds, clients_per_round, local_epochs, lr, filename=None):
    global_model = global_model.to(device)
    global_model.train()

    acc_list = []
    loss_list = []

    for rnd in range(num_rounds):
        sampled_indices = random.sample(range(len(client_loaders)), clients_per_round)
        weights_list = []
        round_loss = 0.0
        round_count = 0

        for idx in sampled_indices:
            client_loader = client_loaders[idx]
            local_model = copy.deepcopy(global_model)
            updated_weights, local_loss_list = train_client(client_loader, local_model, local_epochs, lr)
            weights_list.append(updated_weights)
            round_loss += sum(local_loss_list)
            round_count += len(local_loss_list)

        global_dict = average_state_dicts(weights_list)
        global_model.load_state_dict(global_dict)

        acc = validate(global_model)
        avg_round_loss = round_loss / round_count if round_count > 0 else 0.0
        acc_list.append(acc)
        loss_list.append(avg_round_loss)

        print(f"Round {rnd+1}, FedAvg (clean) Accuracy: {acc:.4f}, Loss: {avg_round_loss:.4f}")

        if filename and (rnd + 1) % 10 == 0:
            np.save(filename + '.npy', np.array(acc_list))
            np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    if filename:
        np.save(filename + '.npy', np.array(acc_list))
        np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    return acc_list, loss_list

# Expectation-based FedAvg (EBM) Implementation
def fed_EBM(global_model, client_loaders, num_rounds, clients_per_round, local_epochs, lr, sigma, S, filename):
    acc_list = []
    loss_list = []
    client_ids = list(range(len(client_loaders)))

    for t in range(num_rounds):
        print(f"\n--- Round {t} ---")
        selected_clients = np.random.choice(client_ids, clients_per_round, replace=False)
        client_updates = []
        round_loss = 0.0
        round_count = 0

        for cid in selected_clients:
            noise_updates = []
            all_losses = []
            for s in range(S):
                noisy_global = add_gaussian_noise(global_model, sigma)
                local_update, local_loss_list = train_client(client_loaders[cid], noisy_global, local_epochs, lr, sigma)
                noise_updates.append(local_update)
                all_losses.extend(local_loss_list)

            expected_update = average_state_dicts(noise_updates)
            client_updates.append(expected_update)
            round_loss += sum(all_losses)
            round_count += len(all_losses)

        new_global_state = average_state_dicts(client_updates)
        global_model.load_state_dict(new_global_state)

        val_acc = validate(global_model)
        avg_round_loss = round_loss / round_count if round_count > 0 else 0.0
        print(f"Round {t}, Validation Accuracy: {val_acc:.4f}, Loss: {avg_round_loss:.4f}")
        acc_list.append(val_acc)
        loss_list.append(avg_round_loss)

        if t % 10 == 0:
            np.save(filename + '.npy', np.array(acc_list))
            np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    if filename:
        np.save(filename + '.npy', np.array(acc_list))
        np.save(filename.replace('acc', 'loss') + '.npy', np.array(loss_list))

    return acc_list, loss_list

# ---- Run EBM Fed Learning ----

central = MLP()
conventionalFedAvg_noisy = MLP()
conventionalFedAvg_clean = MLP()
ebmFedAvg = MLP()

# --- Centralized Model ---
print("Centralized model:")
print(central)
print("total params:", num_params(central))
central_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

acc_mlp_centralized, loss_mlp_centralized = train_centralized(
    model=central,
    train_loader=central_train_loader,
    test_loader=test_loader,
    num_epochs=num_rounds,
    lr=lr
)
np.save('./acc_mlp_centralized.npy', acc_mlp_centralized)
np.save('./loss_mlp_centralized.npy', loss_mlp_centralized)
print("Centralized model accuracy:", acc_mlp_centralized)

# --- FedAvg Noisy ---
print("Conventional FedAvg Noisy model:")
print(conventionalFedAvg_noisy)
print("total params:", num_params(conventionalFedAvg_noisy))
acc_mlp_avg, loss_mlp_avg = fed_avg(
    global_model=conventionalFedAvg_noisy,
    client_loaders=iid_client_train_loader,
    num_rounds=num_rounds,
    clients_per_round=clients_per_round,
    local_epochs=local_epochs,
    lr=lr,
    sigma=SIGMA,
    filename='./acc_mlp_avg'
)
np.save('./acc_mlp_avg.npy', acc_mlp_avg)
np.save('./loss_mlp_avg.npy', loss_mlp_avg)
print("FedAvg accuracy:", acc_mlp_avg)

# --- FedAvg Clean ---
print("Conventional FedAvg Clean model:")
print(conventionalFedAvg_clean)
print("total params:", num_params(conventionalFedAvg_clean))
acc_mlp_avg_clean, loss_mlp_avg_clean = fed_avg_clean(
    global_model=conventionalFedAvg_clean,
    client_loaders=iid_client_train_loader,
    num_rounds=num_rounds,
    clients_per_round=clients_per_round,
    local_epochs=local_epochs,
    lr=lr,
    filename='./acc_mlp_avg_clean'
)
np.save('./acc_mlp_avg_clean.npy', acc_mlp_avg_clean)
np.save('./loss_mlp_avg_clean.npy', loss_mlp_avg_clean)
print("FedAvg Clean accuracy:", acc_mlp_avg_clean)

# --- Expectation-Based FedAvg (EBM) ---
print("Expectation-based FedAvg (EBM) model:")
print(ebmFedAvg)
print("total params:", num_params(ebmFedAvg))
acc_mlp_ebm, loss_mlp_ebm = fed_EBM(
    global_model=ebmFedAvg,
    client_loaders=iid_client_train_loader,
    num_rounds=num_rounds,
    clients_per_round=clients_per_round,
    local_epochs=local_epochs,
    lr=lr,
    sigma=SIGMA,
    S=S,
    filename='./acc_mlp_ebm'
)
np.save('./acc_mlp_ebm.npy', acc_mlp_ebm)
np.save('./loss_mlp_ebm.npy', loss_mlp_ebm)
print("EBM FedAvg accuracy:", acc_mlp_ebm)
