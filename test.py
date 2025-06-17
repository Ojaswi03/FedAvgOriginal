
"""
EBM Implementation

"""
import os
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
bsz = 10 # Batch size for local training
SIGMA = 1.0     # Standard deviation for Gaussian noise
S = 5           # Number of noise samples per client
num_clients = 100 # Total number of clients
num_rounds = 50 # Total number of communication rounds
clients_per_round = 10 # Number of clients selected per round
local_epochs = 1 # Number of local epochs per client
lr = 0.05 # Learning rate for local training

# Load Data
train_data, test_data = fetch_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
iid_client_train_loader = iid_partition_loader(train_data, bsz=bsz)
# set up noniid_client_train_loader as needed.

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

# def train_client(client_loader, global_model, num_local_epochs, lr):
#     # Returns a trained model (local update)
#     local_model = copy.deepcopy(global_model)
#     local_model.to(device)
#     local_model.train()
#     optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

#     #  implement 2nd half of Eq13 
#     # Gradient of loss function take the norm an dthe apply varience and add to loss funtion
#     for _ in range(num_local_epochs):
#         for x, y in client_loader:
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             out = local_model(x)
#             loss = criterion(out, y)
#             loss.backward()
#             optimizer.step()
#     return local_model.state_dict()


def train_client(client_loader, global_model, num_local_epochs, lr, lambd=0.01):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for _ in range(num_local_epochs):
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = local_model(x)
            loss = criterion(output, y)

            # Backward to get gradients
            grads = []
            loss.backward(retain_graph=True)

            for param in local_model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().clone().flatten())

            flat_grads = torch.cat(grads)

            # Now compute variance of the gradients
            grad_variance = torch.var(flat_grads)
            #Norm of loss from loss = criterion(output, y)
            loss_norm = torch.norm(flat_grads)
            #norm ** 2 Square
            grad_norm_square = torch.norm(flat_grads) ** 2

            # Total loss: empirical + lambda * variance
            total_loss = loss + (grad_variance * grad_norm_square)

            # Recompute to get gradients of the total loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return local_model.state_dict()



def add_gaussian_noise(model, sigma):
    noisy_model = copy.deepcopy(model)
    for param in noisy_model.parameters():
        noise = torch.normal(0.0, sigma, size=param.data.size(), device=param.data.device)
        print(f"Noise mean: {noise.mean().item()} to parameter with shape {param.data.shape}")
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



# centralized training

def fed_Centralized(global_model, client_loaders, num_rounds,
                    clients_per_round, local_epochs, lr,
                    sigma, S, filename, verbose=True):
    acc_list = []

    # Merge all client data into one DataLoader
    merged_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in client_loaders])
    merged_loader = torch.utils.data.DataLoader(merged_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.SGD(global_model.parameters(), lr=lr)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    for t in range(num_rounds):
        if verbose:
            print(f"\n--- Centralized Round {t} ---")
        global_model.train()

        for epoch in range(local_epochs):
            for x, y in merged_loader:
                x = x.to(next(global_model.parameters()).device)
                y = y.to(next(global_model.parameters()).device)

                optimizer.zero_grad()
                output = global_model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        val_acc = validate(global_model)  # Direct call
        acc_list.append(val_acc)
        if verbose:
            print(f"Round {t}, Validation Accuracy: {val_acc:.4f}")

        if t % 10 == 0 or t == num_rounds - 1:
            np.save(f"{filename}_{t}.npy", np.array(acc_list))

    return np.array(acc_list), global_model


def fed_EBM(global_model, client_loaders, num_rounds, clients_per_round, local_epochs, lr, sigma, S, filename):
    acc_list = []
    client_ids = list(range(len(client_loaders)))

    for t in range(num_rounds):
        print(f"\n--- Round {t} ---")
        selected_clients = np.random.choice(client_ids, clients_per_round, replace=False)
        client_updates = []

        for cid in selected_clients:
            # For each client, take S expectation samples
            noise_updates = []
            for s in range(S):
                # 1. Add noise to global model
                noisy_global = add_gaussian_noise(global_model, sigma)
                # 2. Train locally
                local_update = train_client(client_loaders[cid], noisy_global, local_epochs, lr)
                noise_updates.append(local_update)
            # 3. Average the S updates for this client
            expected_update = average_state_dicts(noise_updates)
            client_updates.append(expected_update)

        # Aggregate client updates
        new_global_state = average_state_dicts(client_updates)
        global_model.load_state_dict(new_global_state)

        val_acc = validate(global_model)
        print(f"Round {t}, Validation Accuracy: {val_acc:.4f}")
        acc_list.append(val_acc)

        if t % 10 == 0:
            np.save(filename + f'_{t}.npy', np.array(acc_list))
    return np.array(acc_list)


def fed_avg_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename):

#def fed_avg_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename):
    round_accuracy = []
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace = False)
        print("clients: ", clients)

        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None

        for i,c in enumerate(clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i+1,num_clients_per_round, c))
            local_model = train_client(c, client_train_loader[c], global_model, num_local_epochs, lr)

            # add local model parameters to running average
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1/num_clients_per_round)
        
        # set global model parameters for the next step
        global_model.load_state_dict(running_avg)

        # validate
        val_acc = validate(global_model)
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(val_acc)

        if (t % 10 == 0):
          np.save(filename+'_{}'.format(t)+'.npy', np.array(round_accuracy))

    return np.array(round_accuracy)







# ---- Run EBM Fed Learning ----
mlp = MLP()
print(mlp)
print("total params:", num_params(mlp))

acc_mlp_ebm = fed_EBM(
    global_model=mlp,
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
print(acc_mlp_ebm)


# acc_mlp_avg = fed_avg_experiment(
#     global_model=mlp,
#     num_clients_per_round=clients_per_round,
#     num_local_epochs=local_epochs,
#     lr=lr,
#     client_train_loader=iid_client_train_loader,
#     max_rounds=num_rounds,
#     filename='./acc_mlp_avg'
# )
# np.save('./acc_mlp_avg.npy', acc_mlp_avg)
# print(acc_mlp_avg)

# acc_mlp_centralized, _ = fed_Centralized(
#     global_model=mlp,
#     client_loaders=iid_client_train_loader,
#     num_rounds=num_rounds,
#     clients_per_round=clients_per_round,
#     local_epochs=local_epochs,
#     lr=lr,
#     sigma=0,
#     S=S,
#     filename='./acc_mlp_centralized'
# )
# np.save('./acc_mlp_centralized.npy', acc_mlp_centralized)

# print(acc_mlp_centralized)
