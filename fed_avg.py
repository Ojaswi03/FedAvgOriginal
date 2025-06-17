
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
bsz = 10 # Batch size for local training
SIGMA = 0.1     # Standard deviation for Gaussian noise
S = 5           # Number of noise samples per client
num_clients = 100 # Total number of clients
num_rounds = 100 # Total number of communication rounds
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

def train_client(client_loader, global_model, num_local_epochs, lr):
    # Returns a trained model (local update)
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    for _ in range(num_local_epochs):
        for x, y in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return local_model.state_dict()

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


# I deal Federated Learning NO EBM

#     In the ideal setting of Federated Averaging (FedAvg), the central server aggregates the model updates from participating clients by calculating a weighted average of their model parameters. This weighted average forms the new global model, which is then sent back to the clients for the next round of training. 
# The Mathematical Formula:
# In the ideal setting, where all clients participate and data distribution is assumed to be Independent and Identically Distributed (IID) or relatively homogeneous, the aggregation formula is: 
# Explanation of the Formula:
# w^(t+1)_global: Represents the global model parameters for the next communication round (t+1).
# k: Represents each participating client in the current round.
# nk/n: This is the weight factor for each client's contribution.
# nk: The number of data samples on client k.
# n: The total number of data samples across all participating clients in the round.
# w^(t+1)_k: Represents the updated model parameters of client k after local training. 
# In essence, the formula implies: 
# Each client (k) trains its local model based on the current global model received from the server, using its local dataset (nk).
# The client sends its updated local model parameters (w^(t+1)_k) back to the server.
# The server aggregates these updates by weighting each client's model parameters proportionally to the size of their local dataset (nk/n).
# The aggregated global model (w^(t+1)_global) is then distributed back to the clients, and the process repeats. 

def fed_avg(global_model, client_loaders, num_rounds, clients_per_round, local_epochs, lr, filename):
    acc_list = []
    client_ids = list(range(len(client_loaders)))

    for t in range(num_rounds):
        print(f"\n--- Round {t} ---")
        selected_clients = np.random.choice(client_ids, clients_per_round, replace=False)
        client_updates = []

        for cid in selected_clients:
            # Train locally on the selected client
            local_update = train_client(client_loaders[cid], global_model, local_epochs, lr)
            client_updates.append(local_update)

        # Aggregate client updates
        new_global_state = average_state_dicts(client_updates)
        global_model.load_state_dict(new_global_state)

        val_acc = validate(global_model)
        print(f"Round {t}, Validation Accuracy: {val_acc:.4f}")
        acc_list.append(val_acc)
        
        if t % 10 == 0:
            np.save(filename + f'_{t}.npy', np.array(acc_list))
        return np.array(acc_list)

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


acc_mlp_avg = fed_avg(
    global_model=mlp,
    client_loaders=iid_client_train_loader,
    num_rounds=num_rounds,
    clients_per_round=clients_per_round,
    local_epochs=local_epochs,
    lr=lr,
    sigma=SIGMA,
    S=S,
    filename='./acc_mlp_avg'
)
np.save('./acc_mlp_avg.npy', acc_mlp_ebm)
print(acc_mlp_avg)

acc_mlp_centralized = fed_EBM(
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




# MLP - iid - m=50 experiment
# mlp_iid_m50 = copy.deepcopy(mlp)
# acc_mlp_iid_m50 = fed_avg_experiment(mlp_iid_m50, num_clients_per_round=50, 
#                                  num_local_epochs=1,
#                                  lr=0.05,
#                                  client_train_loader = iid_client_train_loader,
#                                  max_rounds=100,# 100
#                                  filename='./acc_mlp_iid_m50',
#                                  sigma_e=0.1)
# print(acc_mlp_iid_m50)
# np.save('./acc_mlp_iid_m50.npy', acc_mlp_iid_m50)


# # MLP - non-iid - m=10 experiment
# mlp_noniid_m10 = copy.deepcopy(mlp)
# acc_mlp_noniid_m10 = fed_avg_experiment(mlp_noniid_m10, num_clients_per_round=10, 
#                                  num_local_epochs=1,
#                                  lr=0.05,
#                                  client_train_loader = noniid_client_train_loader,
#                                  max_rounds=300,
#                                  filename = './acc_mlp_noniid_m10')
# print(acc_mlp_noniid_m10)
# np.save('./acc_mlp_noniid_m10.npy', acc_mlp_noniid_m10)



# # MLP - noniid - m=50 experiment
# mlp_noniid_m50 = copy.deepcopy(mlp)
# acc_mlp_noniid_m50 = fed_avg_experiment(mlp_noniid_m50, num_clients_per_round=50, 
#                                  num_local_epochs=1,
#                                  lr=0.05,
#                                  client_train_loader = noniid_client_train_loader,
#                                  max_rounds=300,
#                                  filename='./acc_mlp_noniid_m50')
# print(acc_mlp_noniid_m50)
# np.save('./acc_mlp_noniid_m50.npy', acc_mlp_noniid_m50)


# cnn = CNN()
# print(cnn)
# print("total params: ", num_params(cnn))


# # CNN - iid - m=10 experiment
# cnn_iid_m10 = copy.deepcopy(cnn)
# acc_cnn_iid_m10 = fed_avg_experiment(cnn_iid_m10, num_clients_per_round=10, 
#                                  num_local_epochs=5,
#                                  lr=0.01,
#                                  client_train_loader = iid_client_train_loader,
#                                  max_rounds=100,  # 100
#                                  filename='./acc_cnn_iid_m10')
# print(acc_cnn_iid_m10)
# np.save('./acc_cnn_iid_m10.npy', acc_cnn_iid_m10)


# # CNN - iid - m=50 experiment
# cnn_iid_m50 = copy.deepcopy(cnn)
# acc_cnn_iid_m50 = fed_avg_experiment(cnn_iid_m50, num_clients_per_round=50, 
#                                  num_local_epochs=5,
#                                  lr=0.01,
#                                  client_train_loader = iid_client_train_loader,
#                                  max_rounds=100,  # 100
#                                  filename='./acc_cnn_iid_m50')
# print(acc_cnn_iid_m50)
# np.save('./acc_cnn_iid_m50.npy', acc_cnn_iid_m50)


# # CNN - non-iid - m=10 experiment
# cnn_noniid_m10 = copy.deepcopy(cnn)
# acc_cnn_noniid_m10 = fed_avg_experiment(cnn_noniid_m10, num_clients_per_round=10, 
#                                  num_local_epochs=5,
#                                  lr=0.01,
#                                  client_train_loader = noniid_client_train_loader,
#                                  max_rounds=200,
#                                  filename='./acc_cnn_noniid_m10')
# print(acc_cnn_noniid_m10)
# np.save('./acc_cnn_noniid_m10.npy', acc_cnn_noniid_m10)



# # CNN - non-iid - m=50 experiment
# cnn_noniid_m50 = copy.deepcopy(cnn)
# acc_cnn_noniid_m50 = fed_avg_experiment(cnn_noniid_m50, num_clients_per_round=50, 
#                                  num_local_epochs=5,
#                                  lr=0.01,
#                                  client_train_loader = noniid_client_train_loader,
#                                  max_rounds=100,
#                                  filename='./acc_cnn_noniid_m50')
# print(acc_cnn_noniid_m50)
# np.save('./acc_cnn_noniid_m50.npy', acc_cnn_noniid_m50)