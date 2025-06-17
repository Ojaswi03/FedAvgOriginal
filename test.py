
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

# # Hyperparameters
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

# --- Helper: Evaluate average loss of model on loader ---
def evaluate_loss(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / total


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
        # print(f"Adding noise with shape {noise.shape} and std {sigma} and noise = {noise}")
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
def average_weights(state_dict_list):

    avg_state_dict = copy.deepcopy(state_dict_list[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dict_list)):
            avg_state_dict[key] += state_dict_list[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dict_list)
    return avg_state_dict

# def running_model_avg(current, next, scale):
#     if current == None:
#         current = next
#         for key in current:
#             current[key] = current[key] * scale
#     else:
#         for key in current:
#             current[key] = current[key] + (next[key] * scale)
#     return current

# --- Main: WCM Federated Training Function ---
def fed_WCM(global_model, client_loaders, max_rounds=100, num_clients_per_round=10, local_epochs=1, lr=0.05,
            sigma=0.1, S= 5, filename='./wcm_acc', device='cpu'):

    round_acc = []
    n_clients = len(client_loaders)

    for t in range(max_rounds):
        print(f"\n--- Round {t} ---")
        selected_clients = np.random.choice(np.arange(n_clients), num_clients_per_round, replace=False)
        client_updates = []

        for cid in selected_clients:
            noisy_models = []
            noisy_losses = []

            for s in range(S):
                # Noise injected global model
                noisy_global = add_gaussian_noise(global_model, sigma)
                # Train on this noisy global
                local_update = train_client(client_loaders[cid], noisy_global, local_epochs, lr)
                # Wrap state_dict into a model to evaluate loss
                local_model = type(global_model)().to(device)
                local_model.load_state_dict(local_update)
                loss = evaluate_loss(local_model, client_loaders[cid], criterion)
                noisy_models.append(local_update)
                noisy_losses.append(loss)

            # Select worst-case update (highest loss)
            worst_idx = np.argmax(noisy_losses)
            client_updates.append(noisy_models[worst_idx])

        # Aggregate updates (FedAvg style)
        avg_update = average_weights(client_updates)
        global_model.load_state_dict(avg_update)

        # Validation accuracy
        acc = validate(global_model)
        print(f"Round {t}, Validation Accuracy: {acc:.4f}")
        round_acc.append(acc)

        if t % 10 == 0:
            np.save(f"{filename}_{t}.npy", np.array(round_acc))

    np.save(f"{filename}.npy", np.array(round_acc))
    return np.array(round_acc)



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

# def fed_avg_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename):
#     round_accuracy = []
#     for t in range(max_rounds):
#         print("starting round {}".format(t))

#         # choose clients
#         clients = np.random.choice(np.arange(100), num_clients_per_round, replace = False)
#         print("clients: ", clients)

#         global_model.eval()
#         global_model = global_model.to(device)
#         running_avg = None

#         for i,c in enumerate(clients):
#             # train local client
#             print("round {}, starting client {}/{}, id: {}".format(t, i+1,num_clients_per_round, c))
#             local_model = train_client(c, client_train_loader[c], global_model, num_local_epochs, lr)

#             # add local model parameters to running average
#             running_avg = running_model_avg(running_avg, local_model.state_dict(), 1/num_clients_per_round)
        
#         # set global model parameters for the next step
#         global_model.load_state_dict(running_avg)

#         # validate
#         val_acc = validate(global_model)
#         print("round {}, validation acc: {}".format(t, val_acc))
#         round_accuracy.append(val_acc)

#         if (t % 10 == 0):
#           np.save(filename+'_{}'.format(t)+'.npy', np.array(round_accuracy))

#     return np.array(round_accuracy)

# Hyperparameters
bsz = 10 # Batch size for local training
SIGMA = 0.1  # 0.1   # Standard deviation for Gaussian noise
S = 10   #5        # Number of noise samples per client
num_clients = 100 # Total number of clients
num_rounds = 50 # Total number of communication rounds
clients_per_round = 10 # Number of clients selected per round
local_epochs = 1 # Number of local epochs per client
lr = 0.05 # Learning rate for local training


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
    filename='./acc_mlp_ebm_10_sigma01'
)

np.save('./acc_mlp_ebm_10_sigma01.npy', acc_mlp_ebm)
print(acc_mlp_ebm)


# mlp_iid_m10 = copy.deepcopy(mlp)
# acc_mlp_iid_m10 = fed_avg_experiment(mlp_iid_m10, num_clients_per_round=10, 
#                                  num_local_epochs=1,
#                                  lr=0.05,
#                                  client_train_loader = iid_client_train_loader,
#                                  max_rounds=50,
#                                  filename='./acc_mlp_iid_m10')
# print(acc_mlp_iid_m10)
# np.save('./acc_mlp_iid_m10.npy', acc_mlp_iid_m10)




# # Example call after you define your loaders and global model
# wcm_acc = fed_WCM(
#     global_model=mlp,
#     client_loaders=iid_client_train_loader,
#     max_rounds=num_rounds,
#     clients_per_round=clients_per_round,
#     local_epochs=local_epochs,
#     lr=lr,
#     sigma=SIGMA,
#     S=S,  # Number of noise samples per client per round
#     filename='./acc_mlp_wcm_10',
#     device=device  # Uncomment if you want to specify the device
# )
# np.save('./acc_mlp_wcm_10.npy', wcm_acc)
# print(wcm_acc)

# # ---- Run EBM Fed Learning with m=50 ----
# clients_per_round = 50
# mlp = MLP()
# print(mlp)
# print("total params:", num_params(mlp))

# acc_mlp_ebm_50 = fed_EBM(
#     global_model=mlp,
#     client_loaders=iid_client_train_loader,
#     num_rounds=num_rounds,
#     clients_per_round=clients_per_round,
#     local_epochs=local_epochs,
#     lr=lr,
#     sigma=SIGMA,
#     S=S,
#     filename='./acc_mlp_ebm_50'
# )

# np.save('./acc_mlp_ebm_50.npy', acc_mlp_ebm_50)
# print(acc_mlp_ebm)





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