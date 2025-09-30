
import numpy as np
import matplotlib.pyplot as plt

# Load accuracy curves
acc_ebm = np.load('./acc_mlp_ebm.npy')
acc_avg_noisy = np.load('./acc_mlp_avg.npy')
acc_avg_clean = np.load('./acc_mlp_avg_clean.npy')
acc_centralized = np.load('./acc_mlp_centralized.npy')

# Load loss curves
loss_ebm = np.load('./loss_mlp_ebm.npy')
loss_avg_noisy = np.load('./loss_mlp_avg.npy')
loss_avg_clean = np.load('./loss_mlp_avg_clean.npy')
loss_centralized = np.load('./loss_mlp_centralized.npy')

# X-axis
rounds = len(acc_ebm)
x = np.arange(1, rounds + 1)

# Set up the figure with two subplots
plt.figure(figsize=(12, 10))

# === Accuracy subplot ===
plt.subplot(2, 1, 1)
plt.title("Test Accuracy vs Communication Rounds", fontsize=14)
plt.plot(x, acc_ebm, label="EBM", color="green", linewidth=2)
plt.plot(x, acc_avg_noisy, label="FedAvg (noisy)", color="blue", linestyle="--", linewidth=2)
plt.plot(x, acc_avg_clean, label="FedAvg (clean)", color="orange", linestyle="-.", linewidth=2)
plt.plot(x, acc_centralized, label="Centralized", color="black", linestyle=":", linewidth=2)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle=':')
plt.legend()

# === Loss subplot ===
plt.subplot(2, 1, 2)
plt.title("Loss vs Communication Rounds", fontsize=14)
plt.plot(x, loss_ebm, label="EBM", color="green", linewidth=2)
plt.plot(x, loss_avg_noisy, label="FedAvg (noisy)", color="blue", linestyle="--", linewidth=2)
plt.plot(x, loss_avg_clean, label="FedAvg (clean)", color="orange", linestyle="-.", linewidth=2)
plt.plot(x, loss_centralized, label="Centralized", color="black", linestyle=":", linewidth=2)
plt.xlabel("Communication Rounds", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle=':')
plt.legend()

plt.tight_layout()
plt.savefig('Images/fed_avg_vs_ebm_accuracy_loss4.png', dpi=300)
plt.show()
