# import numpy as np
# import matplotlib.pyplot as plt

# acc_mlp_EBM = np.load('./acc_mlp_ebm.npy')
# acc_mlp_centralized = np.load('./acc_mlp_centralized.npy')
# acc_mlp_avg = np.load('./acc_mlp_avg.npy')
# x = np.arange(1, len(acc_mlp_EBM) + 1)

# plt.figure(figsize=(8, 6))
# plt.title("FedAvg test accuracy after $t$ rounds on iid MNIST")
# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")

# # Set axis dynamically to your data range
# min_acc = np.min(acc_mlp_EBM)
# max_acc = np.max(acc_mlp_EBM)
# plt.axis([0, len(x)+1, max(0.1, min_acc-0.02), min(1.0, max_acc+0.03)])

# plt.plot(x, acc_mlp_EBM, label='2NN, $m=10$, $E=1$')

# # Only adding baselines that make sense
# if max_acc > 0.97:
#     plt.axhline(y=0.97, color='r', linestyle='dashed', label='0.97 baseline')
# if max_acc > 0.99:
#     plt.axhline(y=0.99, color='b', linestyle='dashed', label='0.99 target')

# plt.legend()
# plt.grid(True)
# # Save in under folder name 'Images'
# plt.savefig('Images/fed_avg_EBM_accuracy3.png', dpi=300)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Load accuracy curves
acc_mlp_EBM = np.load('./acc_mlp_ebm.npy')
acc_mlp_centralized = np.load('./acc_mlp_centralized.npy')
acc_mlp_avg = np.load('./acc_mlp_avg.npy')

# Create x-axis (communication rounds)
x = np.arange(1, len(acc_mlp_EBM) + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.title("Test Accuracy vs. Communication Rounds (MNIST, IID Clients)", fontsize=14)
plt.xlabel("Communication Rounds", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)

plt.plot(x, acc_mlp_EBM, label="EBM", color="green", linewidth=2)
plt.plot(x, acc_mlp_avg, label="FedAvg", color="blue", linestyle="--", linewidth=2)
plt.plot(x, acc_mlp_centralized, label="Centralized", color="black", linestyle=":", linewidth=2)

# Axis limits
min_acc = min(acc_mlp_EBM.min(), acc_mlp_avg.min(), acc_mlp_centralized.min())
max_acc = max(acc_mlp_EBM.max(), acc_mlp_avg.max(), acc_mlp_centralized.max())
plt.ylim([max(0.1, min_acc - 0.02), min(1.0, max_acc + 0.03)])
plt.xlim([0, len(x) + 1])

# Optional baseline markers
if max_acc > 0.97:
    plt.axhline(y=0.97, color='gray', linestyle='dashed', linewidth=1, label='97% Baseline')
if max_acc > 0.99:
    plt.axhline(y=0.99, color='gray', linestyle='dotted', linewidth=1, label='99% Target')

plt.grid(True, linestyle=':')
plt.legend(fontsize=11)
plt.tight_layout()

# Save plot
plt.savefig('Images/fed_avg_vs_ebm_vs_centralized_accuracy.png', dpi=300)
plt.show()
