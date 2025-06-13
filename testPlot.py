import numpy as np
import matplotlib.pyplot as plt

acc_mlp_EBM = np.load('./acc_mlp_ebm.npy')
x = np.arange(1, len(acc_mlp_EBM) + 1)

plt.figure(figsize=(8, 6))
plt.title("FedAvg test accuracy after $t$ rounds on iid MNIST")
plt.xlabel("Communication rounds $t$")
plt.ylabel("Test accuracy")

# Set axis dynamically to your data range
min_acc = np.min(acc_mlp_EBM)
max_acc = np.max(acc_mlp_EBM)
plt.axis([0, len(x)+1, max(0.8, min_acc-0.02), min(1.0, max_acc+0.03)])

plt.plot(x, acc_mlp_EBM, label='2NN, $m=10$, $E=1$')

# Only adding baselines that make sense
if max_acc > 0.97:
    plt.axhline(y=0.97, color='r', linestyle='dashed', label='0.97 baseline')
if max_acc > 0.99:
    plt.axhline(y=0.99, color='b', linestyle='dashed', label='0.99 target')

plt.legend()
plt.grid(True)
plt.savefig('fed_avg_EBM_accuracy.png')
plt.show()
