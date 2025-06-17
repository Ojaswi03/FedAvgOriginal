# import numpy as np
# import matplotlib.pyplot as plt


# acc_mlp_iid_m10 = np.load('./acc_mlp_iid_m10.npy')
# acc_mlp_iid_m50 = np.load('./acc_mlp_iid_m50.npy')
# acc_cnn_iid_m10 = np.load('./acc_cnn_iid_m10.npy')
# acc_cnn_iid_m50 = np.load('./acc_cnn_iid_m50.npy')

# acc_mlp_noniid_m10 = np.load('./acc_mlp_noniid_m10.npy')
# acc_mlp_noniid_m50 = np.load('./acc_mlp_noniid_m50.npy')
# acc_cnn_noniid_m10 = np.load('./acc_cnn_noniid_m10.npy')
# acc_cnn_noniid_m50 = np.load('./acc_cnn_noniid_m50.npy')

# x = np.arange(1, len(acc_mlp_iid_m10) + 1)
# plt.figure(figsize=(8,6))

# plt.title("FedAvg test accuracy after $t$ rounds on iid MNIST")

# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")
# plt.axis([0, 100, 0.9, 1])

# plt.axhline(y=0.97, color='r', linestyle='dashed')
# plt.axhline(y=0.99, color='b', linestyle='dashed')

# plt.plot(x, acc_mlp_iid_m10, label='2NN, $m=10$, $E=1$')
# plt.plot(x, acc_mlp_iid_m50, label='2NN, $m=50$, $E=1$')

# plt.plot(x, acc_cnn_iid_m10, label='CNN, $m=10$, $E=5$')
# plt.plot(np.arange(1,52), acc_cnn_iid_m50, label='CNN, $m=50$, $E=5$')

# plt.legend()
# plt.grid(True)
# plt.savefig('fed_avg_mlp_iid_accuracy.png')
# plt.show()


# x = np.arange(1,301)
# plt.figure(figsize=(8,6))

# plt.title("FedAvg test accuracy after $t$ rounds on non-iid MNIST")

# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")
# plt.axis([0, 300, 0.85, 1])

# plt.axhline(y=0.97, color='r', linestyle='dashed')
# plt.axhline(y=0.99, color='b', linestyle='dashed')

# plt.plot(x, acc_mlp_noniid_m10[:300], label='2NN, $m=10$, E=1')
# plt.plot(x, acc_mlp_noniid_m50, label='2NN, $m=50$, E=1')

# plt.plot(x[:200], acc_cnn_noniid_m10, label='CNN, $m=10$, E=5')
# plt.plot(x[:100], acc_cnn_noniid_m50, label='CNN, $m=50$, E=5')

# plt.legend()

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt


# acc_mlp_iid_m10 = np.load('./acc_mlp_iid_m10.npy')
# acc_mlp_iid_m50 = np.load('./acc_mlp_iid_m50.npy')
# acc_cnn_iid_m10 = np.load('./acc_cnn_iid_m10.npy')
# acc_cnn_iid_m50 = np.load('./acc_cnn_iid_m50.npy')

# # acc_mlp_noniid_m10 = np.load('./acc_mlp_noniid_m10.npy')
# # acc_mlp_noniid_m50 = np.load('./acc_mlp_noniid_m50.npy')
# # acc_cnn_noniid_m10 = np.load('./acc_cnn_noniid_m10.npy')
# # acc_cnn_noniid_m50 = np.load('./acc_cnn_noniid_m50.npy')

# # --- CHANGE STARTS ---
# # Instead of hardcoding 'x = np.arange(1,101)', make it dynamic for each plot
# # For acc_mlp_iid_m10:
# x_mlp_iid_m10 = np.arange(1, len(acc_mlp_iid_m10) + 1)
# # For acc_mlp_iid_m50:
# x_mlp_iid_m50 = np.arange(1, len(acc_mlp_iid_m50) + 1)
# # For acc_cnn_iid_m10:
# x_cnn_iid_m10 = np.arange(1, len(acc_cnn_iid_m10) + 1)
# # For acc_cnn_iid_m50:
# x_cnn_iid_m50 = np.arange(1, len(acc_cnn_iid_m50) + 1)
# # --- CHANGE ENDS ---


# plt.figure(figsize=(8,6))

# plt.title("FedAvg test accuracy after $t$ rounds on iid MNIST")

# plt.xlabel("Communication rounds $t$")
# plt.ylabel("Test accuracy")
# plt.axis([0, 50, 0.9, 1])

# plt.axhline(y=0.97, color='r', linestyle='dashed')
# plt.axhline(y=0.99, color='b', linestyle='dashed')

# # --- CHANGE STARTS ---
# # Use the newly defined dynamic x-arrays for plotting
# plt.plot(x_mlp_iid_m10, acc_mlp_iid_m10, label='2NN, $m=10$, $E=1$')
# plt.plot(x_mlp_iid_m50, acc_mlp_iid_m50, label='2NN, $m=50$, $E=1$')

# plt.plot(x_cnn_iid_m10, acc_cnn_iid_m10, label='CNN, $m=10$, $E=5$')
# # If you have a line for acc_cnn_iid_m50, adjust it similarly
# plt.plot(x_cnn_iid_m50, acc_cnn_iid_m50, label='CNN, $m=50$, $E=5$')
# # --- CHANGE ENDS ---

# # ... rest of your plotting code, adjust any other plots similarly
# plt.legend()
# plt.grid(True)
# plt.savefig('fed_avg_iid_accuracy.png')
# plt.show() # Add this line to display the plot if not already present





import numpy as np
import matplotlib.pyplot as plt

acc_mlp_EBM = np.load('./acc_mlp_ebm_10.npy')
acc_mlp_WCM = np.load('./acc_mlp_wc.npy')
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
plt.plot(x, acc_mlp_WCM, label='2NN, $m=10$, $E=1$ (WCM)')
# Only adding baselines that make sense
if max_acc > 0.97:
    plt.axhline(y=0.97, color='r', linestyle='dashed', label='0.97 baseline')
if max_acc > 0.99:
    plt.axhline(y=0.99, color='b', linestyle='dashed', label='0.99 target')

plt.legend()
plt.grid(True)
plt.savefig('fed_avg_EBM_WCM_accuracy.png')
plt.show()
