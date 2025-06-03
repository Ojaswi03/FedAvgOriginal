import numpy as np
import matplotlib.pyplot as plt


acc_mlp_iid_m10 = np.load('./acc_mlp_iid_m10.npy')
acc_mlp_iid_m50 = np.load('./acc_mlp_iid_m50.npy')
acc_cnn_iid_m10 = np.load('./acc_cnn_iid_m10.npy')
acc_cnn_iid_m50 = np.load('./acc_cnn_iid_m50.npy')

acc_mlp_noniid_m10 = np.load('./acc_mlp_noniid_m10.npy')
acc_mlp_noniid_m50 = np.load('./acc_mlp_noniid_m50.npy')
acc_cnn_noniid_m10 = np.load('./acc_cnn_noniid_m10.npy')
acc_cnn_noniid_m50 = np.load('./acc_cnn_noniid_m50.npy')

x = np.arange(1,101)
plt.figure(figsize=(8,6))

plt.title("FedAvg test accuracy after $t$ rounds on iid MNIST")

plt.xlabel("Communication rounds $t$")
plt.ylabel("Test accuracy")
plt.axis([0, 100, 0.9, 1])

plt.axhline(y=0.97, color='r', linestyle='dashed')
plt.axhline(y=0.99, color='b', linestyle='dashed')

plt.plot(x, acc_mlp_iid_m10, label='2NN, $m=10$, $E=1$')
plt.plot(x, acc_mlp_iid_m50, label='2NN, $m=50$, $E=1$')

plt.plot(x, acc_cnn_iid_m10, label='CNN, $m=10$, $E=5$')
plt.plot(np.arange(1,52), acc_cnn_iid_m50, label='CNN, $m=50$, $E=5$')

plt.legend()

plt.show()


x = np.arange(1,301)
plt.figure(figsize=(8,6))

plt.title("FedAvg test accuracy after $t$ rounds on non-iid MNIST")

plt.xlabel("Communication rounds $t$")
plt.ylabel("Test accuracy")
plt.axis([0, 300, 0.85, 1])

plt.axhline(y=0.97, color='r', linestyle='dashed')
plt.axhline(y=0.99, color='b', linestyle='dashed')

plt.plot(x, acc_mlp_noniid_m10[:300], label='2NN, $m=10$, E=1')
plt.plot(x, acc_mlp_noniid_m50, label='2NN, $m=50$, E=1')

plt.plot(x[:200], acc_cnn_noniid_m10, label='CNN, $m=10$, E=5')
plt.plot(x[:100], acc_cnn_noniid_m50, label='CNN, $m=50$, E=5')

plt.legend()

plt.show()


