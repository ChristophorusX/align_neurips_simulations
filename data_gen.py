import numpy as np
import fa_two_layer


def lr_data(n_samples, n_dim):
    X = np.random.randn(n_samples, n_dim)
    ground_truth = np.random.randn(n_dim)
    y = X @ ground_truth
    return X, y

def rand_nn_data(n_samples, n_dim, n_hidden, activation='relu'):
    seed = np.random.randint(100000)
    net = fa_two_layer.TwoLayerNetwork(activation, n_dim, n_hidden, n_samples, seed)
    X = np.random.randn(n_samples, n_dim)
    y = net.forward(X)
    return X, y

if __name__ == '__main__':
    X, y = rand_nn_data(20, 4, 100)
    print(X)
    print(y)
    X, y = lr_data(20, 4)
    print(X)
    print(y)
