#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from paillier_encryption import Client, Server
from Federated_learning import federated_learning, local_learning
import mnist_loader
import network
import numpy as np
if __name__ == '__main__':
    config = {
        'n_clients': 5,
        'key_length': 1024,
        'n_iter': 50,
        'eta': 1.5,
    }
    # load data, train/test split and split training data between client




    server = Server(key_length=config['key_length'])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print ("开始训练，较耗时，请稍等。。。")
    net = network.Network([784, 30, 10])
    n_clients = config['n_clients']
    clients = []
    names = ['Hospital {}'.format(i) for i in range(1, n_clients + 1)]
    for i in range(n_clients):
        clients.append(Client(names[i], training_data[0], training_data[1], server.pubkey,[784, 30, 10]))
    for each_client in clients:
        each_client.SGD(training_data, 30, 10, 3.0, test_data = test_data)

    # first each hospital learns a model on its respective dataset for comparison.
    #local_learning(X, y, X_test, y_test, config)
    # and now the full glory of federated learning
    #federated_learning(X, y, X_test, y_test, config)