#!/usr/bin/env python
# coding: utf-8

# %%

import numpy as np
import scipy.sparse as sparse
import pandas as pd

# global variables
# This will change the initial condition used. Currently it starts from the first value
shift_k = 0

approx_res_size = 5000


model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 8,
                'd': 22}

res_params = {'radius': 0.1,
              'degree': 3,
              'sigma': 0.5,
              'train_length': 100000,
              'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
              'num_inputs': model_params['N'],
              'predict_length': 10000,
              'beta': 0.0001
              }


# The ESN functions for training
def generate_reservoir(size, radius, degree):
    sparsity = degree/float(size)
    A = sparse.rand(size, size, density=sparsity).toarray()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A


def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'], res_params['train_length']))
    for i in range(res_params['train_length']-1):
        states[:, i+1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input[:, i]))
    return states


def train_reservoir(res_params, data):
    print("train_reservoir: Generating reservoir")
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N']/res_params['num_inputs'])

    print("train_reservoir: Generating Win")
    Win = np.zeros((res_params['N'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])

    print("train_reservoir: Generating reservoir layer")
    states = reservoir_layer(A, Win, data, res_params)

    print("train_reservoir: Training")
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win


def train(res_params, states, data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2, np.shape(states2)[0]-2):
        if (np.mod(j, 2) == 0):
            states2[j, :] = (states[j-1, :]*states[j-2, :]).copy()
    U = np.dot(states2, states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(states2, data.transpose()))
    return Wout.transpose()


def predict(A, Win, res_params, x, Wout):
    # Define array for storing predictions at all time steps
    output = np.zeros((res_params['num_inputs'], res_params['predict_length']))

    # Loop over all time steps
    for i in range(res_params['predict_length']):
        if i % 100 == 0 or i == 0:
            print(f"predict: Time step {i}")

        x_aug = x.copy()

        # Perform additional nonlinearity
        for j in range(2, np.shape(x_aug)[0]-2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = x[j-1]*x[j-2]

        # Apply Wout to x_aug to form prediction
        output[:, i] = np.dot(Wout, x_aug)

        x = np.tanh(np.dot(A, x) + np.dot(Win, output[:, i]))
    return output, x


# %%

print("Reading input data")
dataf = pd.read_csv('3tier_lorenz_v3.csv', header=None)
data = np.transpose(np.array(dataf))

# %%

# Train reservoir
print("Training reservoir")
print(f"res_params: {res_params}")
x, Wout, A, Win = train_reservoir(res_params, data[:, shift_k:shift_k+res_params['train_length']])

# %%

# Prediction
print("Running prediction")
output, _ = predict(A, Win, res_params, x, Wout)

# %%

print("Saving output")
np.save('Expansion_2step_back' + 'R_size_train_' + str(res_params['train_length']) + '_Rd_' +
        str(res_params['radius']) + '_Shift_' + str(shift_k) + '.npy', output)
