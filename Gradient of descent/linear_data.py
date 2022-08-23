from tkinter.tix import InputOnly
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# code to generate the fake data 
true_slope = 10.889
true_intercept = 3.456
input_var = np.arange(0.0,100.0)
output_var = true_slope * input_var + true_intercept + 500.0 * np.random.rand(len(input_var))
plt.figure()
plt.scatter(input_var, output_var)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# code for the cost function {1}
def cost_function(input_var, output_var, params):
    m = len(input_var)
    cost_sum = 0
    for x,y in zip(input_var, output_var):
        y_hat = np.dot(params, np.array([1,x]))
        cost_sum += (y_hat - y)**2
    cost_function = cost_sum / (2*m)
    return cost_function 

# batch method {2}
def batch_method(input_var, output_var, params, alpha, max_iter):
    iteration = 0
    m = len(input_var)
    cost = np.zeros(max_iter)
    params_store = np.zeros([2,max_iter])

    while iteration < max_iter:
        cost[iteration] = cost_function(input_var, output_var, params)
        params_store[:,iteration] = params

        print("------------------------------")
        print(f"iteration: {iteration}")
        print(f'cost: {cost[iteration]}')


        for x,y in zip(input_var, output_var):
            y_hat = np.dot(params, np.array([1.0,x]))
            gradient = (np.array([1.0,x])*(y-y_hat))/m
            params += alpha * gradient

        iteration += 1

    return params, cost, params_store 

# training code {3}
x_train, x_test, y_train, y_test = train_test_split(input_var, output_var, test_size=0.20)
inital_params = np.array([20.0,80.0])
alpha_batch = 1e-3
max_iter = 100
params_hat_batch, cost_batch, params_store_batch =\
    batch_method(x_train, y_train, inital_params, alpha_batch, max_iter)


# stochastic method {4}
def stochastic_method(input_var, output_var, params, alpha):
    m = len(input_var)
    cost = np.zeros(m)
    params_store = np.zeros([2, m])

    iteration = 0
    for x,y in zip(input_var, output_var):
        cost[iteration] = cost_function(input_var, output_var, params)
        params_store[:,iteration] = params

        print("------------------------------")
        print(f"iteration: {iteration}")
        print(f'cost: {cost[iteration]}')

        y_hat = np.dot(params, np.array([1,x]))
        gradient = (np.array([1,x])*(y - y_hat))/m
        params += alpha*gradient

        iteration += 1

    return params, cost, params_store 

alpha = 1e-3
params_0 = np.array([20.0, 80.0])
params_hat, cost, params_store =\
stochastic_method(x_train, y_train, params_0, alpha)

#graph code 
plt.figure()
plt.scatter(x_test, y_test)
plt.plot(x_test, params_hat_batch[0] + params_hat_batch[1]*x_test, 'g', label='batch')
plt.plot(x_test, params_hat[0] + params_hat[1]*x_test, '-r', label='stochastic')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
print(f'batch      T0, T1: {params_hat_batch[0]}, {params_hat_batch[1]}')
print(f'stochastic T0, T1: {params_hat[0]}, {params_hat[1]}')
rms_batch = np.sqrt(np.mean(np.square(params_hat_batch[0] + params_hat_batch[1]*x_test - y_test)))
rms_stochastic = np.sqrt(np.mean(np.square(params_hat[0] + params_hat[1]*x_test - y_test)))
print(f'batch rms:      {rms_batch}')
print(f'stochastic rms: {rms_stochastic}')

plt.figure()
plt.plot(np.arange(max_iter), cost_batch, 'r', label='batch')
plt.plot(np.arange(len(cost)), cost, 'g', label='stochastic')
plt.xlabel('iteration')
plt.ylabel('normalized cost')
plt.legend()
plt.show()
print(f'min cost with BGD: {np.min(cost_batch)}')
print(f'min cost with SGD: {np.min(cost)}')



