# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import importlib

from proj1_helpers import *

#initial_w is the initial weight vector
#gamma is the step-size
#max_iters is the number of steps to run
#lambda_ is always the regularization parameter

def least_squares_GD(y, tx, initial_w, max_itters, gamma):
    print("calc Linear regression using gradient descent")
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_itters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        loss = compute_loss_mse(y, tx, w)
        gradient = least_squares(y,tx)
        print(gradient)
       # ***************************************************
        # update w by gradient
        # ***************************************************
        w = w + gamma*gradient
        print("Gradient Descent({bi}/{ti}): loss={l},w={w}".format(
              bi=n_iter, ti=max_itters - 1, l=loss, w=w))

    return loss, w


def linear_regresssion_GD_mse(y, tx, initial_w, max_itters, gamma):
    print("calc Linear regression using gradient descent")
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    losses = []
    w = initial_w
    for n_iter in range(max_itters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        loss = compute_loss_mse(y, tx, w)
        gradient = compute_gradient_mse(y, tx, w)
        print(gradient)
       # ***************************************************
        # update w by gradient
        # ***************************************************
        w = w + gamma*gradient

        print("Gradient Descent({bi}/{ti}): loss={l},w={w}".format(
              bi=n_iter, ti=max_itters - 1, l=loss, w=w))

    return losses, w

def least_squares_SGD(y,tx,initial_w,max_iters,gamma): #giving 0.745 acc
    print("Calc Linear regression using stochastic gradient descent")
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    # ***************************************************
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        print("MINIBATCHES COMING")
        print(minibatch_y)
        print(minibatch_tx)
        # Define parameters to store w and loss
        ws = [initial_w]
        w = initial_w
        for n_iter in range(max_iters):
            # ***************************************************
            # compute gradient and loss
            # ***************************************************
            loss = compute_loss_mse(minibatch_y,minibatch_tx,w)
            gradient=compute_gradient_mse(minibatch_y,minibatch_tx,w)
            print("loss = " + str(loss))
            print("gradient = " + str(gradient))
            # ***************************************************
            # update w by gradient
            # ***************************************************
            w = w+np.multiply(gamma,gradient)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, ws

def least_squares(y,tx):
    print("Calc Least squares regression using normal equations")
    return np.linalg.inv(np.transpose(tx)@tx)@np.transpose(tx)@y

def ridge_regression(y,tx,lambda_):
    print("Ridge regression using normal equations")
    """implement ridge regression."""
    # ***************************************************
    # ridge regression: TODO
    # ***************************************************
    w = np.linalg.inv(np.transpose(tx) @ tx + lambda_) @ np.transpose(tx) @ y
    # w = np.linalg.inv(np.transpose(tx)@tx+np.identity(lambda_))@np.transpose(tx)@y
    summe = np.sum(np.power(y - tx @ w, 2))
    summe = summe / (2 * len(y))
    result = summe + lambda_ * np.sum(w ** 2)
    return result

def logistic_regression(y,tx,initial_w, max_itters,gamma):
    print("Logistic regression using gradient descent or SGD")
    # "Logistic regression is the appropriate regression analysis to conduct when theession produces a logistic curve, which is limited to values between 0 and 1. Logistic regression is similar to a linear regression, but the curve is constructed using the natural logarithm of the “odds” of the target variable, rather than the probab dependent variable is dichotomous (binary)." - from the Internet
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24


    sigmoid = 1/(1+np.exp(-tx))

    w = initial_w

    for i in range(max_itters):
        pediction = np.dot(w*tx)
        activationFunction = sigmoid(pediction) # make between 0-1
        print(activationFunction)



def reg_logisitc_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    print("Regularized logistic regression using gradient descent or SGD")


DATA_TRAIN_PATH = '../data/train.csv'
y, tx, ids = load_csv_data(DATA_TRAIN_PATH)
w = np.zeros(30)

#losses, w_updated_mse = linear_regresssion_GD_mse(y, tx, w, 100, 0.2)
losses, w_updated_leastSquares = least_squares_GD(y, tx, w, 100, 0.2)

#print(w_updated_mse)
print(w_updated_leastSquares)

w_compare = [-2.80368247e+01,5.39471822e-01,-1.00522284e+01,1.13621369e+01,4.25777258e+02,5.05303694e+02,4.25229617e+02,-4.92012404e-01,3.50325763e+00,3.41860458e+01,2.13402302e-02,9.88750534e-02,4.25298113e+02,-5.73547481e+00,7.63626536e-03,1.38542432e-02,-1.44032246e+00,1.02196190e-02,-1.20750185e-02,3.44476629e+00,-3.26639200e-03,2.92616549e+01,5.18460080e-01,2.19436118e+02,2.03730018e+02,2.03732874e+02,4.43313020e+02,4.25186123e+02,4.25193362e+02,4.13618429e+01]


DATA_TEST_PATH = '../data/test.csv'
y_test, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)


y_pred = predict_labels(w_updated_leastSquares, tx_test)

create_csv_submission(ids_test,y_pred,"fyujn")