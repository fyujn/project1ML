# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import importlib

from proj1_helpers import *

#initial_w is the initial weight vector
#gamma is the step-size
#max_iters is the number of steps to run
#lambda_ is always the regularization parameter
#from scripts.proj1_helpers import *



def handleOutliers(tx):



    # think of mean or median
    meansPerColumn = np.mean(tx, axis=0)

    for i in range(30):
        for j in range(250000):
            if tx[j][i] == -999:5
                tx[j][i] = meansPerColumn[i]
    return tx

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

def least_squares_GD(y, tx, w, max_itters, gamma):#giving 0.745 acc
    print("calc Linear regression using gradient descent")
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    for n_iter in range(max_itters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        gradient = least_squares(y,tx)
       # ***************************************************
        # update w by gradient
        # ***************************************************
        w = w + gamma*gradient
    print(compute_loss_rmse(y,tx,w))
    print(compute_loss_mse(y,tx,w))
    print("Gradient Descent({bi}/{ti}): w={w}".format(bi=n_iter, ti=max_itters - 1, w=w))

    return w




def least_squares_SGD(y,tx,w,max_iters,gamma):  #giving 0.541	0.401	with 1000 iterations gamma = 0.2
    print("Calc Linear regression using stochastic gradient descent")
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    # ***************************************************
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        # Define parameters to store w and loss

        for n_iter in range(max_iters):
            # ***************************************************
            # compute gradient
            # ***************************************************
            gradient = least_squares(minibatch_y, minibatch_tx)
            # ***************************************************
            # update w by gradient
            # ***************************************************
            w = w + gamma * gradient

    print("Gradient Descent({bi}/{ti}): w={w}".format(bi=n_iter, ti=max_iters - 1, w=w))

    return w

def least_squares(y,tx):
    #print("Calc Least squares regression using normal equations")
    return np.linalg.inv(np.transpose(tx)@tx)@np.transpose(tx)@y

def ridge_regression(y,tx,lambda_):
    #print("Ridge regression using normal equations")
    """implement ridge regression."""
    # ***************************************************
    # ridge regression
    # ***************************************************
    l_= lambda_*2*30
    innerSum = np.linalg.inv(np.dot(np.transpose(tx), tx) + np.dot(l_, np.identity(30)))
    sumX = np.dot(innerSum,np.transpose(tx))
    w = np.dot(sumX, y)

    return w

def logistic_regression(y,tx,w, max_itters,gamma):
    print("Logistic regression using gradient descent or SGD")
    # "Logistic regression is the appropriate regression analysis to conduct when theession produces a logistic curve, which is limited to values between 0 and 1. Logistic regression is similar to a linear regression, but the curve is constructed using the natural logarithm of the “odds” of the target variable, rather than the probab dependent variable is dichotomous (binary)." - from the Internet
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24


    """
    ypred = predict_labels(w,tx)
    sigmoid = 1/(1+np.exp(-ypred))
    hoch = np.transpose(w)@np.transpose(tx)
    print(hoch)
    po = np.power(sigmoid,-hoch)

    print(po)
    mü = 1/(1+po)

    print("MÜ")
    print(mü)

    s = np.diag(mü)

    w = np.linalg.inv(np.transpose(tx)@s@tx)@np.transpose(tx)@(s@(tx@w+predict_labels(w,tx)-mü))
    """
    ypred = predict_labels(w,tx)
    sigmoid = 1/(1+np.exp(-y))


    for n_iter in range(max_itters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        print("Step0")
        z = np.dot(tx,w)
        print("Step1")
        a = np.linalg.inv(np.dot(z, np.transpose(tx)))
        print("Step2")
        b = a@np.transpose(tx)
        print("Step3")
        c = b @ w
        print("Step 4")
        d = d@y

        gradient = d
         # ***************************************************
        # update w by gradient
        # ***************************************************
        w = w + gamma*gradient
    return w





    for i in range(max_itters):
        pediction = np.dot(w*tx)
        activationFunction = sigmoid(pediction) # make between 0-1
        print(activationFunction)



def reg_logisitc_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    print("Regularized logistic regression using gradient descent or SGD")


def run_ridge(y,tx):
    """
    1.0120632391308362
    Lambda 32


    1.0117232823257554
    INDEX 6 (0.06)

    --> Best so far
    1.0110984126186728
    INDEX 11 (lambda = 0.55)


    --> new best
    1.0110746757782039
    INDEX 23 (lambda = 0.053)
    """
    losses = []
    ws = []
    for i in np.arange(0.5,0.6,0.01):
        w_new = ridge_regression(y,tx,i)
        loss = compute_loss_rmse(y, tx, w_new)
        losses.append(loss)
        ws.append(w_new)

    print(losses)
    print(" LOSS RMSE ")
    print(compute_loss_rmse(y,tx,w_new))
    minIndex = losses.index(min(losses))
    print("WEIGHT MATRIX " + str(ws[minIndex]))
    print("INDEX " + str(minIndex))
    plt.plot(losses)
    plt.show()
    return ws[minIndex]






def run():
    print("Load data ...")
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tx, ids = load_csv_data(DATA_TRAIN_PATH)
    w = np.zeros(30)

    print("Handle Outliers per mean ...")
    handleOutliers(tx)

    print("Run Method ...")
    #w_new = least_squares_GD(y, tx, w, 200, 1)
    #w_new = least_squares_SGD(y, tx, w, 100000, 0.01)
    #w_new = logistic_regression(y,tx,w,1000,0.2)
    w_new = run_ridge(y, tx);


    DATA_TEST_PATH = '../data/test.csv'


    print("Output Data ...")
    #execute file
    y_test, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

    y_pred = predict_labels(w_new, tx_test)
    create_csv_submission(ids_test, y_pred, "fyujn")

w_compare = [-2.80368247e+01,5.39471822e-01,-1.00522284e+01,1.13621369e+01,4.25777258e+02,5.05303694e+02,4.25229617e+02,-4.92012404e-01,3.50325763e+00,3.41860458e+01,2.13402302e-02,9.88750534e-02,4.25298113e+02,-5.73547481e+00,7.63626536e-03,1.38542432e-02,-1.44032246e+00,1.02196190e-02,-1.20750185e-02,3.44476629e+00,-3.26639200e-03,2.92616549e+01,5.18460080e-01,2.19436118e+02,2.03730018e+02,2.03732874e+02,4.43313020e+02,4.25186123e+02,4.25193362e+02,4.13618429e+01]
w_0745 = [1.60583532e-03,-1.44064742e-01,-1.21068034e-01,-1.09519870e-02-3.87743154e-01,9.46936127e-03,-5.20717046e-01,6.50207171e+00-7.61470193e-04,-5.45449842e+01,-4.42440406e+00,1.90157268e+00,1.28065548e+00,5.47101777e+01,-6.63603993e-03,-1.90865700e-02,5.48053124e+01,-1.06832978e-02,1.94699716e-02,7.38450105e-02,7.08974899e-03,-1.08668920e-02,-6.60896070e+00,-2.81600995e-02,1.66286578e-02,2.04234543e-02,-3.36094832e-02,-1.16732964e-01-2.22175995e-01,5.45541828e+01]



run()


