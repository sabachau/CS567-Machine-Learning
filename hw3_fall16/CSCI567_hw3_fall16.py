import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import sys
sys.path.append('./libsvm-3.21/python/')
from svmutil import *
import time
import math


# def epsilon():
#     # return np.exp(-1*pow(x,2)/(0.02))/math.sqrt(0.2*math.pi)
#     mu, sigma = 0, 0.1  # mean and standard deviation
#     s = mu, sigma, 1000)
#     # print s
#     return s

def plot(data,title):
    for key in data:
        plt.hist(data[key], bins=10)
        plt.title(title+"\nFunction g" +str(key)+"(x)")
        plt.show()

def calc_target(x):
    target = []
    for x_i in x:
        target.append(2*(x**2)+np.random.normal(0,0.1,1))
    return target

def generate_theta_matrix(n):
    a = np.zeros((n,n), int)
    np.fill_diagonal(a, 1)
    a[0][0] = 0
    return a

def get_optimal_theta(X, Y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)

def getThetaRegularized(X,Y,lamda,num_parameters):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)+lamda*generate_theta_matrix(num_parameters)), X.T), Y)


def standardize_data(train_data):
    return (train_data-np.mean(train_data,0))/np.std(train_data,axis=0)

def generateHypothesis(theta,data):
    return np.sum(theta*data,1)

def get_MSE(htheta, target):
    return (1/float(htheta.shape[0]))*(np.sum(np.square(htheta-target)))
def get_bias_sq(hypothesis,target):
    return np.square(hypothesis-target)

def generate_gx(x,true_y):
    gx=[1]
    final_theta_vector = []
    for i in range(5):
        temp = 0
        feature_vector = []
        for index,j in enumerate(range(i+1)):
            feature_vector.append(x**j)
        theta = get_optimal_theta(feature_vector,true_y)
        gx.append(generateHypothesis(theta,feature_vector))
        final_theta_vector.append(theta)
    get_MSE(gx,true_y)

    print 'gx: ',gx
    return gx

def generate_feature_vector(x,num):
    feature_vec = []
    print feature_vec
    for i in range(num):
        feature_vec.append(x**i)
    return feature_vec


def generate_dataset(row, col):
    num_size = row * col
    dataset = np.random.uniform(-1, 1, num_size).reshape(num_size, 1)  # .reshape((100,10))
    target_Matrix = np.asarray(map(lambda x: 2 * (x ** 2) + np.random.normal(0, 0.1, 1), dataset))
    target_Matrix = np.ndarray.flatten(target_Matrix)
    # target_Matrix = np.split(target_Matrix, row)
    X_Matrix = np.ones([num_size, 1])
    for i in range(0, 5):
        X_Matrix = np.append(X_Matrix, np.power(dataset, i), axis=1)
    return dataset, target_Matrix, X_Matrix

def dataset_size(row,col):
    dataset,target_Matrix,X_Matrix = generate_dataset(row,col)
    target_Matrix = np.split(target_Matrix, row)
    X_Matrix = np.split(X_Matrix,row)
    mse_dict = {}
    mse_avg_dict = {}
    theta_dict = {}
    print '*****AVG MEAN SQUARE ERROR ON TRAINING SET*****\ng-function:\n'
    for i in range(6):
        # print 'g-function:i = ',i
        thetaset = []
        mseset = []
        for dataset,targetset in zip(X_Matrix,target_Matrix):
            if i==0:
                extracted_cols = dataset[:,[0]]
                theta = 1
            else:
                extracted_cols = dataset[:, range(1, i + 1)]
                theta = get_optimal_theta(extracted_cols,targetset)
            thetaset.append(theta)
            hypothesis = generateHypothesis(theta, extracted_cols)
            mse = get_MSE(hypothesis, targetset)
            mseset.append(mse)
        theta_dict[i] = thetaset
        mse_dict[i] = mseset
        mse_avg_dict[i] = np.divide(np.sum(mseset),row)
        print 'i = ',i+1,':', mse_avg_dict[i]
    # plot(mse_dict)
    testset, test_target_Matrix, test_X_Matrix = generate_dataset(row, col)
    avg_theta_diction = {}
    bias = {}
    bias_dict = {}
    gbar = {}
    print '\n*****BIAS_SQUARE ON TEST SET*****\ng-function:\n'
    for i in range(6):
        theta_vector = theta_dict[i]
        sum_theta_vector = sum(theta_vector)
        avg_thetas = sum_theta_vector/row
        numpy_avg_theta = np.asarray(avg_thetas)
        avg_theta_diction[i] = numpy_avg_theta
        if i == 0:
            extracted_cols = test_X_Matrix[:, [0]]
        else:
            extracted_cols = test_X_Matrix[:, range(1, i + 1)]
        test_hypothesis = generateHypothesis(extracted_cols, numpy_avg_theta)
        gbar[i] = numpy_avg_theta
        bias_sq = get_bias_sq(test_hypothesis, test_target_Matrix)
        bias[i] = bias_sq
        bias_dict[i] = np.divide(np.sum(bias_sq),row*col)
        print 'i = ',i+1,':', bias_dict[i]

    print '\n*****VARIANCE ON TEST SET*****\ng-function:\n'
    test_target_Matrix = np.split(test_target_Matrix, 100)
    test_X_Matrix = np.split(test_X_Matrix,row)
    variance_list =[]
    variance_dict = {}
    for i in range(6):
        theta_g_bar = gbar[i]
        theta_vector = theta_dict[i]
        for dataset, targetset,theta_per_dataset in zip(test_X_Matrix, test_target_Matrix, theta_vector):
            if i == 0:
                extracted_cols = dataset[:, [0]]
            else:
                extracted_cols = dataset[:, range(1, i + 1)]
            gDx = generateHypothesis(theta_per_dataset,extracted_cols)
            gBarX = generateHypothesis(theta_g_bar,extracted_cols)
            diff = np.subtract(gDx,gBarX)
            squared = np.square(diff)
            expectation = np.sum(squared)/col
            variance_list.append(expectation)
        variance_dict[i] = np.divide(np.sum(variance_list),row)
        print 'i = ', i + 1, ':', variance_dict[i]

def regularize(row,col,equation_no):

    dataset, target_Matrix, X_Matrix = generate_dataset(row, col)
    target_Matrix = np.split(target_Matrix, row)
    X_Matrix = np.split(X_Matrix, row)
    for lamda in [0.001,0.003,0.01,0.03,0.1,0.3,1.0]:
        print '\n\n=>LAMBDA: ',lamda

        i=2
        thetaset = []
        for dataset, targetset in zip(X_Matrix, target_Matrix):
            extracted_cols = dataset[:, range(1, i + 1)]
            theta = getThetaRegularized(extracted_cols, targetset,lamda,i)
            thetaset.append(theta)
        testset, test_target_Matrix, test_X_Matrix = generate_dataset(row, col)
        avg_theta_diction = {}
        bias = {}
        bias_dict = {}
        gbar = 0
        print '\nBias_square: ',
        i=2
        sum_theta_vector = sum(thetaset)
        avg_thetas = sum_theta_vector / row
        numpy_avg_theta = np.asarray(avg_thetas)
        avg_theta_diction[i] = numpy_avg_theta
        extracted_cols = test_X_Matrix[:, range(1, i + 1)]
        test_hypothesis = generateHypothesis(extracted_cols, numpy_avg_theta)
        gbar = numpy_avg_theta
        bias_sq = get_bias_sq(test_hypothesis, test_target_Matrix)
        bias[i] = bias_sq
        bias_dict[i] = np.divide(np.sum(bias_sq), row * col)
        print bias_dict[i]

        print '\nVariance: ',
        test_target_Matrix = np.split(test_target_Matrix, 100)
        test_X_Matrix = np.split(test_X_Matrix, row)
        variance_list = []
        variance_dict = {}
        i=2
        theta_g_bar = gbar
        for dataset, targetset in zip(test_X_Matrix, test_target_Matrix):
            extracted_cols = dataset[:, range(1, i + 1)]
            gDx = generateHypothesis(thetaset, extracted_cols)
            gBarX = generateHypothesis(theta_g_bar, extracted_cols)
            diff = np.subtract(gDx, gBarX)
            squared = np.square(diff)
            expectation = np.sum(squared) / col
            variance_list.append(expectation)
        variance_dict[i] = np.divide(np.sum(variance_list), row)
        print variance_dict[i]

def kernel():
    num_list = range(29)
    transformtobinarylist = [1, 6, 7, 13, 14, 25, 28]
    threevectors = [x for x in num_list if x not in transformtobinarylist]
    # (-1,0,1)->0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27

    ourTrainingData = pd.read_csv("phishing-train-features.txt", sep="\t", header=None)
    binary_vec = ourTrainingData.iloc[:, transformtobinarylist]
    three_vec = ourTrainingData.iloc[:, threevectors]
    three_vec = three_vec.replace(-1, 0)
    f_space = pd.concat([pd.get_dummies(binary_vec[feature]) for feature in binary_vec], axis=1,
                        keys=binary_vec.columns)
    concatenated_result = pd.concat([f_space, three_vec], axis=1)

    # linear SVM
    labels_train = pd.read_csv("phishing-train-label.txt")
    labels_train = labels_train.transpose()
    train_mat = scipy.io.loadmat('phishing-train.mat')
    train_target = train_mat['label']
    y_training = train_target.flatten()
    x_training = concatenated_result.values.astype(float).tolist()
    problem_sttmnt = svm_problem(y_training, x_training)

    print 'Cross Validation=3'
    for i in range(-6,3):
        c_param = math.pow(4,i)
        print 'C = ', str(c_param)
        parameter = svm_parameter('-t 0 -v 3 -c ' + str(c_param) + ' -q')
        startTime = time.time()
        model = svm_train(problem_sttmnt, parameter)
        endTime = time.time()
        time_taken = endTime - startTime
        print 'running time : ', str(time_taken), 'sec'

    print 'Kernel SVM - polynomial'
    for i in range(-3, 8):
        c_param = math.pow(4, i)
        print 'C = ', str(c_param)
        for deg in range(1, 4):
            print 'degree = ', deg
            parameter = '-t 1 -v 3 -d ' + str(deg) + ' -c ' + str(c_param)
            startTime = time.time()
            this_model = svm_train(problem_sttmnt, parameter)
            endTime = time.time()
            time_taken = endTime - startTime
            print 'running time : ', time_taken, 'sec'

    print 'Kernel SVM - RBF'
    for i in range(-3, 8):  # C_values_poly_Kernel:
        c_param = math.pow(4, i)
        print' C = ', c_param
        for j in range(-7, 0):
            gamma_param = math.pow(4, j)
            parameter = '-t 2 -v 3 -g ' + str(gamma_param) + ' -c ' + str(c_param)
            print 'gamma = ', gamma_param
            startTime = time.time()
            this_model = svm_train(problem_sttmnt, parameter)
            endTime = time.time()
            time_taken = endTime - startTime
            print 'running time:', time_taken, 'sec'


def main():
    print '\n\n################dataset of 100x10################'
    dataset_size(100,10)
    print '\n################dataset of 100x100################'
    dataset_size(100,100)
    print '\n################ Regularization ################'
    regularize(100,100,2)
    print'\nKERNEL'
    kernel()

if __name__ == '__main__':
    main()