from __future__ import division  # floating point division
import csv
import random
import math
import pickle

import numpy as np
#import utilities as utils
#import dataloader as dtl
import classalgorithm as algs
import matplotlib.pyplot as plt
from w2v import getSample


def get_k_fold_data(k, K, trainX, trainY):
    # check variable
    Xlen = len(trainX)
    Ylen = len(trainY)
    assert k<K and Xlen == Ylen

    # calculate index
    k_Xlen = Xlen/K
    start_pos = int(k_Xlen*k)
    if (k+1 == K):
        end_pos = Xlen+1
    end_pos = int(k_Xlen*(k+1))
    # get training set and validation set
    k_valX = trainX[start_pos:end_pos]
    k_valY = trainY[start_pos:end_pos]
    k_trainX = np.append(trainX[:start_pos],trainX[end_pos:Xlen+1],axis=0)
    k_trainY = np.append(trainY[:start_pos],trainY[end_pos:Xlen+1],axis=0)

    return k_trainX, k_trainY, k_valX, k_valY
    


def classify():
    # init variables
    run = True
    plot = False
    numruns = 1
    k_fold = True
    K = 3
    dataset_file = "dataset_tr_te.pkl"

    classalgs = {
        #'Logistic Regression': algs.LogitReg(),
        'Neuron Network': algs.CNN_Class,
    }
    numalgs = len(classalgs)

    parameters = (
        #{'regularizer': 'None', 'stepsize':0.1, 'num_steps':300, 'batch_size':20},
        {'regularizer': 'None', 'stepsize':0.1, 'num_steps':300, 'batch_size':20, 'hidden': 200},
    )
    numparams = len(parameters)

    accuracy = {}
    #for learnername in classalgs:
    #    accuracy[learnername] = np.zeros((numparams, numruns, K))

    # load dataset & shuffle 
    # trainX, testX = pickle. load(open(dataset_file, "rb"))
    trainX, trainY = getSample('train')
    testX, testY = getSample('test')
    valX, valY = getSample('validate')
    
    np.random.seed(1)
    np.random.shuffle(trainX)
    np.random.seed(1)
    np.random.shuffle(trainY)

    weights = []
    
    # Run learning algorithm
    if run:
        for r in range(numruns):
            print(('Running on train={0}, val={1}, test={2} samples for run {3}').
                  format(trainX.shape[0], valX.shape[0], testX.shape[0], r))
    
            # test different parameters 
            for p in range(numparams):
                params = parameters[p]
                
                # run every algorithm
                for learnername, learner in classalgs.items():
                    
                    weights_k = []

                    # run K-fold algorithm
                    for counter_k in range(K):
                        k_trainX, k_trainY, k_valX, k_valY = get_k_fold_data(counter_k, K, trainX, trainY)
                        print(k_trainX[0],k_trainY[0])
                        print(k_trainX[10],k_trainY[10])
                        print(k_trainX[20],k_trainY[20])
                        # Reset learner for new parameters
                        learner.reset(params)
                        print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                        # Train model
                        learner.learn(k_trainX, k_trainY, k_valX, k_valY)
                        # get error
                        accuracy[learnername][p, r, counter_k] = learner.get_accuracy()
                        # get weights
                        weights_k.append(learner.get_weights())

                    best_accuracy_over_K = np.argmax(accuracy[learnername][p, r])
                    best_weights_over_K = weights_k[best_accuracy_over_K]
                    weights.append((best_weights_over_K, accuracy[learnername][p, r, best_accuracy_over_K]))
                    print("best_accuracy_over_K: ", best_accuracy_over_K)
                    
                    # Test model
                    # predictions = learner.predict(testX)
                    # acc = utils.getaccuracy(testY, predictions)
                    # print ('accuracy for ' + learnername + ': ' + str(acc))
                    # accuracy[learnername][p,r] = acc
    # Save best weights
    pickle.dump(weights, open("weights.pkl", "wb"))

    # plot
    if plot == True:
        print("PLOT!")
        num_epochs = 300
        accuracy_val, accuracy_test, accuracy_train, best_accuracy, best_weight = pickle. load(  open("learning_acc.pkl", "rb")) 
        print("best_accuracy :", best_accuracy)
        epi = np.arange(0, num_epochs, 1)
        plt.plot(epi,accuracy_val, label='validation accuracy')
        #plt.plot(epi,accuracy_test, label='test accuracy')
        plt.plot(epi,accuracy_train, label='train accuracy')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy %') 
        plt.legend()    


def main():
    classify()
    
    print("--------------done!-----------------")
    
    
main()
