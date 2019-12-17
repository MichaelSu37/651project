from __future__ import division  # floating point division
import csv
import random
import math
import pickle, os

import numpy as np
#import utilities as utils
#import dataloader as dtl
# import classalgorithm as algs
import matplotlib.pyplot as plt
# from w2v import getSample


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
    test = True
    plot = True
    plot_len = 0
    numruns = 1
    k_fold = True
    K = 3
    dataset_file = "dataset_tr_te.pkl"

    classalgs = {
        #'Logistic Regression': algs.LogitReg(),
        'CNN': CNN_Class(),
    }
    numalgs = len(classalgs)

    parameters = (
        #{'regularizer': 'None', 'stepsize':0.1, 'num_steps':300, 'batch_size':20},
        {'regularizer': 'None', 'stepsize':0.01, 'epochs':20, 'bSize':20, 'hidden': 200},
    )
    numparams = len(parameters)

    accuracy = {}
    for learnername in classalgs:
       accuracy[learnername] = np.zeros((numparams, numruns, K, 100000))

    # load dataset & shuffle 
    # trainX, testX = pickle. load(open(dataset_file, "rb"))

    # process data
    trainDatafile = '/content/drive/My Drive/651Project/dataAfterSplit/train_balanced.pkl'
    testDatafile = '/content/drive/My Drive/651Project/dataAfterSplit/test.pkl'
    valDatafile = '/content/drive/My Drive/651Project/dataAfterSplit/val.pkl'

    if (os.path.exists(trainDatafile) and os.path.exists(testDatafile) and os.path.exists(valDatafile)):
        with open(trainDatafile, 'rb') as f:
            trainX, trainY = pickle.load(f)
        with open(testDatafile, 'rb') as f:
            testX, testY = pickle.load(f)
        with open(valDatafile, 'rb') as f:
            valX, valY = pickle.load(f)
    elif (os.path.exists('train.pkl') and os.path.exists('test.pkl') and os.path.exists('val.pkl')):
        with open('train.pkl', 'rb') as f:
            trainX, trainY = pickle.load(f)
        with open('test.pkl', 'rb') as f:
            testX, testY = pickle.load(f)
        with open('val.pkl', 'rb') as f:
            valX, valY = pickle.load(f)

    else:
        trainX, trainY = getSample('train')
        with  open('train.pkl', 'wb') as f:
            pickle.dump((trainX, trainY), f)

        testX, testY = getSample('test')
        with  open('test.pkl', 'wb') as f:
            pickle.dump((testX, testY), f)

        valX, valY = getSample('validate')
        with  open('val.pkl', 'wb') as f:
            pickle.dump((valX, valY), f)
    
    # randomize training data
    # np.random.seed(1)
    # np.random.shuffle(trainX)
    # np.random.seed(1)
    # np.random.shuffle(trainY)

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
                        # Reset learner for new parameters
                        learner.reset(params)
                        print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                        # Train model
                        learner.learn(k_trainX, k_trainY, k_valX, k_valY)
                        # get error
                        temp_acc = learner.get_accuracy()
                        plot_len = len(temp_acc)
                        accuracy[learnername][p, r, counter_k, :plot_len] = np.add(accuracy[learnername][p, r, counter_k, :plot_len],temp_acc)
                        # get weights
                        weights_k.append(learner.get_weights())
                        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # learner.test(testX, testY)
                        break

                    best_accuracy_over_K = np.argmax(accuracy[learnername][p, r])
                    # best_weights_over_K = weights_k[best_accuracy_over_K]
                    # weights.append((best_weights_over_K, accuracy[learnername][p, r, best_accuracy_over_K]))
                    print("best_accuracy_over_K: ", best_accuracy_over_K)
                    
                    # Test model
                    
                    # predictions = learner.predict(testX)
                    # acc = utils.getaccuracy(testY, predictions)
                    # print ('accuracy for ' + learnername + ': ' + str(acc))
                    # accuracy[learnername][p,r] = acc
    # Save best weights
    # pickle.dump(weights, open("weights.pkl", "wb"))

    # test accuracy for files
    if test:
        testXfile = '/content/drive/My Drive/651Project/dataAfterSplit/reformed.pkl'
        testYfile = '/content/drive/My Drive/651Project/dataAfterSplit/ylabel.pkl'
        with open(testXfile, 'rb') as f:
            testX = pickle.load(f)
        with open(testYfile, 'rb') as f:
            testY = pickle.load(f)
        predict_file = np.zeros(len(testX))
        print("--------------test on",len(testX),"data-----------------")

        for i in range(len(testX)):
            result = learner.predict(testX[i])
            result = torch.sum(result)
            print(result)
            # the file has multiple writer
            if result > 0:
                predict_file[i] = 1
        wrong = 0
        for i in range(len(testX)):
            wrong += abs(testY[i]-predict_file[i])
            print(testY[i],predict_file[i])
            acc = 1 - wrong / len(testX)
        print(acc)


    # plot
    if plot == True:
        print("PLOT CNN Result!")
        #num_epochs = 300
        for learnername, learner in classalgs.items():
            accuracy_plot = accuracy[learnername][0,0,0,:plot_len]
            epi = np.arange(0, plot_len, 1)
            plt.plot(epi, accuracy_plot, label = 'CNN')
        
        # for p in range (numparams):
        #     temp_acc = np.zeros(K)
        #     for r in range (numruns):
        #         print("accuracy :", accuracy['CNN'][p,r])
        #         temp_acc += accuracy['CNN'][p,r]
        #     plt.plot(epi, temp_acc, label = 'validation accuracy p='+str(p))
                
        #epi = np.arange(0, num_epochs, 1)
        #plt.plot(epi,accuracy_val, label='validation accuracy')
        #plt.plot(epi,accuracy_test, label='test accuracy')
        #plt.plot(epi,accuracy_train, label='train accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Sigmoid Cross Entropy Loss') 
        plt.legend()    


def main():
    classify()
    
    print("--------------done!-----------------")
    
    
main()
