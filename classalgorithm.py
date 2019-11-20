from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        """
        X: shape of Nx100x100 np array
        Y: shape of Nx100 np array
        """
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].view(1,100,100), self.Y[idx].view(1)


class CNN(nn.Module):
    """
    Input shape batches x channels x height x width (i.e. n x 1 x 100 x 100)
    Output shape is n x 1
    """
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(5, 16, 5)
        self.fc1 = nn.Linear(16*23*23, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 5 x 100 x 100
        x = self.pool(x)            # 5 x 50 x 50
        x = F.relu(self.conv2(x))   # 16 x 46 x 46
        x = self.pool(x)            # 16 x 23 x 23
        x = x.view(-1, 16*23*23)   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x

class CNN_Class(Classifier):
    """
    Convolutional neural net, takes in 100 words at a time and out put 0 or 1
    """

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01, "epochs": 5, "bSize":30, "stepsize":0.001}
        self.reset(parameters)
        # self.net = CNN()
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.net = CNN()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.params["stepsize"], momentum=0.9)
        self.accuracy = 0.0


    def createDataset(self, X,Y):
        x_tensor = torch.from_numpy(X)
        x_tensor = x_tensor.view(-1, 1, 100, 100)
        y_tensor = torch.from_numpy(Y).view(-1)
        return torch.utils.data.TensorDataset(x_tensor,y_tensor)


    def learn(self, Xtrain, ytrain, Xval, yval):
        """ Learns using the traindata """
        trainSet = self.createDataset(Xtrain, ytrain)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=self.params["bSize"], shuffle=True, num_workers=2)
        valSet = self.createDataset(Xval, yval)
        valLoader = torch.utils.data.DataLoader(valSet, batch_size=self.params["bSize"], shuffle=True, num_workers=2)

        print("start learning")
        for epoch in range(self.params["epochs"]):
            running_loss = 0.0
            for i, data in enumerate(trainLoader):
                # get data
                inputs, labels = data
                print(inputs.shape,flush=True)
                print(labels.shape,flush=True)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # train
                outputs = self.net(inputs)
                print(1,flush=True)
                loss = self.criterion(outputs, labels)
                print(2,flush=True)
                loss.backward()
                print(3,flush=True)
                self.optimizer.step()
                print(4,flush=True)

                running_loss += loss.item()
                print(".", end="",flush=True)
                if i%100 == 99:
                    print("[%d %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print("done training.")
        print("start validating")
        wrong = 0
        for data in valLoader:
            inputs, labels = data
            outputs = self.net(inputs)
            outputs[outputs>0.5] = 1
            outputs[outputs<=0.5] = 0
            wrong +=torch.sum( abs(labels-outputs))
        
        self.accuracy = 1 - wrong / len(valSet)

    # def predict(self, Xtest):
    #     ytest = np.dot(Xtest, self.weights)
    #     ytest[ytest > 0] = 1
    #     ytest[ytest < 0] = 0
    #     return ytest

    def get_accuracy(self):
        return self.accuracy

    def get_weights(self):
        return self.net.state_dict()