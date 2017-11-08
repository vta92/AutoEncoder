import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


#these files have no headers
movies=pd.read_csv('ml-1m/movies.dat', sep='::', header=None, encoding='latin-1', engine='python')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::', header=None, encoding='latin-1', engine='python')
users=pd.read_csv('ml-1m/users.dat', sep='::', header=None, encoding='latin-1', engine='python')

movies.columns=['Movie_ID','Name','Tags']
ratings.columns=['User_ID','Movie_ID','Rating','Timestamp']
users.columns=['User_ID','Gender','Age','Job_ID','Zip_Code']

train_set1 = pd.read_csv('ml-100k/u1.base',delimiter='\t')
test_set1 = pd.read_csv('ml-100k/u1.test', delimiter='\t')

train_set1.columns=['User_ID','Movie_ID','Rating','Timestamp']
test_set1.columns=['User_ID','Movie_ID','Rating','Timestamp']


#have to convert from pd to np array for processing later
train_set1 = np.array(train_set1,dtype='int')
test_set1 = np.array(test_set1,dtype='int')

#dropping the timestamps
train_set1= train_set1[:,:-1]
test_set1 = test_set1[:,:-1]

#assuming the max user id = total number of user in a given dataset, we have:
total_users = max(train_set1[:,0].max(), test_set1[:,0].max())
total_movies = max(train_set1[:,1].max(), test_set1[:,1].max())


#we have a 943x1682 matrix from the 2 variables above, with each cell contains a rating
#pytorch expects a list of list
def matrix_rep(data, total_users, total_movies):
    matrix=[[0 for i in range(total_movies)] for i in range(total_users)]
    
    for row in range(len(data)): #get the number of rows in the whole test set
        user_id, movie_id, rating = data[row][0], data[row][1], data[row][2]
        matrix[user_id-1][movie_id-1] = float(rating)
            
    return matrix

train_set1 = matrix_rep(train_set1, total_users, total_movies)
train_set1 = torch.FloatTensor(train_set1)

test_set1 = matrix_rep(test_set1, total_users, total_movies)
test_set1 = torch.FloatTensor(test_set1)
#end of processing
#########################################################################################################
class auto_encoder(nn.Module):
    def __init__(self, total_movies, hid1_size, hid2_size):
        super(auto_encoder, self).__init__()
        #first 2 layers vis->hid1, with the number of features or movies specified, 30 nodes in hid1 layer
        #we have in total of 3 hidden layers with in features and out features, with linear transformation + bias = true
        #goal: vis == decoding/outlayer
        self.vh1 = nn.Linear(total_movies,hid1_size)
        self.h1h2 = nn.Linear(hid1_size, hid2_size)
        self.h2h3 = nn.Linear(hid2_size, hid1_size) #stacked AE, needs to mirror the prev layer
        self.h3o = nn.Linear(hid1_size, total_movies)
        #activation function
        self.activation_func = nn.Tanh() #need to experiment to see the best act. func
    
    def prop_forward (self,vector):
        vector = self.activation_func(self.vh1(vector))
        vector = self.activation_func(self.h1h2(vector))
        vector = self.activation_func(self.h2h3(vector))
        vector = self.h3o(vector) #hope that the output matches the input
        return vector

def training(epochs, metrics, optimizer):
    global auto_encoder
    accuracy = list() #store the accuracy for graph        
    for epoch in range (epochs):
        training_loss = 0
        num_users_who_rated = 0.0
        
        for i in range(total_users):
            in_feature = train_set1[i] #create a fake dim of batchsize = 1 vector. Pytorch expects this
            in_feature = Variable(in_feature).unsqueeze(0)
            target = in_feature.clone() #the original input, unchanged. For loss calculation
            
            if torch.sum(target.data > 0) > 0: #make sure the input is not empty, ppl rate at least 1 movie
                out_feature = auto_encoder.prop_forward(in_feature)
                out_feature[target == 0] = 0 #unrated = can't count in loss function
                #don't want to compute grad w/r target
                target.require_grad = False
                
                loss = metrics(out_feature, target)
                mean_correction =  total_movies/float(torch.sum(target.data > 0)+1e-20) #only select movies with ratings, make sure denom != 0
                loss.backward() #gradient
                training_loss += np.sqrt(loss.data[0]*mean_correction)
                num_users_who_rated += 1.0
                optimizer.step() 
                #difference between backward() and step/opt: backward decides the direction of change, opt will determine the magnitude
        print('epoch ' + str(epoch) +': ' + str(training_loss/num_users_who_rated))
        accuracy.append(training_loss/num_users_who_rated)
    return accuracy

#test set has the solution to the unwatched movies in the training set. Therefore we still predict the training set.
#no need to worry about gradient/backprop as usual
def predict(metrics):
    global auto_encoder
    test_loss = 0
    counter = 0
    
    for i in range(total_users):
        in_feature = train_set1[i]
        in_feature = Variable(in_feature).unsqueeze(0)
        target = Variable(test_set1[i])
        
        if torch.sum(target.data > 0) > 0:
            result = auto_encoder.prop_forward(in_feature)
            target.require_grad = False
            result[target == 0] = 0
            
            loss = metrics(result, target)
            mean_correction =  total_movies/float(torch.sum(target.data > 0)+1e-20)
            test_loss += np.sqrt(loss.data[0]*mean_correction)
            counter += 1
        
    print('test mean loss' +': ' + str(test_loss/counter))
        
        


def graph(accuracy):
    x = [i for i in range(len(accuracy))]
    y = accuracy
    plt.plot(x,y)
    plt.ylabel("mean corrected training loss")
    plt.xlabel("epochs")
    plt.show()

    
        #we want the loss to be less than 1, so our error is less than 1 star rating
'''architecture of AE defined here'''
hid1_size = 30
hid2_size = 15
epochs = 30 #testing out with 300    
conditions = [total_movies, hid1_size, hid2_size]

auto_encoder = auto_encoder(int(conditions[0]), conditions[1], conditions[2])

metrics = nn.MSELoss(size_average=True)
optimizer = optim.Adam(auto_encoder.parameters(), lr=0.003, weight_decay = 0.2) #inherited the parameters() method from nn

train_accuracy = training(epochs, metrics, optimizer)

graph = graph(train_accuracy)

predict(metrics)









    
    
        
        