import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pandas as pd

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
        vector = self.activation_func(self.h1h2(xformation))
        vector = self.activation_func(self.h2h3(xformation))
        vector = self.h3o(xformation) #hope that the output matches the input
        return vector

'''architecture of AE defined here'''
hid1_size = 20
hid2_size = 10    
conditions = [total_movies, hid1_size, hid2_size]

auto_encoder = auto_encoder(int(conditions[0]), conditions[1], conditions[2])

metrics = nn.MSELoss(size_average=True)
optimizer = optim.Adam(auto_encoder.parameters(), lr=0.002, weight_decay = 0.3) #inherited the parameters() method from nn













    
    
        
        