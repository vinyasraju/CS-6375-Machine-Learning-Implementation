
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import math
from math import exp
from operator import mul


# In[3]:

def removeNullFromData(dataframe):
    updated_dataframe = dataframe
    for i in range(len(dataframe)):
        for j in range(len(dataframe.iloc[i, :])):
            if(dataframe.iloc[i,j] == '?'):
                updated_dataframe = updated_dataframe.drop(dataframe.index[i])
                break

    return updated_dataframe


# In[4]:

def meanNormalize(array):
    mean = np.average(array)
    std_dev = np.std(array)
    normalizedArray = (array - mean)/std_dev
    return normalizedArray


# In[5]:

def standardizeData(dataframe_without_nulls):
    normalised_dataframe = dataframe_without_nulls
    for i in range(len(dataframe_without_nulls.columns)):
        data_type = dataframe_without_nulls.dtypes[i]
        if(data_type == np.int64 or data_type == np.float64):
            normalised_dataframe[i] = meanNormalize(dataframe_without_nulls[i])
    return normalised_dataframe


# In[6]:

def encodeLabels(normalised_dataframe):
    preprocessed_dataframe = normalised_dataframe
    for j in range(len(preprocessed_dataframe.columns)):
        data_type = preprocessed_dataframe.dtypes[j]
        if(data_type == np.object): 
            list_unique = preprocessed_dataframe[j].unique().tolist()
            for i in range(len(preprocessed_dataframe)):
                value = preprocessed_dataframe.iloc[i, j]
                label = list_unique.index(value)
                preprocessed_dataframe.iloc[i, j] = label
    return preprocessed_dataframe


# In[7]:

def normaliseClass(preprocessed_dataframe):
    j = len(preprocessed_dataframe.columns)-1
    preprocessed_dataframe[j]
    list_unique = preprocessed_dataframe[j].unique().tolist()

    a = list(list_unique)
    n = len(a)
    interval = 1/(n-1)
    b = a
    value = 0
    for i in range(len(a)):
        b[i] = value
        value = value + interval

    b = np.array(b)


    for i in range(len(preprocessed_dataframe)):
        value = preprocessed_dataframe.iloc[i, j]
        #print(value)
        label = b[list_unique.index(value)]
        preprocessed_dataframe.iloc[i, j] = label
        
    return preprocessed_dataframe


# In[8]:

def writeToFile(processed_dataframe, output_path):
    processed_dataframe.to_csv(output_path, sep=',',index= None, header=None)


# In[14]:

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#output_path = "irisdataset.csv"

url = input("Enter the URL of the dataset:")
output_path = input("Enter the output path for processed data:")


# In[15]:

dataframe = pd.read_csv(url, skipinitialspace=True, na_values='.', header = None)
dataframe_without_nulls = removeNullFromData(dataframe)
normalised_dataframe = standardizeData(dataframe_without_nulls)
preprocessed_dataframe = encodeLabels(normalised_dataframe)
processed_dataframe = normaliseClass(preprocessed_dataframe)
writeToFile(processed_dataframe, output_path)
print("Pre Processing has been done")


# In[16]:

writeToFile(processed_dataframe, output_path)


# In[ ]:




# In[17]:

#processing has been done, now model evaluation and testing


# In[ ]:




# In[18]:

#sigmoid activation function
#the condition is set to prevent overflow/underflow
def sigmoid(net):
    if(net > 50):
        net = 50
    elif(net<-50):
        net = -50
    return 1.0 / (1.0 + exp(-net))


# In[19]:

#function to train network
def trainNetwork(max_iterations, dataframe, list_weights, count_nodes, learning_rate):
    for n in range(max_iterations):    
        for i in range(len(dataframe)):
            input_vector = dataframe.iloc[i,:-1]
            input_vector = list(input_vector)
            input_vector.insert(0, 1)
            input_vector = np.array(input_vector)        
            target = dataframe.iloc[i,-1]
            list_sigmoid, output = forward_propagation(list_weights, input_vector, count_nodes)
            list_weights = back_propagation(list_weights, input_vector, list_sigmoid, count_nodes, learning_rate, target)
    return list_weights


# In[20]:

#create a network with randomly initialised weights
def createWeights(count_input_nodes, count_hidden_layers, count_hidden_nodes):
    count_hidden_nodes.insert(0, count_input_nodes)
    count_nodes = count_hidden_nodes
    list_weights = list()
    
    for i in range(count_hidden_layers):
        j = i + 1
        W = np.matrix(np.random.uniform(-1,1, size=(count_nodes[j]-1, count_nodes[i])))
        list_weights.append(W.tolist())

    i = i + 1
    j = i + 1
    W = np.matrix(np.random.uniform(-1,1, size=(count_output_nodes, count_nodes[i])))
    list_weights.append(W.tolist())
    return list_weights, count_nodes


# In[21]:

#calculate and return the accuracy of the model
def getAccuracy(dataframe, list_weights, count_nodes):
    j = len(dataframe.columns)-1
    dataframe[j]
    list_unique = dataframe[j].unique().tolist()
    list_unique.sort()
    num_corrects = 0
    for i in range(len(dataframe)):
        input_vector = dataframe.iloc[i,:-1]
        input_vector = list(input_vector)
        input_vector.insert(0, 1)
        input_vector = np.array(input_vector)
        target = dataframe.iloc[i,-1]
        list_sigmoid, output = forward_propagation(list_weights, input_vector, count_nodes)
        label  = findLabel(list_unique, output[0])
        if(label == target):
            num_corrects = num_corrects+1
    accuracy = num_corrects/len(dataframe)
    return accuracy


# In[22]:

#forward pass
def forward_propagation(list_weights, input_vector, count_nodes):
    list_sigmoid = list()
    
    for i in range(len(count_nodes)):
        temp = 0
        sigmoid_at_level = list()
        sigmoid_at_level.append(1)
        if(i==0):
            vector = input_vector
        else:
            vector
            
        temp = np.array(vector)*list_weights[i]
        net = list()
        for j in range(len(temp)):
            net.append(sum(temp[j]))
        
        for j in range(len(net)):
                sigmoid_at_level.append(sigmoid(net[j]))
            
        list_sigmoid.append(sigmoid_at_level)
        vector = sigmoid_at_level
    
    output = sigmoid_at_level[1:]
    return list_sigmoid, output


# In[23]:

#to find closest label
def findLabel(array,value):
    index = np.searchsorted(array, value, side="left")
    if index > 0 and (index == len(array) or math.fabs(value - array[index-1]) < math.fabs(value - array[index])):
        return array[index-1]
    else:
        return array[index]


# In[24]:

#back propagation

def back_propagation(list_weights, input_vector, list_sigmoid, count_nodes, learning_rate, target):
    #print("Back Prop")
    output = list_sigmoid[-1][1]
    x = len(count_nodes)
    
    list_delta = list()
    
    for i in range(len(count_nodes), 0, -1):
        delta_at_level = list()
        delta_weights_level = 0
        if(i==len(count_nodes)):
            temp = (target - output)*output*(1-output)
            delta_at_level.insert(0, temp)
            list_delta.insert(0, delta_at_level)
        else:
            delta_at_level = np.array(list_sigmoid[i-1]) * (1 - np.array(list_sigmoid[i-1])) * np.array(list_delta[x-i-1]) * np.array(list_weights[i])
            delta_weights_level = learning_rate * list_delta[x-i-1][0] * np.array(list_sigmoid[i-1])
            list_delta.insert(0, delta_at_level.tolist())
            mytemp = list_weights[i][0] + delta_weights_level
            list_weights[i][0] = list(mytemp)
            
    delta_weights_level = learning_rate * np.matrix(list_delta[0][0][1:]).T * np.matrix(input_vector)
    delta_weights_level = delta_weights_level.tolist()
    
    a = np.matrix(delta_weights_level)
    b = np.matrix(list_weights[0])
    c = np.add(a,b)
    temp = c.tolist()
    list_weights[0] = temp
    
    return list_weights


# In[25]:

def printWeights(list_weights, num_layers):
    for i in range(num_layers):
        print("\n\nLayer ", i)
        list_weights_layer = list_weights[i]
    
        for k in range(len(list_weights_layer[0])):
            print("\n\tNeuron", k, "weights:\n")
            for j in range(len(list_weights_layer)):
                print("\t\t",list_weights_layer[j][k])


# In[27]:


#Following input is required from user

path_input_dataset = input("Enter the input Dataset: ")
dataframe = pd.read_csv(path_input_dataset, header=None)
dataframe = dataframe.sample(frac=1)
training_percent = int(input("Enter the training percent: "))

max_iterations = int(input("Enter the maximum iteration:"))
learning_rate = 0.9

count_features = len(dataframe.columns)-1+1 # -1 for the class label, +1 for bias

count_hidden_layers = int(input("Enter the number of hidden layer:"))

count_hidden_nodes = list()

for i in range(0, count_hidden_layers):
    value = int(input("Enter the nodes in hidden layers: "))
    count_hidden_nodes.append(value+1)
    
count_input_nodes = count_features
count_weight_matrices = count_hidden_layers + 1
count_output_nodes = 1

num_examples = len(dataframe)
index = int(training_percent * num_examples / 100)
training_dataset = dataframe[0:index]
testing_dataset = dataframe[index:]

list_weights, count_nodes = createWeights(count_input_nodes, count_hidden_layers, count_hidden_nodes)


# In[29]:

list_weights = trainNetwork(max_iterations, training_dataset, list_weights, count_nodes, learning_rate)


# In[30]:

training_accuracy = getAccuracy(training_dataset, list_weights, count_nodes)
print("Traing Accuracy:", training_accuracy*100)


# In[31]:

testing_accuracy = getAccuracy(testing_dataset, list_weights, count_nodes)
print("Testing Accuracy:", testing_accuracy*100)


# In[34]:

training_error = 1 - training_accuracy
testing_error = 1 - testing_accuracy
print("Training Error:", training_error*100)
print("Testing Error:", testing_error*100)


# In[35]:

num_layers = len(list_weights)
printWeights(list_weights, num_layers)


# In[ ]:



