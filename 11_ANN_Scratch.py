import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import expit
import statistics
from sklearn.model_selection import train_test_split

##############################################################################################
###################################### Preparing Dataset #####################################
##############################################################################################

data = pd.read_csv('letter-recognition.csv',header=None)
attributes = pd.read_csv('Attributes.csv',header=None)
attributes_names = []
for i in range(0,len(attributes[0])):
    attributes_names.append(attributes[0][i])

data.columns = attributes_names

X = data.iloc[:,1:]
X = X.values
Y = data.iloc[:,0]

temp = []
for i in range(len(Y)):
    val = Y[i]
    val = ord(val) - 65
    temp.append(val)
Y = np.array(temp)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.85)

print("Training Size :- ",len(X_train),"Test Size :- ",len(X_test))
X_train = X_train.T
X_test = X_test.T



##############################################################################################
######################################## IMPORTANT FUNCTIONS #################################
##############################################################################################

def Tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.maximum(x,0)

def derivative_ReLU(x):
    return x > 0

def derivative_Tanh(x):
    return (1 - Tanh(x)*Tanh(x))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 25 + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def softmax(Z):
    A = expit(Z) / sum(expit(Z))
    return A

def CrossEntropyLoss(A,one_hot_Y):
    return -np.mean(one_hot_Y * np.log(A + 1e-8))
    
aplha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

aplha = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
def AdamDemo(W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO,
                        dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO,learning_rate,
                                    v_dW1,v_dW2,v_dW3,v_dW4,v_dW5,v_dW6,v_dW7,v_dW8,v_dWO,
                                    v_db1,v_db2,v_db3,v_db4,v_db5,v_db6,v_db7,v_db8,v_dbO,
                                    s_dW1,s_dW2,s_dW3,s_dW4,s_dW5,s_dW6,s_dW7,s_dW8,s_dWO,
                                    s_db1,s_db2,s_db3,s_db4,s_db5,s_db6,s_db7,s_db8,s_dbO):
            
        ################# Adding Momentum to Weights ################# 
    v_dW1 = beta1*v_dW1 + (1 - beta1)*dW1
    v_dW2 = beta1*v_dW2 + (1 - beta1)*dW2
    v_dW3 = beta1*v_dW3 + (1 - beta1)*dW3
    v_dW4 = beta1*v_dW4 + (1 - beta1)*dW4
    v_dW5 = beta1*v_dW5 + (1 - beta1)*dW5
    v_dW6 = beta1*v_dW6 + (1 - beta1)*dW6
    v_dW7 = beta1*v_dW7 + (1 - beta1)*dW7
    v_dW8 = beta1*v_dW8 + (1 - beta1)*dW8
    v_dWO = beta1*v_dWO + (1 - beta1)*dWO

        ################ Adding Momentum to biases ###################
    v_db1 = beta1*v_db1 + (1 - beta1)*db1
    v_db2 = beta1*v_db2 + (1 - beta1)*db2
    v_db3 = beta1*v_db3 + (1 - beta1)*db3
    v_db4 = beta1*v_db4 + (1 - beta1)*db4
    v_db5 = beta1*v_db5 + (1 - beta1)*db5
    v_db6 = beta1*v_db6 + (1 - beta1)*db6
    v_db7 = beta1*v_db7 + (1 - beta1)*db7
    v_db8 = beta1*v_db8 + (1 - beta1)*db8
    v_dbO = beta1*v_dbO + (1 - beta1)*dbO



        #^^^^^^^^^^^^^^^^^ Adding RMS to Weights ^^^^^^^^^^^^^^^^^^ 
    s_dW1 = beta2*s_dW1 + (1 - beta2)*(dW1**2)
    s_dW2 = beta2*s_dW2 + (1 - beta2)*(dW2**2)
    s_dW3 = beta2*s_dW3 + (1 - beta2)*(dW3**2)
    s_dW4 = beta2*s_dW4 + (1 - beta2)*(dW4**2)
    s_dW5 = beta2*s_dW5 + (1 - beta2)*(dW5**2)
    s_dW6 = beta2*s_dW6 + (1 - beta2)*(dW6**2)
    s_dW7 = beta2*s_dW7 + (1 - beta2)*(dW7**2)
    s_dW8 = beta2*s_dW8 + (1 - beta2)*(dW8**2)
    s_dWO = beta2*s_dWO + (1 - beta2)*(dWO**2)

        #^^^^^^^^^^^^^^^^^^^ Adding RMS to biases ^^^^^^^^^^^^^^^^^^^^^
    s_db1 =    beta2*s_db1 + (1 - beta2)*(db1**2)
    s_db2 =    beta2*s_db2 + (1 - beta2)*(db2**2)
    s_db3 =    beta2*s_db3 + (1 - beta2)*(db3**2)
    s_db4 =    beta2*s_db4 + (1 - beta2)*(db4**2)
    s_db5 =    beta2*s_db5 + (1 - beta2)*(db5**2)
    s_db6 =    beta2*s_db6 + (1 - beta2)*(db6**2)
    s_db7 =    beta2*s_db7 + (1 - beta2)*(db7**2)
    s_db8 =    beta2*s_db8 + (1 - beta2)*(db8**2)
    s_dbO =    beta2*s_dbO + (1 - beta2)*(dbO**2)


        ########################### OPTIMIZED GRADIENTS ##############################

    dW1_ =  v_dW1/np.sqrt(s_dW1 +    epsilon)
    dW2_ =  v_dW2/np.sqrt(s_dW2 +    epsilon)
    dW3_ =  v_dW3/np.sqrt(s_dW3 +    epsilon)
    dW4_ =  v_dW4/np.sqrt(s_dW4 +    epsilon)
    dW5_ =  v_dW5/np.sqrt(s_dW5 +    epsilon)
    dW6_ =  v_dW6/np.sqrt(s_dW6 +    epsilon)
    dW7_ =  v_dW7/np.sqrt(s_dW7 +    epsilon)
    dW8_ =  v_dW8/np.sqrt(s_dW8 +    epsilon)
    dWO_ =  v_dWO/np.sqrt(s_dWO +    epsilon)

    db1_ =  v_db1/np.sqrt(s_db1 +    epsilon)
    db2_ =  v_db2/np.sqrt(s_db2 +    epsilon)
    db3_ =  v_db3/np.sqrt(s_db3 +    epsilon)
    db4_ =  v_db4/np.sqrt(s_db4 +    epsilon)
    db5_ =  v_db5/np.sqrt(s_db5 +    epsilon)
    db6_ =  v_db6/np.sqrt(s_db6 +    epsilon)
    db7_ =  v_db7/np.sqrt(s_db7 +    epsilon)
    db8_ =  v_db8/np.sqrt(s_db8 +    epsilon)
    dbO_ =  v_dbO/np.sqrt(s_dbO +    epsilon)

    W1 = W1 - learning_rate*dW1_
    b1 = b1 - learning_rate*db1_

    W2 = W2 - learning_rate*dW2_
    b2 = b2 - learning_rate*db2_

    W3 = W3 - learning_rate*dW3_
    b3 = b3 - learning_rate*db3_

    W4 = W4 - learning_rate*dW4_
    b4 = b4 - learning_rate*db4_

    W5 = W5 - learning_rate*dW5_
    b5 = b5 - learning_rate*db5_

    W6 = W6 - learning_rate*dW6_
    b6 = b6 - learning_rate*db6_

    W7 = W7 - learning_rate*dW7_
    b7 = b7 - learning_rate*db7_

    W8 = W8 - learning_rate*dW8_
    b8 = b8 - learning_rate*db8_

    WO = WO - learning_rate*dWO_
    bO = bO - learning_rate*dbO_

    return W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO,v_dW1,v_dW2,v_dW3,v_dW4,v_dW5,v_dW6,v_dW7,v_dW8,v_dWO,v_db1,v_db2,v_db3,v_db4,v_db5,v_db6,v_db7,v_db8,v_dbO,s_dW1,s_dW2,s_dW3,s_dW4,s_dW5,s_dW6,s_dW7,s_dW8,s_dWO,s_db1,s_db2,s_db3,s_db4,s_db5,s_db6,s_db7,s_db8,s_dbO


def initalize_param(weight_init):
    ##################### Initial Weight Initialization ###################
    if weight_init=='RANDOM':
        W1 = np.random.rand(hidden_nodes_1,n_features) - 0.5
        b1 = np.random.rand(hidden_nodes_1,1) - 0.5
        
        W2 = np.random.rand(hidden_nodes_2,hidden_nodes_1) - 0.5
        b2 = np.random.rand(hidden_nodes_2,1) - 0.5
        
        W3 = np.random.rand(hidden_nodes_3,hidden_nodes_2) - 0.5
        b3 = np.random.rand(hidden_nodes_3,1) - 0.5
        
        W4 = np.random.rand(hidden_nodes_4,hidden_nodes_3) - 0.5
        b4 = np.random.rand(hidden_nodes_4,1) - 0.5
        
        W5 = np.random.rand(hidden_nodes_5,hidden_nodes_4) - 0.5
        b5 = np.random.rand(hidden_nodes_5,1) - 0.5

        W6 = np.random.rand(hidden_nodes_6,hidden_nodes_5) - 0.5
        b6 = np.random.rand(hidden_nodes_6,1) - 0.5

        W7 = np.random.rand(hidden_nodes_7,hidden_nodes_6) - 0.5
        b7 = np.random.rand(hidden_nodes_7,1) - 0.5

        W8 = np.random.rand(hidden_nodes_8,hidden_nodes_7) - 0.5
        b8 = np.random.rand(hidden_nodes_8,1) - 0.5

        WO = np.random.rand(n_classes,hidden_nodes_8) - 0.5
        bO = np.random.rand(n_classes,1) - 0.5

        return W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,WO,bO

    ###################### Xavier Weight Initalization ########################
    if weight_init=='XAVIER':
        W1 = np.random.randn(hidden_nodes_1,n_features)*np.sqrt(1/n_features)
        b1 = np.random.rand(hidden_nodes_1,1)*np.sqrt(1/n_features)
        
        W2 = np.random.rand(hidden_nodes_2,hidden_nodes_1)*np.sqrt(1/hidden_nodes_1)
        b2 = np.random.rand(hidden_nodes_2,1)*np.sqrt(1/hidden_nodes_1)
        
        W3 = np.random.rand(hidden_nodes_3,hidden_nodes_2)*np.sqrt(1/hidden_nodes_2)
        b3 = np.random.rand(hidden_nodes_3,1)*np.sqrt(1/hidden_nodes_2)
        
        W4 = np.random.rand(hidden_nodes_4,hidden_nodes_3)*np.sqrt(1/hidden_nodes_3)
        b4 = np.random.rand(hidden_nodes_4,1)*np.sqrt(1/hidden_nodes_3)
        
        W5 = np.random.rand(hidden_nodes_5,hidden_nodes_4)*np.sqrt(1/hidden_nodes_4)
        b5 = np.random.rand(hidden_nodes_5,1)*np.sqrt(1/hidden_nodes_4)

        W6 = np.random.rand(hidden_nodes_6,hidden_nodes_5)*np.sqrt(1/hidden_nodes_5)
        b6 = np.random.rand(hidden_nodes_6,1)*np.sqrt(1/hidden_nodes_5)

        W7 = np.random.rand(hidden_nodes_7,hidden_nodes_6)*np.sqrt(1/hidden_nodes_6)
        b7 = np.random.rand(hidden_nodes_7,1)*np.sqrt(1/hidden_nodes_6)

        W8 = np.random.rand(hidden_nodes_8,hidden_nodes_7)*np.sqrt(1/hidden_nodes_7)
        b8 = np.random.rand(hidden_nodes_8,1)*np.sqrt(1/hidden_nodes_7)

        WO = np.random.rand(n_classes,hidden_nodes_8)*np.sqrt(1/hidden_nodes_8)
        bO = np.random.rand(n_classes,1)*np.sqrt(1/hidden_nodes_8)
                
        return W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,WO,bO


def Forward_Prop(W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,WO,bO,X,activation):
    
    if activation=='ReLU':
        #print("FP ",X.shape)
        Z1 = W1.dot(X) + b1
        A1 = ReLU(Z1)

        Z2 = W2.dot(A1) + b2
        A2 = ReLU(Z2)

        Z3 = W3.dot(A2) + b3
        A3 = ReLU(Z3)

        Z4 = W4.dot(A3) + b4
        A4 = ReLU(Z4)

        Z5 = W5.dot(A4) + b5
        A5 = ReLU(Z5)

        Z6 = W6.dot(A5) + b6
        A6 = ReLU(Z6)

        Z7 = W7.dot(A6) + b7
        A7 = ReLU(Z7)

        Z8 = W8.dot(A7) + b8
        A8 = ReLU(Z8)

        ZO = WO.dot(A8) + bO
        AO = softmax(ZO)

        return Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,A6,Z6,A7,Z7,A8,Z8,AO,ZO    
    
    elif activation=='Tanh':
        Z1 = W1.dot(X) + b1
        A1 = Tanh(Z1)

        Z2 = W2.dot(A1) + b2
        A2 = Tanh(Z2)

        Z3 = W3.dot(A2) + b3
        A3 = Tanh(Z3)

        Z4 = W4.dot(A3) + b4
        A4 = Tanh(Z4)

        Z5 = W5.dot(A4) + b5
        A5 = Tanh(Z5)

        Z6 = W6.dot(A5) + b6
        A6 = Tanh(Z6)

        Z7 = W7.dot(A6) + b7
        A7 = Tanh(Z7)

        Z8 = W8.dot(A7) + b8
        A8 = Tanh(Z8)

        ZO = WO.dot(A8) + bO
        AO = softmax(ZO)

        return Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,A6,Z6,A7,Z7,A8,Z8,AO,ZO 


def Backpropagation(Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,A6,Z6,A7,Z7,A8,Z8,AO,ZO,
                                W1,W2,W3,W4,W5,W6,W7,W8,WO,X,Y,activation):
    
    if activation=='ReLU':
        one_hot_Y = one_hot(Y)
        #print(AO.shape,one_hot_Y.shape)
        dZO = AO - one_hot_Y
        dWO = 1 / m * dZO.dot(A8.T)
        dbO = 1 / m *np.sum(dZO)

        dZ8 = WO.T.dot(dZO) * derivative_ReLU(Z8)
        dW8 = 1 / m *dZ8.dot(A7.T)
        db8 = 1 / m *np.sum(dZ8)

        dZ7 = W8.T.dot(dZ8) * derivative_ReLU(Z7)
        dW7 = 1 / m *dZ7.dot(A6.T)
        db7 = 1 / m *np.sum(dZ7)

        dZ6 = W7.T.dot(dZ7) * derivative_ReLU(Z6)
        dW6 = 1 / m *dZ6.dot(A5.T)
        db6 = 1 / m *np.sum(dZ6)

        dZ5 = W6.T.dot(dZ6) * derivative_ReLU(Z5)
        dW5 = 1 / m *dZ5.dot(A4.T)
        db5 = 1 / m *np.sum(dZ5)
        
        dZ4 = W5.T.dot(dZ5) * derivative_ReLU(Z4)
        dW4 = 1 / m *dZ4.dot(A3.T)
        db4 = 1 / m *np.sum(dZ4)

        dZ3 = W4.T.dot(dZ4) * derivative_ReLU(Z3)
        dW3 = 1 / m *dZ3.dot(A2.T)
        db3 = 1 / m *np.sum(dZ3)

        dZ2 = W3.T.dot(dZ3) * derivative_ReLU(Z2)
        dW2 = 1 / m *dZ2.dot(A1.T)
        db2 = 1 / m *np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
        dW1 = 1 / m *dZ1.dot(X.T)
        db1 = 1 / m *np.sum(dZ1)

        return dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO
    

    if activation=='Tanh':
        one_hot_Y = one_hot(Y)
        dZO = AO - one_hot_Y
        dWO = 1 / m * dZO.dot(A8.T)
        dbO = 1 / m *np.sum(dZO)

        dZ8 = WO.T.dot(dZO) * derivative_Tanh(Z8)
        dW8 = 1 / m *dZ8.dot(A7.T)
        db8 = 1 / m *np.sum(dZ8)

        dZ7 = W8.T.dot(dZ8) * derivative_Tanh(Z7)
        dW7 = 1 / m *dZ7.dot(A6.T)
        db7 = 1 / m *np.sum(dZ7)

        dZ6 = W7.T.dot(dZ7) * derivative_Tanh(Z6)
        dW6 = 1 / m *dZ6.dot(A5.T)
        db6 = 1 / m *np.sum(dZ6)

        dZ5 = W6.T.dot(dZ6) * derivative_Tanh(Z5)
        dW5 = 1 / m *dZ5.dot(A4.T)
        db5 = 1 / m *np.sum(dZ5)

        dZ4 = W5.T.dot(dZ5) * derivative_Tanh(Z4)
        dW4 = 1 / m *dZ4.dot(A3.T)
        db4 = 1 / m *np.sum(dZ4)

        dZ3 = W4.T.dot(dZ4) * derivative_Tanh(Z3)
        dW3 = 1 / m *dZ3.dot(A2.T)
        db3 = 1 / m *np.sum(dZ3)

        dZ2 = W3.T.dot(dZ3) * derivative_Tanh(Z2)
        dW2 = 1 / m *dZ2.dot(A1.T)
        db2 = 1 / m *np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * derivative_Tanh(Z1)
        dW1 = 1 / m *dZ1.dot(X.T)
        db1 = 1 / m *np.sum(dZ1)

        return dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO

def SGD(W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO,
                                dW1,db1,dW2,db2,dW3,db3,dW4,db4,
                                dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO,learning_rate):
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1

    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    W3 = W3 - learning_rate*dW3
    b3 = b3 - learning_rate*db3

    W4 = W4 - learning_rate*dW4
    b4 = b4 - learning_rate*db4

    W5 = W5 - learning_rate*dW5
    b5 = b5 - learning_rate*db5

    W6 = W6 - learning_rate*dW6
    b6 = b6 - learning_rate*db6

    W7 = W7 - learning_rate*dW7
    b7 = b7 - learning_rate*db7

    W8 = W8 - learning_rate*dW8
    b8 = b8 - learning_rate*db8

    WO = WO - learning_rate*dWO
    bO = bO - learning_rate*dbO

    return W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO


def get_predict(A):
    return np.argmax(A,0)

def get_accuracy(pred,true):
    return np.sum(pred==true) / true.size

def fit_ANN(X_train,Y_train,X_test,Y_test,learning_rate=0.01,epochs=10,activation='ReLU',
                                        weight_initial='RANDOM',optimizer='SGD'):
    
    W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,WO,bO = initalize_param(weight_initial)
    
    History = []
    if optimizer=='ADAM':
        #opti_function = AdamOptimizer()
        #++++++++++++++++++ Momentum +++++++++++++++++++++++++++++++
        v_dW1,v_dW2,v_dW3,v_dW4,v_dW5,v_dW6,v_dW7,v_dW8,v_dWO = 0,0,0,0,0,0,0,0,0
        v_db1,v_db2,v_db3,v_db4,v_db5,v_db6,v_db7,v_db8,v_dbO = 0,0,0,0,0,0,0,0,0
        
        #++++++++++++++++++ RMS ++++++++++++++++++++++++++++++++++++++
        s_dW1,s_dW2,s_dW3,s_dW4,s_dW5,s_dW6,s_dW7,s_dW8,s_dWO = 0,0,0,0,0,0,0,0,0
        s_db1,s_db2,s_db3,s_db4,s_db5,s_db6,s_db7,s_db8,s_dbO = 0,0,0,0,0,0,0,0,0

    
    #N_BATCHES = math.ceil(X_train.shape[1]/batch_size)
    
    for epoch in range(epochs):
            ############ MODEL TRAINING ###############
        Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,A6,Z6,A7,Z7,A8,Z8,AO,ZO= Forward_Prop(W1,b1,W2,b2,W3,b3,
                                                                                W4,b4,W5,b5,W6,b6,W7,b7,
                                                                                W8,b8,WO,bO,
                                                                                X_train,activation)
            
            #print(AO.shape,one_hot(Y_train_batch).shape)
        loss = CrossEntropyLoss(AO,one_hot(Y_train))
            
        dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO = Backpropagation(Z1,A1,Z2,
                                                                    A2,Z3,A3,Z4,A4,Z5,A5,A6,Z6,A7,Z7,
                                                                    A8,Z8,AO,ZO,W1,W2,W3,W4,W5,W6,W7,W8,WO,
                                                                    X_train,Y_train,activation)
            
        if optimizer=='SGD':
            W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO = SGD(W1,W2,W3,W4,W5,W6,W7,W8,WO,
                                                                b1,b2,b3,b4,b5,b6,b7,b8,bO,
                                                                dW1,db1,dW2,db2,dW3,db3,dW4,db4,
                                                                dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO,
                                                                learning_rate)
        elif optimizer=='ADAM':
            
            W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO,v_dW1,v_dW2,v_dW3,v_dW4,v_dW5,v_dW6,v_dW7,v_dW8,v_dWO,v_db1,v_db2,v_db3,v_db4,v_db5,v_db6,v_db7,v_db8,v_dbO,s_dW1,s_dW2,s_dW3,s_dW4,s_dW5,s_dW6,s_dW7,s_dW8,s_dWO,s_db1,s_db2,s_db3,s_db4,s_db5,s_db6,s_db7,s_db8,s_dbO = AdamDemo(W1,W2,W3,W4,W5,W6,W7,W8,WO,b1,b2,b3,b4,b5,b6,b7,b8,bO,
                                                                                                                                                                                                                                                                                                dW1,db1,dW2,db2,dW3,db3,dW4,db4,dW5,db5,dW6,db6,dW7,db7,dW8,db8,dWO,dbO,learning_rate,
                                                                                                                                                                                                                                                                                                v_dW1,v_dW2,v_dW3,v_dW4,v_dW5,v_dW6,v_dW7,v_dW8,v_dWO,
                                                                                                                                                                                                                                                                                                v_db1,v_db2,v_db3,v_db4,v_db5,v_db6,v_db7,v_db8,v_dbO,
                                                                                                                                                                                                                                                                                                s_dW1,s_dW2,s_dW3,s_dW4,s_dW5,s_dW6,s_dW7,s_dW8,s_dWO,
                                                                                                                                                                                                                                                                                                s_db1,s_db2,s_db3,s_db4,s_db5,s_db6,s_db7,s_db8,s_dbO)





        train_loss = loss
        ########### EVALUATION ON TRAINING SET #########
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,prob_train,_ = Forward_Prop(W1,b1,W2,b2,W3,b3,
                                                W4,b4,W5,b5,W6,b6,W7,b7,
                                                W8,b8,WO,bO,
                                                X_train,activation)
        predictions = get_predict(prob_train)
        train_acc = get_accuracy(predictions,Y_train)

        ########### EVALUATION ON TEST SET #############
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,prob,_ = Forward_Prop(W1,b1,W2,b2,W3,b3,
                                                W4,b4,W5,b5,W6,b6,W7,b7,
                                                W8,b8,WO,bO,
                                                X_test,activation)
        test_pred = get_predict(prob)
        test_acc = get_accuracy(test_pred,Y_test)
        Y_test_ = one_hot(Y_test)
        test_loss = CrossEntropyLoss(prob,Y_test_)

        ########### SAVING HISTORY ####################
        his = {
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc':test_acc
        }
        History.append(his)        
        
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}]  Training Acc-{:.4f} Training Loss-{:.4f} Test Acc-{:.4f} Test Loss-{:.4f}"
                            .format(epoch+1,train_acc,train_loss,test_acc,test_loss))
    return History

np.seterr(divide='ignore', invalid='ignore')
####################################################################################################
########################################### PARAMETERS #############################################
####################################################################################################

m = len(data)
n_features = 16
n_classes = 26

hidden_nodes_1 = 20
hidden_nodes_2 = 40
hidden_nodes_3 = 60
hidden_nodes_4 = 80
hidden_nodes_5 = 100
hidden_nodes_6 = 80
hidden_nodes_7 = 60
hidden_nodes_8 = 40

LEARNING_RATE = 0.001
EPOCHS = 300
ACTIVATION_FUNCTION = 'Tanh'                   # Tanh / ReLU
WEIGHT_INITIALIZATION_TYPE = 'XAVIER'          # XAVIER / RANDOM
OPTIMIZER = 'ADAM'  



####################################################################################################
################################################## FIT #############################################
####################################################################################################
History = fit_ANN(X_train,Y_train,X_test,Y_test,LEARNING_RATE,EPOCHS,ACTIVATION_FUNCTION,
                                            WEIGHT_INITIALIZATION_TYPE,OPTIMIZER)






##############################################################################################################
############################################### Plotting Loss and Acc Curve ##################################
##############################################################################################################


history = pd.DataFrame(History)
train_acc = history.iloc[:,1]
train_loss = history.iloc[:,0]
test_acc = history.iloc[:,3]
test_loss = history.iloc[:,2]
train_acc = train_acc.values
train_loss = train_loss.values
test_acc = test_acc.values
test_loss = test_loss.values


plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), train_acc, label='Training Accuracy')
plt.plot(range(EPOCHS), test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS),train_loss, label='Training Loss')
plt.plot(range(EPOCHS), test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()