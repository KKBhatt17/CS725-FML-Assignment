import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import random

np.random.seed(42)

class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        raise NotImplementedError
    def __call__(self,features, is_train=False):
        raise NotImplementedError


def get_features(csv_path,is_train=False,scaler=None):
    feature_data = pd.read_csv(csv_path,usecols=['latitude','longitude','brightness','scan','track','acq_date','acq_time','satellite','daynight','confidence','bright_t31'])
    feature_data[["year", "month", "day"]] = feature_data["acq_date"].str.split("-", expand = True)
    features = ['latitude','longitude','brightness','scan','track','day','month','year','acq_time','satellite','confidence','bright_t31','daynight']
    feature_df= pd.get_dummies(feature_data[features])
    ####

    # Below Lines are The part of BASIS FUNCTIONS. Comment them out for question number 1.
    #feature_df=basis_function_1(feature_df)
    feature_df=basis_function_2(feature_df)
    
    
    #These lines are not a part of basis functions.
    feature_matrix = np.array(feature_df)
    return feature_matrix


def get_targets(csv_path):

    target_data = pd.read_csv(csv_path,usecols=['frp'])
    target = np.array(target_data)
    return target
     

def analytical_solution(feature_matrix, targets, C=0.0):
    
    feature_transpose = feature_matrix.transpose()
    norm = C*np.eye(feature_matrix.shape[1])
    first_term = np.dot(feature_transpose,feature_matrix)
    second_term = np.add(first_term,norm)
    third_term = np.linalg.inv(second_term)
    fourth_term = np.dot(third_term,feature_transpose)
    a_solution = np.dot(fourth_term,targets)
    return a_solution

def get_predictions(feature_matrix, weights):
    
    weights_transpose = weights.transpose()
    row_count=int(feature_matrix.shape[0])
    predictions=np.empty([row_count])

    for i in range(0,feature_matrix.shape[0]):
        predictions[i] = np.dot(weights_transpose,feature_matrix[i])
        if(predictions[i]<0):
            predictions[i]=0
    
    return predictions

def mse_loss(feature_matrix, weights, targets):
    
    sum1=0.0
    weights_transpose = weights.transpose()
    row_count=int(feature_matrix.shape[0])
    predictions=np.empty([row_count])

    for i in range(0,feature_matrix.shape[0]):
        predictions[i] = np.dot(weights_transpose,feature_matrix[i])
        if(predictions[i]<0):
            predictions[i]=0
        sum1=sum1+((predictions[i]-targets[i][0])**2)

    mse = sum1/row_count
    return mse
    

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    sum1=0.0
    for i in range(0,weights.shape[0]):
        sum1=sum1+(weights[i]*weights[i])
    
    return sum1

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    mse_error=mse_loss(feature_matrix,weights,targets)
    l2_penalty=l2_regularizer(weights)
    total_loss=mse_error+(C*l2_penalty)
    return total_loss

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    x_transpose = feature_matrix.transpose()
    xw = np.dot(feature_matrix,weights)
    y_minus_xw = np.subtract(targets,xw)
    error=np.dot(x_transpose,y_minus_xw)
    minus_two=-2
    error=error*minus_two
    scaling_factor=2*C
    penalty=weights*scaling_factor
    gradient1=np.add(error,penalty)
    return gradient1

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''
    no_of_rows = feature_matrix.shape[0]
    no_of_columns=feature_matrix.shape[1]

    sampled_feature_matrix=np.empty([batch_size,no_of_columns])
    sampled_targets=np.empty([batch_size,1])

    n = random.randint(0,no_of_rows-batch_size-10)

    for i in range(0,batch_size):
        index=n+i
        sampled_targets[i]=targets[index]

    for i in range(0,batch_size):
        index=n+i
        for j in range(0,no_of_columns):
            sampled_feature_matrix[i][j]=feature_matrix[index][j]
    
    return (sampled_feature_matrix, sampled_targets)
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    initial_weights=np.zeros([n,1])
    return initial_weights

def update_weights(weights, gradients, lr):   
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''
    gradients=gradients*lr
    weights=np.subtract(weights,gradients)
    return weights

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
    # allowed to modify argument list as per your need
    # return True or False
    raise NotImplementedError

def basis_function_1(feature_df):
    feature_df["brightness_1"]=feature_df["brightness"]**2
    feature_df["bright_t31_1"]=feature_df["bright_t31"]**2
    feature_df["confidence_1"]=feature_df["confidence"]**2
    feature_df["bright_t31_4"] = feature_df["bright_t31"] ** 4
    feature_df["brightness_4"] = feature_df["brightness"] ** 4
    feature_df["brightness_5"] = feature_df["brightness"] ** 5
    feature_df["bright_t31_5"] = feature_df["bright_t31"] ** 5
    feature_df["bright_t31_6"] = feature_df["bright_t31"] ** 6
    feature_df["brightness_6"] = feature_df["brightness"] ** 6

    return feature_df

def basis_function_2(feature_df):
    feature_df["brightness_1"]=feature_df["brightness"]**2
    feature_df["bright_t31_1"]=feature_df["bright_t31"]**2
    feature_df["confidence_1"]=feature_df["confidence"]**2
    feature_df["bright_t31_4"] = feature_df["bright_t31"] ** 4
    feature_df["brightness_4"] = feature_df["brightness"] ** 4
    feature_df["brightness_5"] = feature_df["brightness"] ** 5
    feature_df["bright_t31_5"] = feature_df["bright_t31"] ** 5
    feature_df["bright_t31_6"] = feature_df["bright_t31"] ** 6
    feature_df["brightness_6"] = feature_df["brightness"] ** 6
    feature_df["add_all"] = feature_df["latitude"] + feature_df["longitude"] + feature_df["brightness"] + feature_df["bright_t31"]
    feature_df["scan"] = feature_df["add_all"] + feature_df["scan"]
    feature_df["track"] = feature_df["add_all"] + feature_df["track"]
    feature_df["scan_2"] = feature_df["scan"]**2
    feature_df["track_2"] = feature_df["track"]**2

    return feature_df

def plot_trainsize_losses(training_data,training_label,dev_data,dev_label):    
    '''
    Description:
    plot losses on the development set instances as a function of training set size 
    '''

    '''
    Arguments:
    # you are allowed to change the argument list any way you like 
    '''
    tf_subsets = []
    tl_subsets = []
    train_size = ["5000","10000","15000","20000","full"]
    devset_loss_list=[]
    tf_subsets.append(training_data[:5000])
    tf_subsets.append(training_data[:10000])
    tf_subsets.append(training_data[:15000])
    tf_subsets.append(training_data[:20000])
    tf_subsets.append(training_data[:])
    tl_subsets.append(training_label[:5000])
    tl_subsets.append(training_label[:10000])
    tl_subsets.append(training_label[:15000])
    tl_subsets.append(training_label[:20000])
    tl_subsets.append(training_label[:])

    for i in range(0,5):
        tf_subset=tf_subsets[i]
        tl_subset=tl_subsets[i]
        w=analytical_solution(tf_subset, tl_subset, C=1)
        devset_loss=do_evaluation(dev_data, dev_label, w)
        devset_loss_list.append(devset_loss)
        #print('train subset {} \t devset_loss: {} '.format(i, devset_loss))

    plt.figure(figsize=(8,6)) 
    plt.title("Train set size vs Developmet set MSE loss plot")
    plt.xlabel("Train set size")
    plt.ylabel("MSE loss on development set")
    plt.plot(train_size, devset_loss_list)
    plt.savefig("plot.jpg")


def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):

    n=int(train_feature_matrix.shape[1])
    weights = initialize_weights(n)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr/step)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

            if(step==eval_steps):
                current_loss=train_loss
                prev_loss=0
            else:
                prev_loss=current_loss
                current_loss=train_loss
            # print("losses= ",current_loss,prev_loss,abs(current_loss - prev_loss))
            if(abs(current_loss - prev_loss)<10):
                break

        '''
        implement early stopping etc. to improve performance.
        '''

    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    #scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('data/train.csv',True), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=0.1)

    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.0000000000000000000000000000000005,
                        C=1.0,
                        batch_size=1000,
                        max_steps=200000,
                        eval_steps=50)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    #Uncomment the below line to plot the Train size vs Dev loss plot
    #plot_trainsize_losses(train_features,train_targets,dev_features, dev_targets)