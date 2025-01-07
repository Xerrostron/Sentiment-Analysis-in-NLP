#!/usr/bin/env python
# coding: utf-8
'''fun: sentiment analysis'''


import torch
import os
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader,Dataset,TensorDataset
# import pandas as pd
from DataLoader import MovieDataset
# from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import time
from LSTM import LSTMModel


'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):

    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)
    #parent directory cpt does not exist


def main():
    gpu_id = 0

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_filename = "model_checkpoint.pth"
    ckp_path = os.path.join(checkpoint_dir, ckpt_filename)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", gpu_id)
    else:
        device = torch.device('cpu')
    
    ## alternative method
    # torch.cuda.set_device(device=0) ## choose gpu number
    print('device: ', device)

    ## ---------------------------------------------------------
    ## please change the parameter settings by yourselves
    ## ---------------------------------------------------------
    mode = 'train'
    losses = []
    iterations = []

    Batch_size =128
    #MUCH MORE SMOOTH GRAPH WITH 128
    #.01, .835 with 64, how wabout 32 and 128?
    #32 has very similar results too
    n_layers = 1 ## choose 1-3 layers
    #more layers ruins the accuracy

    ## input seq length aligned with data pre-processing
    input_len = 150
    global_step = 0

    ## word embedding length try 200 and 100
    embedding_dim = 200

    # lstm hidden dim
    hidden_dim = 50
    # binary cross entropy
    output_size = 1
    num_epoches = 1
    ## please change the learning rate by youself
    #.01 best results right now (.83 for acc)
    #0.1 is horrible
    learning_rate = .01
    # gradient clipping
    clip = 5
    load_cpt = False #True
    #CHANGED CKP PATH
    #ckp_path = 'cpt/name.pt'
    # embedding_matrix = None
    ## use pre-train Glove embedding or not?
    pretrain = False

    ##-----------------------------------------------------------------------
    ## Bonus (5%): complete code to add GloVe embedding file path below.
    ## Download Glove embedding from https://nlp.stanford.edu/data/glove.6B.zip
    ## "embedding_dim" defined above shoud be aligned with the dimension of GloVe embedddings
    ## if you do not want bonus, you can skip it.
    ##-----------------------------------------------------------------------
    glove_file = 'path/glove.6B.200d.txt' ## change by yourself
    

    ## ---------------------------------------------------------
    ## step 1: create data loader in DataLoader.py
    ## complete code in DataLoader.py (not Below)
    ## ---------------------------------------------------------
    
    
    ## step 2: load training and test data from data loader [it is Done]
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size,\
                                    shuffle=True,num_workers=1)
    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size,\
                                shuffle=False,num_workers=1)


    ## step 3: [Bonus] read tokens and load pre-train embedding [it is Done]
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    ## -----------------------------------------------
    ## step 4: import model from LSTM.py
    ## complete the code in "def forward(self, x)" in LSTM.py file
    ## then import model from LSTM.py below
    ## and also load model to device
    ## -----------------------------------------------
    #model = LSTM(xxx,xxx)
   # model.to()
   #Double Check
    model = LSTMModel(vocab_size=vocab_size,output_size=output_size,embedding_dim=100,embedding_matrix=embedding_matrix,hidden_dim=hidden_dim,n_layers=n_layers,input_len=input_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ##-----------------------------------------------------------
    ## step 5: complete code to define optimizer and loss function
    ##-----------------------------------------------------------
    #exempting the parameter: net.parameters() ??
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  # 
    ## define Binary Cross Entropy Loss below
    loss_fun = nn.BCELoss()
    
    ## step 6: load checkpoint
    if load_cpt:
        print("*"*10+'loading checkpoint'+'*'*10)
        ##-----------------------------------------------   
        ## complete code below to load checkpoint
        ##-----------------------------------------------
        checkpoint = torch.load(ckp_path)  # Load the checkpoint file
        model.load_state_dict(checkpoint['model_state_dict'])  # Load the model's state_dict from the checkpoint
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer's state_dict from the checkpoint
        epoches = checkpoint['epoch']  # Retrieve the epoch information from the checkpoint
        global_step = checkpoint['global_step']  # Retrieve the global_step information from the checkpoint


    ## step 7: model training
    print('*'*89)
    print('start model training now')
    print('*'*89)
    
    if mode == 'train':
        model.train()
        counter = 0
        for epoches in range(num_epoches):
             #set counter to 0 when another epoch is being trained
            #OR, have counter outside?
            for x_batch, y_labels in training_generator:
                
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)
                ##-----------------------------------------------
                ## complete code to get predict result from model
                ##-----------------------------------------------
                #def forward(self, x) returns out
                y_out = model(x_batch) 

                ##-----------------------------------------------
                ## complete code to get loss
                ##-----------------------------------------------
                loss = loss_fun(y_out, y_labels) 
                

                ## step 8: back propagation [Done]
                optimizer.zero_grad()
                loss.backward()
                
                losses.append(loss.item())
                counter = counter+1 #after 1 training loop in an epoch, +1
                iterations.append(counter)
                ## **clip_grad_norm helps prevent the exploding gradient problem in LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                global_step = global_step + 1
                
            ##-----------------------------------------------   
            ## step 9: complete code below to save checkpoint
            ##-----------------------------------------------
            print("**** save checkpoint ****")
            #global_step referenced before assignment
            _save_checkpoint(ckp_path, model, epoches, global_step, optimizer)
    
# Plot the loss function curve calculations of iterations and losses after training
    plt.plot(iterations, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Function Curve')
    plt.show()
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Obtain predictions from the model on the test set
    all_y_true = []
    all_y_pred = []


    ## step 10: complete code below for model testing
    ## predict result is a single value between 0 and 1, such as 0.8, so
    ## we can use y_pred = torch.round(y_out) to predict label 1 or 0
    ##------------------------------------------------------------------

    print("----model testing now----")
    print("----model testing now----")
    testloader = torch.utils.data.DataLoader(test_set, Batch_size,
                                     shuffle=True, num_workers=1)
    model.eval()
    with torch.no_grad():  # Disable gradient tracking during inference
     for x_batch, target in testloader:  # Iterate over test data batches
        x_batch = x_batch.to(device)  # Move batch to device
        y_out = model(x_batch)  # Pass test data through the model to obtain predictions
        y_pred = torch.round(y_out)
            # Append predictions and ground truth labels to lists
        all_y_true.extend(target.cpu().numpy())
        all_y_pred.extend(y_pred.cpu().numpy())
    

# Compute evaluation metrics
    # Compute evaluation metrics using all predictions and ground truth labels
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    conf_matrix = confusion_matrix(all_y_true,all_y_pred)
# Print or log the confusion matrix and evaluation metrics
 
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('Confusion Matrix', conf_matrix)

    
 #This code: compiles and runs with no errors. Now, go over the hyperparameters and results   


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    


    

    