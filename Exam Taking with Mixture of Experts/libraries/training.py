from .exam import get_exam_questions_for_router

from classes.loss import Loss
from classes.models import RNN, Router
from sentence_transformers import SentenceTransformer

import math
import pandas as pd
import torch
import torch.nn as nn
import random


def repeat_data(X, seq_len):
    X_new = torch.ones(X.size()[0], X.size()[1], seq_len)
    X_new = torch.einsum("nit,ni->nit", X_new, X)
    return X_new

def get_batches(X, Y, batch_size=100, n_batches=None):
    X_batches, Y_batches = [], []
    
    indices = [i for i in range(X.size()[0])]
    random.shuffle(indices)
    
    if n_batches is None:
        n_batches = math.ceil(X.size()[0] // batch_size)
        
    for batch_num in range(n_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size
        X_batches.append(X[indices[start:end]])
        Y_batches.append(Y[indices[start:end]])
    
    return X_batches, Y_batches

def train_rnn(model, X_train, Y_train, X_val, Y_val, n_batches=None, batch_size=100, n_epochs=10, lr=.1, betas=(.9, .999)):
    
    X_train_batches, Y_train_batches = get_batches(X_train, Y_train, n_batches=n_batches, batch_size=batch_size)
    
    n_train = sum([batch.size()[0] for batch in X_train_batches])
    n_batches = len(X_train_batches)
    
    #print(n_train)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=0)
    
    loss_fn = Loss()
    
    losses = {"val": [], "train": []}
    
    for epoch in range(1, n_epochs+1):
        
        for batch_num in range(n_batches):
            optimizer.zero_grad()

            X_batch = X_train_batches[batch_num]
            Y_batch = Y_train_batches[batch_num]
            
            out, hidden = model(X_batch)
            loss = loss_fn(out, Y_batch)
            
            loss.backward()

            optimizer.step()
            
        Y_hat_val, _ = model(X_val)
        loss_val = loss_fn(Y_hat_val, Y_val)
        losses["val"].append(loss_val.item()) 
        
        Y_hat_train, _ = model(X_train)
        loss_train = loss_fn(Y_hat_train, Y_train)
        losses["train"].append(loss_train.item())
            
        """
        
        if (10 * epoch) % n_epochs == 0:
            print("Epoch = {epoch}".format(epoch=epoch))
            print("Loss = {loss}".format(loss=loss_for_epoch))
            print("")
        
        """
        
    return model, losses

def tune_hyperparameters_for_rnn(lrs=[.1, .05, .01], batch_sizes=[500, 10000], n_epochs=10, seq_lens=[20, 50], hidden_sizes=[1, 2], betas=[(.9, .999), (.99, .99)]):
    data_train = pd.read_csv("data/polynomial_minima/train.csv")
    
    loss_data = []
    
    loss_fn = Loss()
    
    for seq_len in seq_lens:
        X_train = torch.Tensor(data_train[["A", "B", "C"]].to_numpy())
        X_train = repeat_data(X_train, seq_len)
        
        Y_train = torch.Tensor(data_train[["Y"]].to_numpy())
        
        data_val = pd.read_csv("data/polynomial_minima/val.csv")
        X_val = torch.Tensor(data_val[["A", "B", "C"]].to_numpy())
        Y_val = torch.Tensor(data_val[["Y"]].to_numpy())

        X_val = repeat_data(X_val, seq_len)
        
        for lr in lrs:
            for batch_size in batch_sizes:
                for hidden_size in hidden_sizes:
                    for beta in betas:
                        
                        model = RNN(hidden_size=hidden_size)
                        trained_model, losses = train_rnn(model, X_train, Y_train, X_val, Y_val, batch_size=batch_size, n_epochs=n_epochs, lr=lr, betas=beta)
                        
                        loss_train = dict(zip(["Training Loss for Epoch %d"%(i+1) for i in range(n_epochs)], losses["train"]))
                        loss_val = dict(zip(["Validation Loss for Epoch %d"%(i+1) for i in range(n_epochs)], losses["val"]))

                        hyperparams = {"lr": lr, "batch_size": batch_size, "n_epochs": n_epochs, "seq_len": seq_len, "hidden_size": hidden_size, "beta1": beta[0], "beta2": beta[1]}
                        
                        data_for_these_hyperparams = {}
                        
                        data_for_these_hyperparams.update(hyperparams)
                        data_for_these_hyperparams.update(loss_train)
                        data_for_these_hyperparams.update(loss_val)
                        
                        loss_data.append(data_for_these_hyperparams)
                        
    loss_data = pd.DataFrame(loss_data)

    return loss_data
        
    
def train_router(model, X_train, Y_train, X_val, Y_val, betas=None, lr=None, n_epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    
    losses = {"train": [], "val": []}
    
    for epoch in range(1,n_epochs+1):
        optimizer.zero_grad()
        
        out = model(X_train)
        
        loss = loss_fn(out, Y_train)
        loss.backward()
        
        optimizer.step()
        
        Y_hat_val = model(X_val)
        
        loss_val = loss_fn(Y_hat_val, Y_val)
        
        losses["train"].append(loss.item())
        losses["val"].append(loss_val.item())
       
    return model, losses    


def tune_hyperparameters_for_router(n_train=800, lrs=[.1, .05, .001], betas=[(.9, .999), (.99, .99)], d_embedding=384, n_classes=3):
    
    X, Y = get_exam_questions_for_router()

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    sentence_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = torch.Tensor(sentence_embedding.encode(X_train))

    Y_train = torch.LongTensor(Y_train)
    
    X_val = X[n_train:]
    Y_val = Y[n_train:]

    X_val = torch.Tensor(sentence_embedding.encode(X_val))

    Y_val = torch.LongTensor(Y_val)

    df = []

    for lr in lrs:
        for beta in betas:
            model = Router(d_embedding, n_classes)

            trained_model, losses = train_router(model, X_train, Y_train, X_val, Y_val, betas=beta, lr=lr, n_epochs=10)

            df_params = {}

            df_params["lr"] = lr
            df_params["beta1"] = beta[0]
            df_params["beta2"] = beta[1]

            loss_train = dict(zip(["Training Loss for Epoch %d"%i for i in range(1, 10+1)], losses["train"]))
            loss_val = dict(zip(["Validation Loss for Epoch %d"%i for i in range(1, 10+1)], losses["val"]))

            df_params.update(loss_train)
            df_params.update(loss_val)

            df.append(df_params)

    loss_data = pd.DataFrame(df)

    return loss_data