# Estrutura básida para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim
from utils.dataset import DataSet
import torchvision.transforms as transforms
from tqdm import tqdm
from model import Model
import logging
from sklearn.metrics import accuracy_score
import random
from random import choices

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# ************************************ DADOS ***************************************************
# Dataset de treinamento e validacao.
train = DataSet(True, transforms.ToTensor())
val = DataSet(False, transforms.ToTensor())

# DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=2000)
validation_loader = torch.utils.data.DataLoader(dataset=val, batch_size=1)

# ************************************* REDE ************************************************
criterion = nn.CrossEntropyLoss()
model = Model(784, 50, 50, 10)
otimizador = optim.SGD(model.parameters(), lr=0.01)


# ************************************ TREINAMENTO E VALIDACAO ********************************************
for epoch in range(1, 2):
    print("Treinamento número {}".format(str(epoch)))
    model.train()
    for step, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        otimizador.zero_grad()
        yPred = model(X.view(-1, 28*28))
        erro = criterion(yPred, y)
        erro.backward()
        otimizador.step()
    
    model.eval()    
    print("Validacao")
    for step, (X, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
        yPred = model(X.view(-1, 28 * 28))
        _, label = torch.max(yPred, 1)
        
# ************************************ TESTE ********************************************
model.eval()

dataset_test = iter(validation_loader)
X, y = val[random.randrange(len(val))]

yPred = model(X.view(-1, 28 * 28))
print(type(yPred))
_, label = torch.max(yPred, 1)
print(label.item(), y)