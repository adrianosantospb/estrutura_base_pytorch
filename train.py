# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim, utils, max
from utils.dataset import DataSet
import torchvision.transforms as transforms
from model import Model
from random import choices, randrange
from datetime import datetime

# ************************************ DADOS ***************************************************
# Dataset de treinamento e validacao.
train = DataSet(True, transforms.ToTensor())
val = DataSet(False, transforms.ToTensor())

# DataLoaders
train_loader = utils.data.DataLoader(dataset=train, shuffle=True, batch_size=2000)
validation_loader = utils.data.DataLoader(dataset=val, batch_size=1)

# ************************************* REDE ************************************************
loss = nn.CrossEntropyLoss()
model = Model(784, 50, 50, 10)

optimizer = optim.SGD(model.parameters(), lr=0.01)

startTime = datetime.now()

# ************************************ TREINAMENTO E VALIDACAO ********************************************
for epoch in range(1, 50):
    print("Etapa de treinamento {}".format(str(epoch)))
    model.train()
    i = 0
    for step, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        yPred = model(X.view(-1, 28*28))
        error = loss(yPred, y)
        error.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():    
        print("Etapa de validação {}".format(str(epoch)))
        for step, (X, y) in enumerate(validation_loader):
            yPred = model(X.view(-1, 28 * 28))
            _, label = max(yPred, 1)

    # TODO: Etapas de avaliacão e armazenamento do modelo.


seconds = (datetime.now() - startTime).total_seconds()
print ("Time taken : {0} seconds".format(seconds))

# ************************************ TESTE ********************************************
model.eval()
with torch.no_grad():
    X, y = val[randrange(len(val))]
    yPred = model(X.view(-1, 28 * 28))
    _, label = max(yPred, 1)
    print("Valor predito: {}. Valor real {}.".format(label.item(), y))