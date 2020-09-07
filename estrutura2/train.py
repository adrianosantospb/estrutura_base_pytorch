# Estrutura básida para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim
from utils.dataset import DataSet
import torchvision.transforms as transforms
from model import Model
from sklearn.metrics import accuracy_score
import random
from random import choices
from torch.autograd import Variable

# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



def main(parser):
    # ************************************ DADOS ***************************************************
    # Dataset de treinamento e validacao.
    train = DataSet(True, transforms.ToTensor())
    val = DataSet(False, transforms.ToTensor())

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=parser.batch_size)
    validation_loader = torch.utils.data.DataLoader(dataset=val, batch_size=parser.batch_size)

    # ************************************* REDE ************************************************
    criterion = nn.CrossEntropyLoss()
    model = Model(784, 50, 50, 10)
    otimizador = optim.SGD(model.parameters(), lr=parser.lr)


    # ************************************ TREINAMENTO E VALIDACAO ********************************************
    for epoch in range(1, parser.epochs):
        logging.info('Treinamento: {}'.format(str(epoch)))
        model.train()
        for step, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            otimizador.zero_grad()
            yPred = model(X.view(-1, 28 * 28))
            erro = criterion(yPred, y)
            erro.backward()
            otimizador.step()
        
        logging.info('Validacao: {}'.format(str(epoch)))
        model.eval() 
        val_loss = 0
        acertos = 0
        for step, (X, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            yPred = model(X.view(-1, 28 * 28))
            val_loss += criterion(yPred, y).item()
            predito = torch.max(yPred,1)[1]
            acertos += (predito == y).sum()
        
        if epoch % 5 == 0:
            # Nome do arquivo dos pesos
            pesos = "{}/{}_pesos.pt".format(parser.dir_save,str(epoch))
            
            # Imprime métricas
            logging.info("Loss error: {:.4f}, Accuracy:{:.4f}".format(val_loss, float(acertos) / (len(validation_loader)*parser.batch_size)))
            
            # Salvar os pesos
            chkpt = {'epoch': epoch,'model': model.state_dict()} 
            torch.save(chkpt, pesos)

    


       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dir_save', default="./pesos")
    parser = parser.parse_args()

    # Main function.
    main(parser)


