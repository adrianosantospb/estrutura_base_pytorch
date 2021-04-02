# Estrutura básida para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

import torch
from torch import nn, optim
from utils.dataset import DataSet
import torchvision.transforms as transforms
from model import ModelCNN
from sklearn.metrics import accuracy_score
import random
from random import choices
from torch.autograd import Variable

# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging
import torch.multiprocessing as mp

import pycuda.driver as cuda
cuda.init()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



def main(parser):
    # ************************************ DADOS ***************************************************
    # Dataset de treinamento e validacao.
    train = DataSet(True, transforms.ToTensor())
    val = DataSet(False, transforms.ToTensor())

    # Selecionar o dispositivo a ser utilizado (CPU ou GPU).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Obtem a proporcao de workers por GPU
    num_worker = 4 * int(torch.cuda.device_count())

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=parser.batch_size,num_workers=num_worker, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=val, batch_size=parser.batch_size)

    # ************************************* REDE ************************************************
    criterion = nn.CrossEntropyLoss()
    model = ModelCNN()
    # GPU
    model.to(device)
    model.share_memory()
    model.half().float()

    otimizador = optim.SGD(model.parameters(), lr=parser.lr)

    # ************************************ TREINAMENTO E VALIDACAO ********************************************
    for epoch in range(1, parser.epochs):
        logging.info('Treinamento: {}'.format(str(epoch)))
        model.train()
        for step, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            otimizador.zero_grad()
            X , y = X.to(device), y.to(device)
            yPred = model(X).to(device)
            erro = criterion(yPred, y)
            erro.backward()
            otimizador.step()
        
        logging.info('Validacao: {}'.format(str(epoch)))
        model.eval() 
        val_loss = 0
        acertos = 0
        for step, (X, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            X,y = X.to(device), y.to(device)
            yPred = model(X).to(device)
            val_loss += criterion(yPred, y).detach().item()
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
        
        # Limpa o cache do CUDA
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dir_save', default="./pesos")
    parser = parser.parse_args()

    # Main function.
    main(parser)


