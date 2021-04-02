# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

from torch.utils.data import Dataset
import torchvision.datasets as datasets

class DataSet(Dataset):
    def __init__(self, train=False, transform=None):
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    def __getitem__(self,index):    
        data, target = self.dataset[index]
        return data, target
 
    def __len__(self):
        return len(self.dataset)
