# Estrutura básica para projetos de Machine Learning e Deep Learning
# Por Adriano Santos.

from torch import nn, relu

class Model(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, output_size):
        super(Model, self).__init__()
        self.entry = nn.Linear(input_size, hidden_1)
        self.hidden = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, output_size)

    def forward(self, x):
        out = relu(self.entry(x))
        out = relu(self.hidden(out))
        out = self.out(out)
        return out