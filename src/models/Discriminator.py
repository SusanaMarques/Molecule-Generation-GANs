import torch.nn as nn
import numpy as np
import torch


class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(self.data_shape))),
            nn.LeakyReLU(0.2),
            nn.Linear(256),
            nn.LeakyReLU(0.2),
            nn.Linear(1),
            nn.Sigmoid()
        )

    def forward(self, mol):
        validity = self.model(mol)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict['model'])
        
        return D