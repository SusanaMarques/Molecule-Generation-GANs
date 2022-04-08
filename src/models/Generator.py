import torch.nn as nn
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, data_shape):
        super(Generator, self).__init__()
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

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

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

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)