import torch
from torch import nn
from torch import optim

import numpy as np
import scipy.stats as sps
from tqdm.notebook import tqdm
from time import time
import json
import os

class DatasetGenerator:

    def __init__(self, N):
        self.N = N

    def sample_init_conditionals_(self, sample_size):
        """
        Генерирует sample_size начальных условий
        """
        n = 20
        limits = sps.uniform.rvs(size=(sample_size * 4, n))
        limits = (np.sort(limits, axis=1) * (self.N + 1))
        limits = limits.astype(np.int)
        repeats = np.hstack((
            limits[:, 0].reshape(-1, 1),
            limits[:, 1:] - limits[:, :-1],
            ((self.N + 1) - limits[:, -1]).reshape(-1, 1)
        ))


        values = sps.uniform(loc=-1, scale=2).rvs(size=(sample_size * 4, n + 1))
        # дадим случайный подогорев каждой из сторон
        # чтобы темепарутра в центре не равна 0
        T = sps.uniform(loc=-1, scale=2).rvs(size=(sample_size * 4))
        values = T.reshape(-1, 1) + values

        self.init_condtionals_ = np.repeat(
            values.flatten(), 
            repeats.flatten()
            ).reshape(sample_size, 4, self.N + 1)


    def sample_f_(self, sample_size):
        """
        Генерирует sample_size правых частей
        """
        x = np.linspace(0, 1, self.N - 1).repeat(self.N - 1)
        y = np.linspace(0, 1, self.N - 1).reshape(1, -1)
        y = y.repeat(self.N - 1, axis=0).flatten()
        X = np.hstack((
            x.reshape(-1, 1),
            y.reshape(-1, 1)
        ))

        assert X.shape == ((self.N - 1) ** 2, 2), "Неверные размеры"

        mu = 10
        amount = 2 * mu + 1
        p = np.arange(-mu, mu + 1).repeat(amount)
        k = np.arange(-mu, mu + 1).reshape(1, -1)
        k = k.repeat(amount, axis=0).flatten()
        Z = np.hstack((
            p.reshape(-1, 1),
            k.reshape(-1, 1)
        ))

        assert Z.shape == (amount ** 2, 2), "Неверные размеры"

        # p = [-mu, ..., 0, ..., mu]
        # x = [0, h, 2 h, ..., 1]
        # Arg[i, j] = 2 \pi (p[i // amount] * x[j // (self.N - 1)] + 
        # + p[i % amount] * x[j % (self.N - 1)])
        Arg = 2 * np.pi * Z @ X.T

        sq_sum = p ** 2 + k ** 2

        cos = np.cos(Arg)
        sin = np.sin(Arg)

        a = sps.uniform(loc=-1, scale=2).rvs(size=(sample_size, amount ** 2))
        b = sps.uniform(loc=-1, scale=2).rvs(size=(sample_size, amount ** 2))
        
        s = 4 / 7
        factor = ((1 - sq_sum / mu ** 2) * (sq_sum < mu ** 2)) ** s
        a = a * factor
        b = b * factor

        f = a @ cos + b @ sin
        f = f.reshape(sample_size, self.N - 1, self.N -1)
        return f
        



    def __call__(self, num_sample=500, batch_size=100, dir="", save=True,
	    print_every=None):
        """
        Генерирует датасет и сохраняет батчи в директорию
        num_samples --- количество элементов датасета
        batch_size ---  размер батча
        dir --- директория датасета
	    print_every --- как часто записывать в историю
        """
        assert num_sample % batch_size == 0, "Некорректный размер батча"

        if (print_every == None):
            print_every = self.N

        labels_file_name = os.path.join(dir, 'labels.json')
        try:
            with open(labels_file_name, 'r') as labels_file:
                labels = json.load(labels_file)
        except FileNotFoundError:
            labels = []


        for i in tqdm(range(num_sample // batch_size)):
            current_time = time()
            labels.append(current_time)
            u = np.zeros((batch_size, self.N + 1, self.N + 1))
            self.sample_init_conditionals_(batch_size)
            u[:, :, 0] = self.init_condtionals_[:, 0, :]
            u[:, 0, :] = self.init_condtionals_[:, 1, :]
            u[:, :, -1] = self.init_condtionals_[:, 2, :]
            u[:, -1, :] = self.init_condtionals_[:, 3, :]

            h = 1.0/self.N
            max_it = self.N ** 2
            x = np.arange(self.N+1) * h # x = i * h, i = 0,...,N
            y = np.arange(self.N+1) * h # y = j * h, j = 0,...,N

            # согласно Деммелю, это примерное количество итераций
            # для получения адекватной неувязки
            max_it = self.N ** 2
            history = np.zeros(
                (batch_size, 
                max_it // print_every, self.N + 1, 
                self.N + 1)
                )
            
            x_index = np.repeat(np.arange(1, self.N)[:, np.newaxis], 
                                self.N-1, axis=1)
            y_index = np.repeat(np.arange(1, self.N)[np.newaxis, :], 
                                self.N-1, axis=0)
            f = self.sample_f_(batch_size)

            np.save(os.path.join(dir, f'init_cond_{current_time}'),
                    self.init_condtionals_)
            np.save(os.path.join(dir, f'right_side_{current_time}'), f)
            
            for j in tqdm(range(max_it)):
                u[:, x_index, y_index] = 0.25 * (u[:, x_index + 1, y_index] + 
                                        u[:, x_index - 1, y_index] + 
                                        u[:, x_index, y_index + 1] + 
                                        u[:, x_index, y_index - 1] - h ** 2 * f)
                if (j % print_every == 0):
                    history[:, j // print_every] = u
            
            # посмотрим неувязку для первого уравнения
            u_new = u[-1].copy()
            delta = u_new[x_index, y_index] -\
                                 0.25 * (u_new[x_index + 1, y_index] + 
                                        u_new[x_index - 1, y_index] + 
                                        u_new[x_index, y_index + 1] + 
                                        u_new[x_index, y_index - 1] - 
                                       h ** 2 * f[-1])
            self.log_delta = np.log(np.max(np.abs(delta)) / np.max(np.abs(u_new)))
            
            np.save(os.path.join(dir, f'solution_{current_time}'), u)
            np.save(os.path.join(dir, f'history_{current_time}'), history)

            # сохраним времена создания данных
            if (save):
                with open(labels_file_name, 'w') as labels_file:
                    json.dump(labels, labels_file)
        # возвратим последнее значение
        return u[-1]

    def load(self, dir="", load_history=False, slice=None):
        labels_file_name = os.path.join(dir, 'labels.json')
        with open(labels_file_name, 'r') as labels_file:
            labels = json.load(labels_file)

        if (slice != None):
            labels = labels[slice]

        result = dict()
        for label in labels:
            solution_path = os.path.join(dir, f'solution_{label}.npy')
            init_conditionals_path =\
                    os.path.join(dir, f'init_cond_{label}.npy')
            right_side_path = \
                    os.path.join(dir, f'right_side_{label}.npy')
            
            if (load_history):
                history_path = os.path.join(dir, f'history_{label}.npy')


            result['solution'] = np.load(solution_path, allow_pickle=True)
            result['init_condtionals'] = np.load(init_conditionals_path,
                                                 allow_pickle=True)
            result['right_side'] = np.load(right_side_path, allow_pickle=True)
            
            if (load_history):
                result['history'] = np.load(history_path, allow_pickle=True)

            self.N = result['solution'].shape[-1] - 1

            yield result



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # input : batch_size, 1, 101, 101 
            nn.Conv2d(1, 5, 4),  # batch_size, 5, 109, 109
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # batch_size, 5, 54, 54 
            nn.Conv2d(5, 10, 3),  # batch_size, 10, 52, 52
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # batch_size, 10, 25, 25
            nn.ReLU(),
            nn.Conv2d(10, 20, 3),  #batch_size, 3, 23, 23
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3), #batch_size, 20, 7, 7  
            nn.Conv2d(20, 30,  4), #batch_size, 30, 4, 4
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(30, 20, 3, stride=2),  # b, 20, 9, 9
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, 3, stride=3, padding=1),  # b, 10, 25, 25
            nn.ReLU(),
            nn.ConvTranspose2d(10, 5, 3, stride=2, padding=1),  # b, 5, 49, 49
            nn.ReLU(),
            nn.ConvTranspose2d(5, 2, 3, stride=2), #batch_size, 2, 99, 99
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3), #batch_size, 1, 101, 101
            nn.Tanh()
        )

    def forward(self, x):
        abs_x = torch.abs(x)
        batch_max_x = abs_x.max(axis=2).values.max(axis=1).values
        batch_max_x = batch_max_x.view(-1, 1, 1)
        x = x / batch_max_x
        batch_size = x.shape[0]
        N = x.shape[1] - 1
        x = x.view(batch_size, 1, N + 1, N + 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, N + 1, N + 1) * batch_max_x
        return x

