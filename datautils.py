import torch
import numpy as np
from torch.utils.data import DataLoader


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


def load_data(dataset_name, data_size, batch_size):
    x = np.load('./datasets/' + f'{dataset_name}/' + f'cluster_x_{data_size}.npy')
    y = np.load('./datasets/' + f'{dataset_name}/' + f'cluster_y_{data_size}.npy')
    dataset = Mydataset(x, y)
    print('n_cluster:', np.unique(y))

    torch.manual_seed(123)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    return data_loader
