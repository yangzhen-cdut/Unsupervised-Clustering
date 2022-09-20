import os
import math
import torch
import datautils
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from tscc import TSCCModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='Micrseismic_Timeseries', type=str, help='The dataset name')
parser.add_argument('--dataset_size', default=4928, type=int, help='The size of dataset')
parser.add_argument('--dim', default=1, type=int, help='The dimension of input')
parser.add_argument('--num_cluster', type=int, default=2, help='The number of cluster')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
parser.add_argument('--repr_dims', type=int, default=32, help='The representation dimension')
parser.add_argument('--lr', type=float, default=0.001, help='The learning rate of pre-training phase')
parser.add_argument('--pretraining_epoch', type=int, default=25, help='The epoch of pre-training phase')
parser.add_argument('--MaxIter1', type=int, default=25, help='The epoch of fine-tuning phase')
args = parser.parse_args()

print("Arguments:", str(args))
print('Loading data... \n', end='')

# Load data
data_loader = datautils.load_data(args.dataset_name, args.dataset_size, args.batch_size)

config = dict(dataset_size=args.dataset_size,
              dataset_name=args.dataset_name,
              pretraining_epoch=args.pretraining_epoch,
              batch_size=args.batch_size,
              MaxIter1=args.MaxIter1,
              lr=args.lr,
              output_dims=args.repr_dims)

model = TSCCModel(data_loader, n_cluster=args.num_cluster, input_dims=args.dim, **config)

model.encoder = torch.load('Micrseismic_Timeseries_Finetuning_phase')
print('finish inital')
model.encoder.eval()


def eval_with_real_data(save=False):
    data = np.zeros([args.dataset_size, 500, 1])
    reps = np.zeros([args.dataset_size, 500, 32])

    ii = 0
    for x, target in data_loader:
        x = Variable(x).cuda()
        u = model.encoder(x)
        u = u.cpu()
        reps[ii * args.batch_size:(ii + 1) * args.batch_size, :, :] = u.data.numpy()
        data[ii * args.batch_size:(ii + 1) * args.batch_size, :, :] = x.cpu().numpy()
        ii = ii + 1
    if save:
        np.save(os.getcwd() + '\\Eval_Representations.npy', reps)
        np.save(os.getcwd() + '\\Eval_Data.npy', data)
    print('finish')


def eval_with_synthetic_data(save=False):
    # Ricker with White Gaussian Noise, SNR = -5dB
    n = 500
    wt = Ricker(n)
    np.random.seed(123)
    noise = np.random.normal(loc=0, scale=1.0, size=(len(wt),))
    nwt = add_noise(wt, noise, -5)
    nwt = nwt / np.max(abs(nwt))  # de-mean

    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.plot(nwt, c='#9B3A4D', linewidth=1.5)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.title('Noisy Ricker with SNR=-5dB', fontsize=20, family='Calibri')
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')

    # Synthetic Noise
    plt.subplot(212)
    t = np.linspace(0, n-1, n)
    lowfre_noise = np.sin((t) * np.pi / 100)
    np.random.seed(123)
    random_noise = np.random.normal(loc=0, scale=1.0, size=(500,))
    noise = random_noise + lowfre_noise
    noise = noise / np.max(abs(noise))
    plt.plot(noise, c='#70A0AC', linewidth=1.5)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.title('Synthetic Noise', fontsize=20, family='Calibri')
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.tight_layout()
    plt.show()

    syndata = np.zeros((2, 500, 1))
    syndata[0, :, 0] = nwt
    syndata[1, :, 0] = noise

    syndata_in = Variable(torch.tensor(syndata, dtype=torch.float32)).cuda()

    syn_reps = model.encoder(syndata_in)
    syn_reps = syn_reps.detach().cpu().numpy()
    if save:
        np.save(os.getcwd() + '\\Eval_Syn_Representations.npy', syn_reps)
        np.save(os.getcwd() + '\\Eval_Syn_Data.npy', np.squeeze(syndata, axis=2))
    print('finish')


def Ricker(n, f0=20, dt=0.001):
    wt = np.zeros(n)
    i = 0
    for k in range(int(-n / 2), int(n / 2)):
        wt[i] = (1 - 2.0 * (math.pi * f0 * k * dt) ** 2) * math.exp(-1 * (math.pi * f0 * k * dt) ** 2)
        i += 1

    return wt


def add_noise(x, noise, SNR):
    """
    :param x: pure signal (np.array)
    :param noise: noise = random.normal(0,1)
    :param SNR: signal-to-noise ratio
    :return: noisy_signal
    """
    try:
        x = np.array(x)
    except:
        pass

    N = len(x.tolist())
    noise = noise - np.mean(noise)
    signal_power = 1.0 / N * sum(x ** 2)
    noise_variance = signal_power / (math.pow(10, SNR / 10))
    NOISE = math.sqrt(noise_variance) / np.std(noise) * noise
    noisy_signal = x + NOISE
    return noisy_signal


if __name__ == '__main__':
    eval_with_synthetic_data()