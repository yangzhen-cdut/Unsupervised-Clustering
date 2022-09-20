import time
import datetime
import argparse
import datautils
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
parser.add_argument('--MaxIter', type=int, default=25, help='The epoch of fine-tuning phase')
args = parser.parse_args()

print("Arguments:", str(args))
print('Loading data... \n', end='')

# Load data
data_loader = datautils.load_data(args.dataset_name, args.dataset_size, args.batch_size)

config = dict(dataset_size=args.dataset_size,
              dataset_name=args.dataset_name,
              pretraining_epoch=args.pretraining_epoch,
              batch_size=args.batch_size,
              MaxIter1=args.MaxIter,
              lr=args.lr,
              output_dims=args.repr_dims)

model = TSCCModel(data_loader, n_cluster=args.num_cluster, input_dims=args.dim, **config)

t = time.time()

if args.pretraining_epoch != 0:
    model.Pretraining()
if args.MaxIter1 != 0:
    model.Finetuning()

t = time.time() - t

print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
