import os
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSCCEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from models.Metrics import nmi, acc


class TSCCModel:
    ''' The TSCC model '''

    def __init__(
            self,
            data_loader,
            dataset_size,
            batch_size,
            pretraining_epoch,
            n_cluster,
            dataset_name,
            input_dims,
            MaxIter=100,
            m=1.5,
            T1=2,
            output_dims=32,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=0.001,
            max_train_length=3000,
            temporal_unit=0):

        ''' Initialize a TS2Vec model '''

        super().__init__()
        self.device = device
        self.lr = lr
        self.num_cluster = n_cluster
        self.batch_size = batch_size
        self.T1 = T1
        self.m = m
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.dataset_name = dataset_name
        self.latent_size = output_dims
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self.u_mean = torch.zeros([n_cluster, output_dims])

        self.encoder = TSCCEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self.encoder)
        self.net.update_parameters(self.encoder)

    def Pretraining(self):
        print('Pretraining...')
        self.encoder.train()
        self.encoder.cuda()
        for param in self.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(self.encoder.parameters(), lr=self.lr)
        prev_ACC = 0
        loss_log = []
        acc_log = []
        nmi_log = []
        for T in range(0, self.pretraining_epoch):
            print('Pretraining Epoch: ', T + 1)
            for x, target in self.data_loader:
                optimizer.zero_grad()
                x = Variable(x).cuda()
                out1, out2 = self.cropping(x, tp_unit=self.temporal_unit, model=self.encoder)
                loss = self.contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.encoder)
            loss_log.append(loss)

            ACC, NMI = self.Kmeans_model_evaluation(T)
            acc_log.append(ACC)
            nmi_log.append(NMI)

            if ACC > prev_ACC:
                prev_ACC = ACC
                with open(self.dataset_name+'_Pretraining_phase', 'wb') as f:
                    torch.save(self.encoder, f)
            print(f"Epoch #{T + 1}: loss={loss}")

        file = os.getcwd() + '\\pretraining.csv'
        data = pd.DataFrame.from_dict({'pretraining': loss_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')
        data.to_csv(file, index=False)

        self.encoder = torch.load(self.dataset_name + '_Pretraining_phase')
        self.plotter(name=self.dataset_name + '_Pretraining_phase', save_fig=False)
        return self.encoder

    def Finetuning(self):
        self.encoder, self.u_mean = self.initialization()
        self.encoder.cuda()
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(self.encoder.parameters(), lr=0.0001)
        ACC_prev = 0.0
        loss_log = []
        acc_log = []
        nmi_log = []
        for T in range(0, self.MaxIter1):
            print('Finetuning Epoch: ', T + 1)
            if T % self.T1 == 1:
                self.u_mean = self.update_cluster_centers()
            for x, target in self.data_loader:
                u = torch.zeros([self.num_cluster, self.batch_size, self.latent_size]).cuda()
                x = Variable(x).cuda()
                for kk in range(0, self.num_cluster):
                    y = self.encode_with_pooling(x)
                    u[kk, :, :] = y.cuda()
                u = u.detach()
                p = self.cmp(u, self.u_mean.cuda())
                p = p.detach()
                self.u_mean = self.u_mean.cuda()
                p = p.T
                p = torch.pow(p, self.m)
                for i in range(0, self.num_cluster):
                    out1, out2 = self.cropping(x, tp_unit=self.temporal_unit, model=self.encoder)
                    u1 = self.encode_with_pooling(x)
                    self.u_mean = self.u_mean.float()
                    loss_c = torch.matmul(p[i, :].unsqueeze(0), torch.sum(torch.pow(u1 - self.u_mean[i, :].unsqueeze(0).repeat(self.batch_size, 1), 2), dim=1))
                    loss_r = self.contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)
                    loss = loss_r + 0.1 * loss_c
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.net.update_parameters(self.encoder)
                    loss_log.append(loss)

            ACC, NMI = self.model_evaluation(T)
            acc_log.append(ACC)
            nmi_log.append(NMI)

            if ACC > ACC_prev:
                ACC_prev = ACC
                with open(self.dataset_name + '_Finetuning_phase', 'wb') as f:
                    torch.save(self.encoder, f)
                with open(self.dataset_name + '_Centers', 'wb') as f:
                    torch.save(self.u_mean, f)
            print(f"Epoch #{T + 1}: loss={loss}")

        file = os.getcwd() + '\\finetuning.csv'
        data = pd.DataFrame.from_dict({'finetuning': loss_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')  # orient='columns'
        data.to_csv(file, index=False)

        self.plotter(name=self.dataset_name + '_Finetuning_phase', save_fig=False)

    def initialization(self):
        print("-----initialization mode--------")
        self.encoder = torch.load('AE_' + self.dataset_name + '_pretrain')
        self.encoder.cuda()
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            u = self.encode_with_pooling(x)
            u = u.cpu()
            datas[(ii) * self.batch_size:(ii + 1) * self.batch_size] = u.data.numpy()
            ii = ii + 1
        # datas = datas.cpu()
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)
        self.u_mean = kmeans.cluster_centers_
        self.u_mean = torch.from_numpy(self.u_mean)
        self.u_mean = Variable(self.u_mean).cuda()
        return self.encoder, self.u_mean

    def Kmeans_model_evaluation(self, T):
        self.encoder.eval()
        datas = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            u = self.encode_with_pooling(x)
            u = u.cpu()
            datas[ii * self.batch_size:(ii + 1) * self.batch_size, :] = u.data.numpy()
            label_true[ii * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1

        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)

        label_pred = kmeans.labels_
        ACC = acc(label_true, label_pred, self.num_cluster)
        NMI = nmi(label_true, label_pred)
        print('ACC', ACC)
        print('NMI', NMI)
        if T == 0:
            np.save('./features/Start_Pretraining_R.npy', datas)
            np.save('./features/Start_Pretraining_y_true.npy', label_true)
        if T == self.pretraining_epoch-1:
            np.save('./features/End_Pretraining_R.npy', datas)
            np.save('./features/End_Pretraining_y_true.npy', label_true)
        return ACC, NMI

    def update_cluster_centers(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).cuda()
        num = torch.zeros([self.num_cluster, self.latent_size]).cuda()
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            u = self.encode_with_pooling(x)
            p = self.cmp(u.unsqueeze(0).repeat(self.num_cluster, 1, 1), self.u_mean)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].T, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])
        self.encoder.cuda()
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
        return self.u_mean

    def cmp(self, u, u_mean):
        p = torch.zeros([self.batch_size, self.num_cluster]).cuda()
        for j in range(0, self.num_cluster):
            p[:, j] = torch.sum(torch.pow(u[j, :, :] - u_mean[j, :].unsqueeze(0).repeat(self.batch_size, 1), 2), dim=1)
        p = torch.pow(p, -1 / (self.m - 1))
        sum1 = torch.sum(p, dim=1)
        p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
        # print(p[1,:])
        return p

    def model_evaluation(self, T):
        datas = np.zeros([self.dataset_size, self.latent_size])
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            u = self.encode_with_pooling(x)
            datas[ii * self.batch_size:(ii + 1) * self.batch_size, :] = u.data.cpu().numpy()
            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            pred_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = y
            true_labels[(ii) * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1

        ACC = acc(true_labels, pred_labels, self.num_cluster)
        NMI = nmi(true_labels, pred_labels)
        print('ACC', ACC)
        print('NMI', NMI)
        if T == 0:
            np.save('./features/Start_Finetuning_R.npy', datas)
            np.save(f'./features/Start_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./features/Start_Finetuning_y_true.npy', true_labels)
        if T == self.MaxIter1-1:
            np.save(f'./features/End_Finetuning_End_Finetuning_R.npy', datas)
            np.save(f'./features/End_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./features/End_Finetuning_y_true.npy', true_labels)
        self.encoder.cuda()
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        return ACC, NMI

    def encode_with_pooling(self, data):
        assert data.ndim == 3
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        if torch.is_tensor(data):
            dataset = TensorDataset(data)
        else:
            dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(self.device, non_blocking=True))
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1), ).transpose(1, 2)
                out = out.cpu()
                out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        if torch.is_tensor(data):
            return output.to(self.device)
        else:
            return output.numpy()

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def plotter(self, name, save_fig=False):
        print('Evaluation')
        legend_properties = {'family': 'Calibri', 'size': '16'}
        target_names = ['Microseismic', 'Noise']
        colors = ['#9B3A4D', '#70A0AC']
        self.encoder.eval()
        label_true = np.zeros(self.dataset_size)
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        for x, target in self.data_loader:
            x = Variable(x).cuda()
            u = self.encode_with_pooling(x)
            u = u.cpu()
            datas[ii * self.batch_size:(ii + 1) * self.batch_size, :] = u.data.numpy()
            label_true[ii * self.batch_size:(ii + 1) * self.batch_size] = target.numpy()
            ii = ii + 1
        redu = TSNE(n_components=2, random_state=123).fit_transform(datas)
        lw = 0.6
        f = plt.figure(figsize=(6, 6))
        ax = f.add_subplot(111)
        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(redu[label_true == i, 0], redu[label_true == i, 1], color=color, alpha=0.5, lw=lw, s=40,
                        label=target_name)
        plt.legend(loc='lower left', shadow=True, scatterpoints=1, prop=legend_properties, facecolor='w', frameon=False)
        ax.axis('off')
        ax.axis('tight')
        if save_fig:
            f.savefig(f'./{name}.png', dpi=600)
        plt.close(f)

    def instance_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)
        z = z.transpose(0, 1)
        sim = torch.matmul(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def temporal_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1)
        sim = torch.matmul(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def contrastive_loss(self, z1, z2, alpha=0.5, temporal_unit=0):
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_loss(z1, z2)
            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_loss(z1, z2)
            d += 1
        return loss / d

    def cropping(self, x, tp_unit, model):
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (tp_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        indx1 = crop_offset + crop_eleft
        num_elem1 = crop_right - crop_eleft
        all_indx1 = indx1[:, None] + np.arange(num_elem1)

        indx2 = crop_offset + crop_left
        num_elem2 = crop_eright - crop_left
        all_indx2 = indx2[:, None] + np.arange(num_elem2)

        out1 = model(x[torch.arange(all_indx1.shape[0])[:, None], all_indx1])
        out1 = out1[:, -crop_l:]

        out2 = model(x[torch.arange(all_indx2.shape[0])[:, None], all_indx2])
        out2 = out2[:, :crop_l]

        return out1, out2
