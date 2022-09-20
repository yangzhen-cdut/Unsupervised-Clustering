import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from models.Metrics import nmi, acc

import seaborn as sns
sns.set()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plotter(S, y, name, loc, save_fig=False):
    ''' function to visualize the outputs of t-SNE '''

    legend_properties = {'family': 'Calibri', 'size': '16'}
    target_names = ['Microseismic', 'Noise']
    colors = ['#9B3A4D', '#70A0AC']
    lw = 0.6
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(S[y == i, 0], S[y == i, 1], c=color, alpha=0.5, lw=lw, s=40, label=target_name)
    plt.legend(loc=loc, shadow=True, scatterpoints=1, prop=legend_properties, facecolor='white', frameon=False)
    plt.tick_params(labelsize=12)
    plt.title(name, fontsize=20, family='Calibri')


def comparison_clustering(save_fig=False):
    filename = './features/'
    f = plt.figure(figsize=(18, 12))

    # K-means on Time Series
    ax = plt.subplot(231)
    x = np.load('./dataset/Time Series/cluster_x_4928.npy')
    x = np.squeeze(x, axis=2)
    y = np.load('./dataset/Time Series/cluster_y_4928.npy')
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
    label_pred = kmeans.labels_
    ACC = acc(y, label_pred, 2)
    NMI = nmi(y, label_pred)
    print('ACC', ACC)
    print('NMI', NMI)
    redu = TSNE(n_components=2, random_state=50).fit_transform(x)
    plotter(redu, y, name='K-means', loc='upper left')
    ax.axis('off')
    ax.axis('tight')

    # DEC
    ax = plt.subplot(232)
    enc = np.load(filename + 'DEC.npy')
    y = np.load(filename + 'DEC_y_true.npy')
    redu = TSNE(n_components=2, random_state=123).fit_transform(enc)
    plotter(redu, y, name='DEC', loc='upper right')
    ax.axis('off')
    ax.axis('tight')

    # DCA
    ax = plt.subplot(233)
    enc = np.load(filename + 'DCA.npy')
    y = np.load(filename + 'DCA_y_true.npy')
    redu = TSNE(n_components=2, random_state=50).fit_transform(enc)
    plotter(redu, y, name='DCA', loc='upper right')
    ax.axis('off')
    ax.axis('tight')

    # DCSS
    ax = plt.subplot(234)
    enc = np.load(filename + 'DCSS.npy')
    y = np.load(filename + 'DCSS_y_true.npy')
    redu = TSNE(n_components=2, random_state=64).fit_transform(enc)
    plotter(redu, y, name='DCSS', loc='upper left')
    ax.axis('off')
    ax.axis('tight')

    # TSCC Pre-training
    ax = plt.subplot(235)
    enc = np.load(filename + 'End_Pretraining_u.npy')
    y = np.load(filename + 'End_Pretraining_y_true.npy')
    redu = TSNE(n_components=2, random_state=88).fit_transform(enc)
    plotter(redu, y, name='TSCC Pre-training', loc='upper left')
    ax.axis('off')
    ax.axis('tight')

    # TSCC Fine-tuning
    ax = plt.subplot(236)
    enc = np.load(filename + 'End_Finetuning_u.npy')
    y = np.load(filename + 'End_Finetuning_y_true.npy')
    redu = TSNE(n_components=2, random_state=0).fit_transform(enc)
    plotter(redu, y, name='TSCC Fine-tuning', loc='upper right')
    ax.axis('off')
    ax.axis('tight')

    plt.tight_layout()
    plt.show()
    if save_fig:
        f.savefig('./results/comparison_results.png', dpi=600)


def representation_visualization(save_fig=False):
    data = np.load('./Eval_Data.npy')
    data = np.squeeze(data, axis=2)
    reps = np.load('./Eval_Representations.npy')
    p1 = 2290
    p2 = 1171
    p3 = 3279
    p4 = 387
    f = plt.figure(figsize=(15, 10))
    plt.subplot(421)
    plt.plot(data[p1, :]/max(abs(data[p1, :])), c='#9B3A4D', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(390, 0.65, 'Microseismic', fontsize=18, family='Calibri')

    plt.subplot(422)
    plt.plot(data[p2, :]/max(abs(data[p2, :])), c='#9B3A4D', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(390, 0.65, 'Microseismic', fontsize=18, family='Calibri')

    plt.subplot(425)
    plt.plot(data[p3, :]/max(abs(data[p3, :])), c='#70A0AC', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(450, 0.65, 'Noise', fontsize=18, family='Calibri')

    plt.subplot(426)
    plt.plot(data[p4, :]/max(abs(data[p4, :])), c='#70A0AC', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(450, 0.65, 'Noise', fontsize=18, family='Calibri')

    # Customize Colormap
    from matplotlib.colors import LinearSegmentedColormap
    import scipy.io as scio
    colormap = scio.loadmat('./colormap.mat')['mycamp']
    moreland_map = LinearSegmentedColormap.from_list('cos', colormap)

    plt.subplot(423)
    ax = sns.heatmap(reps[p1, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()

    plt.subplot(424)
    ax = sns.heatmap(reps[p2, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()

    plt.subplot(427)
    ax = sns.heatmap(reps[p3, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()

    plt.subplot(428)
    ax = sns.heatmap(reps[p4, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()
    if save_fig:
        plt.savefig('./results/reprs.png', dpi=600)
    plt.show()


def syn_representation_visualization(save_fig=False):
    data = np.load('./Eval_Syn_Data.npy')
    reps = np.load('./Eval_Syn_Representations.npy')

    f = plt.figure(figsize=(7.5, 10))
    plt.subplot(411)
    plt.plot(data[0, :], c='#9B3A4D', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(365, 0.76, 'Synthetic ricker', fontsize=18, family='Calibri')

    plt.subplot(413)
    plt.plot(data[1, :], c='#70A0AC', linewidth=2)
    plt.tick_params(labelsize=14)
    plt.margins(x=0)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Amplitude', fontsize=20, family='Calibri')
    plt.text(365, 0.73, 'Synthetic noise', fontsize=18, family='Calibri')

    # Customize Colormap
    from matplotlib.colors import LinearSegmentedColormap
    import scipy.io as scio
    colormap = scio.loadmat('./colormap.mat')['mycamp']
    moreland_map = LinearSegmentedColormap.from_list('cos', colormap)

    plt.subplot(412)
    ax = sns.heatmap(reps[0, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()

    plt.subplot(414)
    ax = sns.heatmap(reps[1, :, :].T, cmap=moreland_map, vmin=-2, vmax=2, xticklabels=False, yticklabels=False, cbar=False)
    ax.axis('on')
    plt.xticks(ticks=[0, 100, 200, 300, 400, 500], labels=[0, 100, 200, 300, 400, 500], rotation=0, fontsize=14)
    plt.yticks(ticks=[0, 16], labels=[32, 16], rotation=0, fontsize=14)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    plt.xlabel('Timestamp', fontsize=20, family='Calibri')
    plt.ylabel('Repr dims', fontsize=20, family='Calibri')
    plt.tight_layout()

    if save_fig:
        plt.savefig('./results/syn_reprs.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    #comparison_clustering()
    syn_representation_visualization(save_fig=False)