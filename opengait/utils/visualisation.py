from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visual_summary(feat, labs, name):
    tsne = TSNE()
    trn_tsne_embeds = tsne.fit_transform(feat)
    scatter(trn_tsne_embeds, labs, str(name)+"_Embeddings")


# Define our own plot function
def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 11))
    c = palette[labels]

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, c=labels, s=40, cmap="Spectral")
    legend1 = ax.legend(*sc.legend_elements(), loc="upper right", title="Ids")
    ax.add_artist(legend1)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
           
    if subtitle != None:
        plt.suptitle(subtitle)
    plt.savefig(subtitle, dpi=1000)