from src.utils.constants import CLASSES

# from torch import tensor
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np


def plot_reconstruction(ground_truth, reconstruction):
    # plot 4 reconstructions and the ground truth
    fig, axs = plt.subplots(ncols=2, nrows=4, squeeze=True, figsize=(10, 10))
    choices = np.random.choice(len(ground_truth), 4)
    for idx, choice in enumerate(choices):
        prediction = reconstruction[choice].cpu().detach().numpy().reshape(50, 50)
        truth = ground_truth[choice].cpu().detach().numpy().reshape(50, 50)
        axs[idx, 0].imshow(np.flip(truth, 0), cmap="plasma")
        axs[idx, 0].set_title("Ground Truth")
        axs[idx, 0].axes.xaxis.set_visible(False)
        axs[idx, 0].axes.yaxis.set_visible(False)

        axs[idx, 1].imshow(np.flip(prediction, 0), cmap="plasma")
        axs[idx, 1].set_title("Prediction")
        axs[idx, 1].axes.xaxis.set_visible(False)
        axs[idx, 1].axes.yaxis.set_visible(False)

    fig.suptitle("Reconstruction of Persistence Images")
    plt.close()
    return fig


def plot_cm(predicted, ground_truth, dataset, name=""):
    # true = (
    #     torch.argmax(true, dim=-1).detach().float().cpu().numpy()
    #     if true.dim() > 1
    #     else true.detach().float().cpu().numpy()
    # )
    # pred = (
    #     torch.argmax(pred, dim=-1).detach().float().cpu().numpy()
    #     if pred.dim() > 1
    #     else pred.detach().float().cpu().numpy()
    # )
    if isinstance(predicted, np.ndarray):
        true = (
            np.argmax(ground_truth, axis=-1).astype(int)
            if ground_truth.ndim > 1
            else ground_truth.astype(int)
        )
        pred = (
            np.argmax(predicted, axis=-1).astype(int)
            if predicted.ndim > 1
            else predicted.astype(int)
        )

    cm = confusion_matrix(y_true=true, y_pred=pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    classes = CLASSES[dataset]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix - {name}",
        ylabel="True label",
        xlabel="Predicted label",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    ax.figure.colorbar(im, ax=ax)
    plt.close("all")
    return fig
