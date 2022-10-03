import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_loss(data, axis, title="Metric Training Loss", label="Dynamic", xlabel="Epoch", ylabel="Loss"):
    axis.set_title(title)
    axis.plot(list(range(len(data))), data, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()

def scatter_XY(X,Y, axis, title="Metric Training Loss", label="Dynamic", xlabel="Epoch", ylabel="Loss", s = 0.05):
    axis.set_title(title)
    axis.scatter(X, Y, label=label, s=s)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()

def seismic_heat_map(data, fig, axis, title="Metric Training Loss", xlabel="Epoch", ylabel="Loss"):

    cmap = cm.get_cmap('seismic', 256)
    psm = axis.pcolormesh(data, cmap=cmap, rasterized=True, vmin=np.min(data), vmax=np.max(data))
    fig.colorbar(psm, ax=axis)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)