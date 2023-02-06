import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import functools

def plot_loss(data, axis, title="Metric Training Loss", label="Dynamic", xlabel="Epoch", ylabel="Loss"):
    """
    This function plots the training loss against the epoch on a specified axis.
    
    Parameters:
    data (list): A list of loss values for each epoch.
    axis (matplotlib axis object): The axis on which to plot the loss.
    title (str, optional): The title of the plot. Default is "Metric Training Loss".
    label (str, optional): The label for the line plot. Default is "Dynamic".
    xlabel (str, optional): The label for the x-axis. Default is "Epoch".
    ylabel (str, optional): The label for the y-axis. Default is "Loss".
    
    Returns:
    None
    """
    
    axis.set_title(title)
    axis.plot(list(range(len(data))), data, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()

    
def scatter_XY(X,Y, axis, title="Scatter Plot", label="Dynamic", xlabel="X", ylabel="Y", s = 0.05):
    """
    This function creates a scatter plot of two arrays X and Y on a specified axis.
    
    Parameters:
    X (list or ndarray): An array of x-values for the scatter plot.
    Y (list or ndarray): An array of y-values for the scatter plot.
    axis (matplotlib axis object): The axis on which to plot the scatter plot.
    title (str, optional): The title of the plot. Default is "Scatter Plot".
    label (str, optional): The label for the scatter plot. Default is "Dynamic".
    xlabel (str, optional): The label for the x-axis. Default is "X".
    ylabel (str, optional): The label for the y-axis. Default is "Y".
    s (float, optional): The size of the markers in the scatter plot. Default is 0.05.
    
    Returns:
    None
    """
    
    axis.set_title(title)
    axis.scatter(X, Y, label=label, s=s)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()

def grid_map(data, fig, axis, title="Pairwise Loss", xlabel="Epoch", ylabel="Loss", cmap = cm.get_cmap('seismic', 256)):
    """
    Plot a grid map of the input data.

    Parameters:

    data : numpy array
        The data to be plotted as a grid map.
    fig : matplotlib.figure
        The figure object to plot the grid map on.
    axis : matplotlib.axis
        The axis object to plot the grid map on.
    title : str, optional
        The title of the plot, by default "Metric Training Loss".
    xlabel : str, optional
        The label for the x axis, by default "Epoch".
    ylabel : str, optional
        The label for the y axis, by default "Loss".

    Returns:
    
    None
    """
    # Plot the grid map on the axis object
    psm = axis.pcolormesh(data, cmap=cmap, rasterized=True, vmin=np.min(data), vmax=np.max(data))
    # Add color bar to the figure
    fig.colorbar(psm, ax=axis)
    # Set title and axis labels
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

def plot_array(data, fig, plot_fn):
    """
    plot_array: Function creates a subplot grid of size (sq_side x sq_side) from an input data array,
    and applies the specified plotting function (plot_fn).
    
    Parameters:
    data (ndarray): Input data of shape (n, m, p), where n is the plot batch dimension, m and p are the height and width of the data, respectively.
    fig (matplotlib figure object): The figure to which the subplots will be added.
    transformation (function): A function to be applied to each element in the batch dimension
    
    Returns:
    None

    Side Effects:
    Displays the image in the specified matplotlib axis and modifies the appearance of the axis.

    Example:
    fig, ax = plt.subplots()
    plot_fn = functools.partial(plot_image, fig = fig, cmap = cmap)
    plot_array(data, fig, plot_fn)
    """
    
    #Compute side length of array
    sq_side = int(data.shape[0]**(1/2))
    
    #Create subplot grid with side length of sq_side
    top_fig = fig.subplots(sq_side, sq_side)
    
    #Iterate through each subplot
    for i in range(sq_side):
        for j in range(sq_side):
            plot_fn(data[i*sq_side + j], top_fig[i, j])

def plot_image(data, fig, cmap = 'gray'):
    """
    plot_image: Function to plot an image using matplotlib.pyplot

    Parameters:
    data (ndarray): 2-D numpy array of image data
    fig (matplotlib.pyplot axis): axis to display the image
    cmap (str, optional): color map to use, default is 'gray'

    Returns:
    None

    Side Effects:
    Displays the image in the specified matplotlib axis and modifies the appearance of the axis. Will automatically normalize if the data is not of type uint8 between 0 and 255 or float between 0 and 1.

    Example:
    fig, ax = plt.subplots()
    plot_image(image_data, fig)
    """
    if len(data.shape) == 3:
        if data.shape[0] == 3:
            data = np.transpose(data, (1,2,0))

    #Check if image data is valid
    if isinstance(data, np.ndarray) == False:
        raise TypeError("Image data must be a numpy array")

    if np.issubdtype(data.flat[0], np.integer):
        if np.max(data) > 255 or np.min(data) < 0:
            #Normalize image data
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data.astype('uint8')

    elif np.issubdtype(data.flat[0], np.floating):
        if np.max(data) > 1 or np.min(data) < 0:
            #Normalize image data
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise TypeError("Image data must be of type int or float")

    #Display image in current subplot
    fig.imshow(data, cmap=cmap)
    
    #Remove grid and ticks from subplot
    fig.grid(False)
    fig.set_xticks([])
    fig.set_yticks([])
    
    #Remove spines from subplot
    for key, spine in fig.spines.items():
        spine.set_visible(False)

def image_array(data, fig, cmap = 'gray'):
    """
    image_array: Function to plot an array of images using matplotlib.pyplot

    Parameters:
    data (ndarray): 3-D numpy array of image data, with the shape (num_images, height, width)
    fig (matplotlib.pyplot figure): figure to display the image array
    cmap (str, optional): color map to use, default is 'gray'

    Returns:
    None

    Side Effects:
    Displays the array of images in the specified matplotlib figure.

    Example:
    fig, ax = plt.subplots()
    image_array(images_data, fig)
    """
    plot_fn = functools.partial(plot_image, cmap = cmap)
    plot_array(data, fig, plot_fn)
            
