import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json

def plot_pred_obs(
        predicted, 
        observed, 
        color_variable=None,
        folds=None,
        cmap='Set1',
        label='',
        out_name=None, 
        show=False,
        text_note=None
        ):

        fig, ax = plt.subplots(figsize=(6, 6))
        # Set the aspect ratio to equal for a square plot
        ax.set_aspect('equal', adjustable='box')

        if folds:
            # Assign different colors for each fold
            unique_folds = set(folds)
            # colors = plt.cm.Set1(np.linspace(0, 1, len(unique_folds)))
            cmap = plt.get_cmap(cmap)
            cmap_iter = iter(cmap.colors)
            colors = [next(cmap_iter) for _ in range(len(unique_folds))]

            for fold, color in zip(unique_folds, colors):
                fold_indices = np.where(np.array(folds) == fold)[0]
                ax.scatter(
                    np.array(observed)[fold_indices],
                    np.array(predicted)[fold_indices],
                    label=f'{fold+1}',
                    color=color,
                    alpha=0.7,
                    marker='+'
                )
            ax.legend(title='Fold', bbox_to_anchor=(0.99, 0.01), loc='lower right')

        else:
            # Create a scatter plot of observed vs. predicted values
            if not color_variable:
                color_variable='black'

            scatter = ax.scatter(
                    observed, 
                    predicted, 
                    c=color_variable,
                    cmap=cmap,
                    marker='+', 
                    alpha=0.7,
                    )
            if color_variable != 'black':
                plt.colorbar(scatter, label=label, shrink=0.8)


        # Add a diagonal line representing perfect predictions (x=y)
        amin = min(min(observed), min(predicted))
        amax = max(max(observed), max(predicted))
        x = np.linspace(amin, amax, 100)

        ax.plot(x, x, linestyle='--', c='r')

        ax.set_xlabel('Observed Values')
        ax.set_ylabel('Predicted Values')

        if text_note:
            ax.text(
                    0.1, 
                    0.9, 
                    text_note, 
                    transform=ax.transAxes, 
                    fontsize=12, 
                    va='top'
                    )

        # Show the plot
        if show:
            plt.show()

        # fig.savefig(os.path.join(out_dir, f'pred_obs_{target_variable}.png'))
        if out_name:
            fig.savefig(out_name)

def get_colors_for_values(clist, cmap_name='Set1'):
    """
    Get a list of colors corresponding to unique values using a specified colormap.

    Parameters:
    - clist (array-like): Array or list of categorical variables.
    - cmap_name (str): Name of the colormap to use.

    Returns:
    - colors (list): List of colors corresponding to unique values.
    """

    norm = (clist - clist.min()) / (clist.max() - clist.min())
    
    # Get the colormap
    cmap = cm.get_cmap(cmap_name)
    
    # Get a list of colors corresponding to unique values
    colors = [cmap(value) for value in norm]
    
    return colors


def correct_json_format(json_path):    #for dinov2 training metrics json
    with open(json_path, 'r+') as file : 
        a = file.read()
        splitted = a.split('}')
        updated_str = splitted[0]
        for i in range(1, len(splitted)-1):
            updated_str += '},' + splitted[i]
        updated_str = '[' + updated_str + '}]'
        file.truncate(0)
        
        file.close()
    file = open(json_path, 'w+')
    file.write(updated_str)
    file.close()

def show_loss(train_metrics_path):

    try:
        data = json.load(open(train_metrics_path))
    except :
        correct_json_format(train_metrics_path)
        data = json.load(open(train_metrics_path))
    N = len(data)
    losses = ['total_loss', 'dino_local_crops_loss', 'dino_global_crops_loss', 'ibot_loss']             #koleo loss

    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    X = np.array([i for i in range(N)])
    for loss in losses :
        Y = np.zeros(N)
        for j in range(N):
            Y[j] = data[j][loss]             
        plt.plot(X, Y, label = loss)
    plt.legend()
    plt.show()
