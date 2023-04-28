"""Style file including helper functions and specific plot/print functions"""

import matplotlib.pyplot as plt
import numpy as np
from util import convert_to_numpy
import torch


### Helper functions

# Return modified string that prints to bold
bold = lambda string: '\033[1m' + string + '\033[0m'

# Return modified string that prints to italics
italic = lambda string: '\x1B[3m' + string + '\x1B[0m'


### Plot functions

def get_top_k_for_plot(grad, k):
    """Return top k gradients for plotting"""
    max_idx = np.argsort(np.abs(grad))[-k:]
    new_grads = np.zeros(len(grad))
    new_grads[max_idx] = grad[max_idx]
    return new_grads

def plot_grads(grads, nrows, ncols, k=-1):
    """grads has size (no. grads, no. features)"""
    fig, axs = plt.subplots(nrows, ncols, dpi=100)
    fig.set_figwidth(6*ncols)
    fig.set_figheight(4*nrows)
    n_features = len(grads[0])

    for i in range(nrows):
        for j in range(ncols):
            # Select top k features (if applicable)
            if k != -1:
                grad = get_top_k_for_plot(grads[i*ncols+j], k)
            else:
                grad = grads[i*ncols+j]
            # Account for singletons
            if nrows==1:
                ax = axs[j]
            else:
                ax = axs[i,j]
            ax.bar(range(n_features), grad)

def plot_single_grad(grad, k=-1):
    """Plot gradients for a single model and test data point"""
    n_features = len(grad)
    plt.figure(dpi=100)
    if k != -1:
        grad = get_top_k_for_plot(grad, k)
    plt.bar(range(n_features), grad)
    plt.show()

def plot_accuracies(
    accs: list[np.ndarray],
    random_sources: list[str],
    bins: int = 50,
):
    """Plot accuracy histograms (columns represent different random sources)"""

    nrows, ncols = 1, len(random_sources)
    fig, axs = plt.subplots(nrows, ncols, dpi=150)
    fig.set_figwidth(6*ncols)

    # Ensure axs is always a list
    if ncols==1:
        axs = [axs]

    for i, random_source in enumerate(random_sources):
        axs[i].hist(accs[i], bins=bins)
        mean = round(accs[i].mean(), 2)
        std = round(accs[i].std(), 2)
        title = f'Mean Accuracy: {mean}%, Std. Dev.: {std}'
        axs[i].set_title(r"$\bf{" + random_source.upper() + '}$\n' + title)
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlabel('Accuracy (%)')

    plt.show()

def plot_figure_3a(
    name: str,
    disagreements_partial: np.ndarray,
    traditional_disagreements: np.ndarray,
    selective_disagreements: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    """Plot Figure 3a from the paper (including selective ensembles)"""

    plt.figure(dpi=150, figsize=(5,4))
    colors = ['blue', 'lime', 'black', 'red']

    for i, random_source in enumerate(random_sources):
        # Account for singletons
        x_axis = [1] + ensemble_sizes

        # Traditional Ensembles
        y_axis = [disagreements_partial[i].mean()*100] + list(traditional_disagreements[i]*100)
        plt.plot(x_axis, y_axis, '-^', markersize=10, label=random_source.upper(), color=colors[i])

        # Selective Ensembles
        y_axis =  [disagreements_partial[i].mean()*100] + list(selective_disagreements[i]*100)
        plt.plot(x_axis, y_axis, '-^', markersize=10,
                 label=random_source.upper()+' (Sel)', color=colors[i+2])

    plt.xticks(x_axis)
    plt.title(name.title(), fontsize=18)
    plt.ylabel('% Individuals With ' + r'$p_{flip}>0$', fontsize=13)
    plt.xlabel('Number of Models Per Ensemble', fontsize=13)
    plt.legend()
    plt.show()

    # Print caption
    print("\nFigure 3a: Percentage of test data with non-zero disagreement")
    print("rate in traditional (i.e., majority vote but not selective")
    print("ensembles. Horizontal axis depicts ensemble size.")
    print("\nThe paper claims that \"while ensembling alone mitigates")
    print("much of the prediction instability, it is unable to eliminate")
    print("it as selective ensembles do\". We discuss our opposing findings.")


### Print functions

def print_table_1(
    name: str,
    accs: list[np.ndarray],
    random_sources: list[str],
):
    """Print Table 1 from the paper"""

    print(bold(f'\nRandomness\t\t{name.title()}'))
    print('-'*40)
    for i, random_source in enumerate(random_sources):
        mean = round(accs[i].mean(), 2)
        std = round(accs[i].std(), 2)
        print(f'{random_source.upper()}\t\t\t{mean}\u00B1{std}')
        print('-'*40)

    # Print caption
    print("\nTable 1: Mean accuracy over 200 models")
    print("trained over changes to random initialization")
    print("and leave-one-out differences in training data.")

def print_table_2(
    name: str,
    disagreements_full: list[float],
    disagreements_partial: list[np.ndarray],
    selective_disagreements: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    """Print Table 2 from the paper"""

    print(bold(f'\nRandomness\t\tn\t\t\t{name.title()}'))
    print('-'*60)

    for i, random_source in enumerate(random_sources):
        print(f'{random_source.upper()}\t\t\t1 (full)\t\t{disagreements_full[i]}')
        dis_partial = round(disagreements_partial[i].mean(), 4)
        print(f'{random_source.upper()}\t\t\t1 (partial)\t\t{dis_partial}')
        for j, ensemble_size in enumerate(ensemble_sizes):
            dis = selective_disagreements[i,j]
            print(f'{random_source.upper()}\t\t\t{ensemble_size}\t\t\t{dis}')
        print('-'*60)

    # Print caption
    print("\nTable 2: Proportion of points with disagreement")
    print("between at least one pair of models (p_flip>0)")
    print("trained with different random seeds (RS) or leave-")
    print("one-out (LOO) differences in training data for")
    print("singleton models (n=1) and selective ensembles (n>1).")
    print("All 200 models are used in the (full) comparison, while")
    print("10 models or ensembles are used in the remaining rows.")

def print_table_3(
    name: str,
    selective_accs: np.ndarray,
    abstention_rates: np.ndarray,
    traditional_accs: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    """Print Table 3 from the paper"""

    key = "accuracy (abstain as error) " \
          "| abstention rate " \
          "| non-selective accuracy"
    print('\n' + bold('Key: ') + key)
    print(bold(f'\nRandomness\t\tn\t\t    {name.title()}'))
    print('-'*60)

    for i, random_source in enumerate(random_sources):
        for j, ensemble_size in enumerate(ensemble_sizes):
            acc = str(selective_accs[i,j].mean())[:4]
            abstain = str(abstention_rates[i,j].mean())[:4]
            traditional = str(traditional_accs[i,j].mean())[:4]
            print(f'{random_source.upper()}\t\t\t{ensemble_size}\t\t{acc}|{abstain}|{traditional}')
        print('-'*60)

    # Print caption
    print("\nTable 3: Accuracy and abstention rate of selective ensembles,")
    print("along with the accuracy of non-selective (traditional ensembles)")
    print("with n ∈ {5,10,15,20} constituents. Results are averaged over 10")
    print("randomly selected models.")

def print_preds_memory(
    preds: np.ndarray | torch.Tensor,
):
    """Print memory usage of prediction arrays"""

    # Convert to numpy if a tensor
    preds = convert_to_numpy(preds)

    # Print preds size
    n_models, n_inputs = preds.shape
    print(f'{n_models} models, {n_inputs} test inputs')

    # Single model memory
    bytes_per_input = preds.itemsize
    bytes = n_inputs * bytes_per_input

    # Determine prefix (KB, MB, GB, etc.) and print
    print(f'{prefix_memory(bytes)} memory (single model predictions)')

    # All models memory
    bytes = n_models*n_inputs*bytes_per_input
    
    # Determine prefix (KB, MB, GB, etc.) and print
    print(f'{prefix_memory(bytes)} memory (all model predictions)')

def prefix_memory(
    memory: int,
):
    """Prefix memory usage with KB, MB, GB, etc."""

    prefixes = ['B', 'KB', 'MB', 'GB']
    for prefix in prefixes:
        if memory < 10**3:
            break
        memory /= 10**3
    return f'{round(memory, 1)}{prefix}'
