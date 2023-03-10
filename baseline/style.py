import matplotlib.pyplot as plt
import numpy as np
from util import convert_to_numpy
import torch

### Helper functions

def bold(
    string: str,
):
    return '\033[1m' + string + '\033[0m'

def italic(
    string: str,
):
    return '\x1B[3m' + string + '\x1B[0m'


### Plot functions

def plot_accuracies(
    accs: list[np.ndarray],
    random_sources: list[str],
):
    nrows, ncols = 1, len(random_sources)
    fig, ax = plt.subplots(nrows, ncols, dpi=150)
    fig.set_figwidth(6*ncols)
    for i, random_source in enumerate(random_sources):
        ax[i].hist(accs[i], bins=50)
        mean = round(accs[i].mean(), 2)
        std = round(accs[i].std(), 2)
        title = f'Mean Accuracy: {mean}%, Std. Dev.: {std}'
        ax[i].set_title(r"$\bf{" + random_source.upper() + '}$\n' + title)
        ax[i].set_ylabel('Frequency')
        ax[i].set_xlabel('Accuracy (%)')
    plt.show()

def plot_figure_3a(
    disagreements_partial: np.ndarray,
    traditional_disagreements: np.ndarray,
    selective_disagreements: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    plt.figure(dpi=150, figsize=(5,4))
    colors = ['blue', 'lime', 'black', 'red']
    for i, random_source in enumerate(random_sources):
        x_axis = [1]+ensemble_sizes

        # Traditional Ensembles
        y_axis = [disagreements_partial[i].mean()*100] + list(traditional_disagreements[i]*100)
        plt.plot(x_axis, y_axis, '-^', markersize=10, label=random_source.upper(), color=colors[i])

        # Selective Ensembles
        y_axis =  [disagreements_partial[i].mean()*100] + list(selective_disagreements[i]*100)
        plt.plot(x_axis, y_axis, '-^', markersize=10, label=random_source.upper()+' (Sel)', color=colors[i+2])
    plt.xticks(x_axis)
    plt.title('FMNIST', fontsize=18)
    plt.ylabel('% Individuals With ' + r'$p_{flip}>0$', fontsize=13)
    plt.xlabel('Number of Models Per Ensemble', fontsize=13)
    plt.legend()
    plt.show()


### Print functions

def print_preds_memory(
    preds: np.ndarray | torch.Tensor,
):
    # Convert to numpy if a tensor
    preds = convert_to_numpy(preds)

    # Print preds size
    n_models, n_inputs = preds.shape
    print(f'{n_models} models, {n_inputs} test inputs')

    # Print memories
    bytes_per_input = preds.itemsize
    print(f'{int(n_inputs*bytes_per_input/10**3)}KB memory (single model predictions)')
    print(f'{int(n_models*n_inputs*bytes_per_input/10**6)}MB memory (all model predictions)')

def print_table_1(
    accs: list[np.ndarray],
    random_sources: list[str],
):
    print(bold('Randomness\t\tFMNIST'))
    print('-'*40)
    for i, random_source in enumerate(random_sources):
        mean = round(accs[i].mean(), 2)
        std = round(accs[i].std(), 2)
        print(f'{random_source.upper()}\t\t\t{mean}\u00B1{std}')
        print('-'*40)
    print("\nTable 1: Mean accuracy over 200 models")
    print("trained over changes to random initialization")
    print("and leave-one-out differences in training data.")

def print_table_2(
    disagreements_full: list[float],
    disagreements_partial: list[np.ndarray],
    selective_disagreements: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    print(bold('Randomness\t\tn\t\t\tFMNIST'))
    print('-'*60)
    for i, random_source in enumerate(random_sources):
        print(f'{random_source.upper()}\t\t\t1 (full)\t\t{disagreements_full[i]}')
        dis_partial = round(disagreements_partial[i].mean(), 4)
        print(f'{random_source.upper()}\t\t\t1 (partial)\t\t{dis_partial}')
        for j, n in enumerate(ensemble_sizes):
            dis = selective_disagreements[i,j]
            print(f'{random_source.upper()}\t\t\t{n}\t\t\t{dis}')
        print('-'*60)
    print("Table 2: Proportion of points with disagreement")
    print("between at least one pair of models (p_flip>0)")
    print("trained with different random seeds (RS) or leave-")
    print("one-out (LOO) differences in training data for")
    print("singleton models (n=1) and selective ensembles (n>1).")
    print("All 200 models are used in the (full) comparison, while")
    print("10 models or ensembles are used in the remaining rows.")

def print_table_3(
    selective_accs: np.ndarray,
    abstention_rates: np.ndarray,
    traditional_accs: np.ndarray,
    ensemble_sizes: list[int],
    random_sources: list[str],
):
    print('accuracy (abstain as error) | abstention rate | non-selective accuracy')
    print(bold('Randomness\t\tn\t\t    FMNIST'))
    print('-'*60)
    for i, random_source in enumerate(random_sources):
        for j, n in enumerate(ensemble_sizes):
            acc = str(selective_accs[i,j].mean())[:4]
            abstain = str(abstention_rates[i,j].mean())[:4]
            traditional = str(traditional_accs[i,j].mean())[:4]
            print(f'{random_source.upper()}\t\t\t{n}\t\t{acc}|{abstain}|{traditional}')
        print('-'*60)
    print("Table 3: Accuracy and abstention rate of selective ensembles,")
    print("along with the accuracy of non-selective (traditional ensembles)")
    print("with n ∈ {5,10,15,20} constituents. Results are averaged over 10")
    print("randomly selected models.")