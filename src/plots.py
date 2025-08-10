import matplotlib.pyplot as plt
import numpy as np
from train import rf_results, fh_results
import config


# Function to plot performance metrics
def plot_performance(seeds, auc_results, dauprc_results, model_name):
    mean_auc = [np.mean(auc_results[i]) if i < len(auc_results) else np.nan for i in range(len(seeds))]
    std_auc = [np.std(auc_results[i]) if i < len(auc_results) else np.nan for i in range(len(seeds))]
    mean_dauprc = [np.nanmean(dauprc_results[i]) if i < len(dauprc_results) else np.nan for i in range(len(seeds))]
    std_dauprc = [np.nanstd(dauprc_results[i]) if i < len(dauprc_results) else np.nan for i in range(len(seeds))]

    # Plot ROC AUC
    plt.figure()
    plt.errorbar(seeds, mean_auc, yerr=std_auc, fmt='-o', capsize=5, label='ROC AUC')
    plt.title(f"{model_name} - Mean ROC AUC Score Across Seeds")
    plt.xlabel("Seed")
    plt.ylabel("Mean ROC AUC")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot DAUPRC
    plt.figure()
    plt.errorbar(seeds, mean_dauprc, yerr=std_dauprc, fmt='-o', capsize=5, label='DAUPRC')
    plt.title(f"{model_name} - Mean DAUPRC Score Across Seeds")
    plt.xlabel("Seed")
    plt.ylabel("Mean DAUPRC")
    plt.ylim(0, 0.1)
    plt.legend()
    plt.grid()
    plt.show()


plot_performance(config.SEEDS, rf_results[0], rf_results[2], model_name="Random Forest")
plot_performance(config.SEEDS, fh_results[0], fh_results[2], model_name="Frequent Hitters")