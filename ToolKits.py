import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def Plot_ROC_Curves(y_true,y_score):
    '''
    Dont forget to do this: during with torch.no_grad
    y_true.extend(target.cpu().numpy().tolist())
    y_score.extend(torch.nn.functional.softmax(s_output, dim=1).cpu().numpy().tolist())
    '''
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    num_classes = len(np.unique(y_true))
    fpr = []
    tpr = []
    weights = np.bincount(y_true) / len(y_true)
    overall_fpr = np.linspace(0, 1, 100)  # 定义整体的fpr范围
    overall_tpr = 0
    for i in range(num_classes):
        y_true_binary = np.where(y_true == i, 1, 0)
        y_score_single_class = y_score[:, i]

        fpr_i, tpr_i, _ = roc_curve(y_true_binary, y_score_single_class)
        roc_auc = auc(fpr_i, tpr_i)

        fpr.append(fpr_i)
        tpr.append(tpr_i)
        auc_scores.append(roc_auc)

        overall_tpr += np.interp(overall_fpr, fpr_i, tpr_i) * weights[i]

    overall_auc = auc(overall_fpr, overall_tpr)

    fig, ax = plt.subplots()
    ax.plot(overall_fpr, overall_tpr, color='b', label='ROC Curve (AUC = {:.2f})'.format(overall_auc))

    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.grid(True)
    plt.show()    

def Plot_Confusion_Matrix(all_targets, all_preds):
    conf_mat = confusion_matrix(all_targets, all_preds)
    num_classes = len(target_test_loader.dataset.classes)

    fig, ax = plt.subplots(figsize=(8, 6))


    im = ax.imshow(conf_mat, cmap='Blues')


    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(target_test_loader.dataset.classes, rotation=90, ha='right', fontsize=18)
    ax.set_yticklabels(target_test_loader.dataset.classes, fontsize=18)
    ax.tick_params(axis='both', which='both', length=0, pad=2)
    ax.grid(visible=False)


    thresh = conf_mat.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(conf_mat[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black", fontsize=18)

    plt.tight_layout()
    plt.savefig(location)





