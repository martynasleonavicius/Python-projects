# -*- coding: utf-8 -*-

"""
This script is used to analyze 
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def score_analysis(scores, path, type_t):
    
    #Scores is a 2d array. We need to transform it into 1d    
    all_scores = []
    for k in scores:
        all_scores += k

        
    average = sum(all_scores)/len(all_scores)
    std = np.std(all_scores, ddof=1)
    plt.axvline(average, color='red', linestyle='dashed', linewidth=2, label=f"Mean {round(average, 2)}")
    plt.axvspan(average-std, average+std, color='green', alpha=0.3, label=f"±1 Std Dev {round(std, 2)}")
    plt.hist(all_scores, bins = 20)
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('Score')
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/score_analysis_{type_t}.png", bbox_inches='tight', dpi=400)
    plt.show()