# -*- coding: utf-8 -*-
"""

"""

#%%Dataset analysis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


        
def ploting_results(dataset, name, name_in_dataset, path, type_t):
    #Calculate the average
    average = get_average(dataset, name_in_dataset)
    #and standard deviation
    standard_dev = get_standard_dev(dataset, name_in_dataset)
    
    
    #plot histogram
    plt.hist(np.array(dataset[name_in_dataset]), bins = 100, density = False)
    plt.xlabel(name)
    plt.ylabel('Counts')
    #Now plot the mean and the standard deviation
    plt.axvspan(average - standard_dev, average + standard_dev, color='green', alpha=0.3, label=f"±1 Std Dev {round(standard_dev, 6)}")
    plt.axvline(average, color='red', linestyle='dashed', linewidth=2, label=f"Mean {round(average, 6)}")
    plt.axvline(average + standard_dev, color='green', linestyle='dotted', linewidth=2)
    plt.axvline(average - standard_dev, color='green', linestyle='dotted', linewidth=2)
    plt.legend()
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{name}.png", bbox_inches='tight', dpi=400)
    plt.show()
    
#%%
def get_average(dataset, name_in_dataset):
    return sum(dataset[name_in_dataset])/len(dataset)

def get_standard_dev(dataset, name_in_dataset):
    return  float(np.std(dataset[name_in_dataset], ddof=1))

#%%
def save_statistics_summary(dataset, path, type_t, units):
    with open(f"{path}/summary_{type_t}.txt", 'w') as f:
        f.write(f'Anisotropy:  {get_average(dataset, "Anisotropy")} std: {get_standard_dev(dataset, "Anisotropy")}\n')
        f.write(f'Area:  {get_average(dataset, f"Area {units}^2")} std: {get_standard_dev(dataset, f"Area {units}^2")}\n')
        f.write(f'Aspect Ratio:  {get_average(dataset, f"Aspect Ratio Width/Height")} std: {get_standard_dev(dataset, "Aspect Ratio Width/Height")}\n')
        f.write(f'Circularity:  {get_average(dataset, f"Circularity")} std: {get_standard_dev(dataset, "Circularity")}\n')
        f.write(f'Eccentricity:  {get_average(dataset, f"Eccentricity")} std: {get_standard_dev(dataset, "Eccentricity")}\n')
        f.write(f'Solidity:  {get_average(dataset, f"Solidity")} std: {get_standard_dev(dataset, "Solidity")}\n')
        f.write(f'Perimeter:  {get_average(dataset, f"Perimeter_{units}")} std: {get_standard_dev(dataset, f"Perimeter_{units}")}\n')
        

#%%

def dataset_analysis(statistics, units, pixel_size, path, type_t):
    #First, see how many disjointed masks there are
    more_than_one = []
    for i in range(len(statistics)):
        for j in statistics[i]:
            if isinstance(j, pd.DataFrame):
                if j['regions'] > 1:
                    more_than_one.append([i+3, j])
                
    #To make the dataset wide statistical analysis,
    #we put all the dataframes from statistics array into a single dataframe
    dataset_df  = pd.DataFrame()
    for g in statistics:
        if isinstance(g, pd.DataFrame):
            dataset_df = pd.concat([dataset_df, g])
    
    
    dataset_df[f'Area {units}^2'] = dataset_df['Area_2_in_px']*pixel_size*pixel_size
    dataset_df[f'Perimeter_{units}'] = dataset_df['Perimeter']*pixel_size
   
    
    """Area"""
    ploting_results(dataset_df,"Area", f'Area {units}^2', path, type_t)
    
    """Eccentricity"""
    ploting_results(dataset_df,"Eccentricity", 'Eccentricity', path, type_t)
    
    """Solidity"""
    ploting_results(dataset_df,"Solidity", 'Solidity', path, type_t)
    
    """Perimeter"""
    ploting_results(dataset_df,"Perimeter", f'Perimeter_{units}', path, type_t)
    
    """Circularity"""
    ploting_results(dataset_df,"Circularity", 'Circularity', path, type_t)
        
    """Anisotropy"""
    ploting_results(dataset_df,"Anisotropy", 'Anisotropy', path, type_t)
    
    """Aspect Ratio"""
    ploting_results(dataset_df,"Aspect", 'Aspect Ratio Width/Height', path, type_t)
    
    save_statistics_summary(dataset_df, path, type_t, units)
        
        
        