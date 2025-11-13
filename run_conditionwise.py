## runQUOT.py
## simple wrapper for track_file to wrap with sbatch

from quot.core import track_file
import argparse
import pickle
import os
import re
import sys


print("")
print('parsing Conditionwise arguments')
parser = argparse.ArgumentParser(description="Process an input file.")
parser.add_argument('-s', '--settingsPKL', required=True, help='Input settings Pickle file path')
parser.add_argument('-c', '--condition', required=False, help='Input condition')

args = parser.parse_args()

print(f"reading settings from: {args.settingsPKL}")
print(f"Working on condition: {args.condition}")

with open(args.settingsPKL , "rb") as f:
    settings = pickle.load(f)
    
sys.path.append(settings['io']['code_directory'])
from init_settings import *
import trackingCodeFunctions
import plotFunctions



cond = args.condition
### read in data
posterior_df = trackingCodeFunctions.aggregate_csv(settings['io']['post_directory'] , f"{cond}*.csv")
posterior_df['condition'] = [re.sub("_[0-9]*_traj.csv$" , '' , f) for f in posterior_df['file']]

MLE_df = trackingCodeFunctions.aggregate_csv(settings['io']['MLE_directory'] , f"{cond}*.csv")
MLE_df['file_basename'] = [os.path.basename(f) for f in MLE_df['source_file']]
MLE_df['condition'] = [re.sub("_[0-9]*_traj.csv$" , '' , f) for f in MLE_df['file_basename']]

rollingMLE = trackingCodeFunctions.aggregate_csv(settings['io']['rolling_window_directory'] , f"{cond}*.csv")
rollingMLE['file_basename'] = [re.sub("::[0-9]*$" , '.csv' , traj) for traj in rollingMLE['ur_trajectory']]
rollingMLE['condition'] = [re.sub("_[0-9]*_traj.csv$" , '' , traj) for traj in rollingMLE['file_basename']]


## QC histograms
outliers = plotFunctions.find_outlier_filenames(posterior_df , mult_on_sd = settings['plot']['mult_on_sd'])

print('')
print(f"allowed mult_on_sd is {settings['plot']['mult_on_sd']}")
print('Removing outliers:')
for ol in outliers:
    print(f"    {ol}")
print('')

posterior_df = posterior_df[~posterior_df['file'].isin(outliers)]
MLE_df = MLE_df[~MLE_df['file_basename'].isin(outliers)]
rollingMLE = rollingMLE[~rollingMLE['file_basename'].isin(outliers)]


## plot histograms & bargraphs
posterior_plot_df = plotFunctions.generate_posterior_plotting_df(posterior_df)
plotFunctions.generate_plots(MLE_df , posterior_plot_df , posterior_df , settings , showPlot = False)


## run HMM
## print HMM parameters & catch updated rollingMLE
rollingMLE = plotFunctions.fitHMM(rollingMLE , settings , cond , min_obs=7)

## handle the case that slowSPT should cont all data, and not trim ends with nan
if settings['saspt']['frame_interval'] > 1:
    rolling_MLE['posterior_pop'] = 0
    rolling_MLE['naive_pop'] = 0
    
## Plot empirical state survival from posterior of rollingMLE
plotFunctions.plot_survival_HMM(rollingMLE , settings , cond , 0 , showPlot = False , min_run = 0)

print('finished HMM')
## save rollingMLE one-file-per-condition in a directory called someting like pooled_rolling_windows/{cond}_rollingMLE.csv

## save PKL of the files that contain plotting data
plotting_pkl_dir = f"{settings['io']['plot_directory']}/plotting_pkls"
os.makedirs(plotting_pkl_dir, exist_ok=True)

plotting_pkl = f"{settings['io']['plot_directory']}/plotting_pkls/{cond}.pkl"
with open(plotting_pkl, "wb") as f:
   pickle.dump({'MLE_df' : MLE_df , 
                'posterior_plot_df' : posterior_plot_df , 
               'rollingMLE' : rollingMLE}, 
               f)

## calcualte bleaching curves and save bleaching curves
bc = plotFunctions.aggregate_bleaching_data(rollingMLE , settings , n_particles=250)
plotFunctions.plot_bleaching_curves(settings , bc)


## save movies

plotFunctions.generate_movies(rollingMLE , settings)

##### TBD

