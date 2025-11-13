## runQUOT.py
## simple wrapper for track_file to wrap with sbatch

from quot.core import track_file
import argparse
import pickle
import os
import re
import sys


print("")
print('parsing QUOT arguments')
parser = argparse.ArgumentParser(description="Process an input file.")
parser.add_argument('-i', '--inputfile', required=True, help='Input ND2 file path')
parser.add_argument('-s', '--settingsPKL', required=True, help='Input settings Pickle file path')
parser.add_argument('-c', '--condition', required=False, help='Input condition')

args = parser.parse_args()

print(f"ND2 input file: {args.inputfile}")
print(f"reading settings from: {args.settingsPKL}")

if args.condition != None:
    print(f"Working on condition: {args.condition}")

with open(args.settingsPKL , "rb") as f:
    settings = pickle.load(f)

    
sys.path.append(settings['io']['code_directory'])
from init_settings import *
import trackingCodeFunctions

input_basename = os.path.basename(args.inputfile)
quot_outcsv = os.path.join(settings['io']['traj_directory'] , re.sub('.nd2' , '_traj.csv' , input_basename))

print(f"Saving trajectories to {quot_outcsv}")

### update settings with dt and image background once per file, subsetting on condition
## should handle the case where one mega-analysis includes files with different dt's
settings = update_settings_with_image_metadata(settings , args.condition)


print('\n  Tracking')
track_file(args.inputfile , out_csv = quot_outcsv , **settings['quot'])

print('\n  Executing SASPT')
trackingCodeFunctions.executeSASPT(quot_outcsv , settings)

print('\n  Analyzing rolling windows')
trackingCodeFunctions.analyze_rolling_windows(quot_outcsv , settings)


