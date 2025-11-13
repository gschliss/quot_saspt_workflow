README

Psuedocode of analysis pipeline 2025_11_11

ToDo:
- include a YAML with the dependency specification
- write a try-it-out Jupyter notebook


Parallel version:
- Read in metadata & extract unique conditions
- Initiate a SLURM job for every condition
    - Main job is post-run plotting, HMM
        - check for dependencies
            - depends: _traj.csv ; _rollingTraj.csv ; _posterior.csv ; _MLE.csv
        - If dependencies don't exist, make them
            - run job-array once per file to generate dependencies
        - Aggreagte _posterior.csv , _MLE.csv , rollingTraj.csv
        - QC _posterior.csv to remove outliers ; remove corresponding trajectories from _MLE & rollingTraj.csv
        - Plot histograms, bar graphs
        - Run HMM
        - print HMM
        - save _rollingTraj.csv with posterior population assignment
        - plot survival curves for each population
        - save DF of survival curve data
        - save movies
- Aggregate across conditions


Future structure... constantly run a helper script looking for new raw_data files, and generate file-wise data ASAP using Watchdog in python