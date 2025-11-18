## many dependencies, including...
## python3.8 , quot , saspt , numpy , ggplot , argparse , and many others	

## this strategy is designed under a "map/reduce" framework to take advantage of the
## cluster design

## for each file,l it performs tracking on a separate cluster node, then when all of the
## files are done for a given condition, it starts another job to jointly analyze
## all of the trajectories from that condition and plot the results

## there are many other hacks / workarounds that I haven't fully reconciled, including
## ones that are designed for other workflows like SlowSPT (very long shutter speeds)
## or when i explicitly want to fit a markov model to the state, or when I want to generate
## a bargraph of the various states.

## Eventually I will make a more stable version of this code that will be more usable,
## but this should run once you install all the dependencies

## just use the command... bash ./command.sh from the test directory.

nohup /lab/tambora_li/Gavin/conda_env_dir/saspt_env/bin/python3 ~/software/spt_tracking_code/quot_workflow/clean_workflow/analyze_directory.py --input_directory ./ &
