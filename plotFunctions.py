from trackingCodeFunctions import *
import os
import sys
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from PIL import Image, ImageDraw

plt.rcParams['font.family'] = 'DejaVu Sans'

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import glob

import imageio
import pims
from tifffile import imwrite

import warnings
warnings.filterwarnings("ignore")


def find_outlier_filenames(posterior_df , mult_on_sd = 2):
    outliers = []
    grouped_density = [
        group["density"].to_numpy()
        for _, group in posterior_df.groupby("file")
    ]
    group_names = posterior_df["file"].drop_duplicates().tolist()
    outliers.append(find_outlier_positions(grouped_density  , group_names , mult_on_sd = mult_on_sd))
    outliers = [item for sublist in outliers for item in sublist]
    return outliers




## this does the statistical *work* underlying the mark_outliers function
## starting from the posterior populations... calculate the median of the
## posterior probabilty for each diffusion rate. for each XY position, calculate
## the KS-distance from the median distribution, and keep any XY position within mult_on_sd
## of that distribution
def find_outlier_positions(grouped_density , group_names , mult_on_sd = 2):

    # Assume `pdfs` is a list or array of 1D arrays (n_pdfs x n_points)
    pdfs = np.array(grouped_density)

    # Step 1: Compute the empirical CDFs
    cdfs = np.cumsum(pdfs, axis=1)
    cdfs = cdfs / cdfs[:, -1][:, np.newaxis]  # Normalize each to 1

    # Step 2: Compute the median CDF
    median_cdf = np.median(cdfs, axis=0)

    # Step 3: KS test between each PDF's CDF and the median CDF
    def ks_statistic(empirical, reference):
        return np.max(np.abs(empirical - reference))

    ks_stats = [ks_statistic(cdf, median_cdf) for cdf in cdfs]

    # Step 4: Flag outliers — e.g., using robust threshold
    threshold = np.median(ks_stats) + mult_on_sd * np.std(ks_stats) ## SD is weird statistic because distribution is one-sided
    outliers = [i for i, stat in enumerate(ks_stats) if stat > threshold]
    outlier_files = [group_names[i] for i in outliers]

#     print("Outlier indices:", outliers)
#     # Print the result
#     for file in outlier_files:
#         print(f"  - {file}:")
    return(outlier_files)

## this calculates the weighted average posterior diffusion rate, i.e. it marginalizes over the whol
## population of molecules, then scales by the population size to make it a PDF
## returns a plottable table of the posterior MLE diffusion rate, with a column to store the condition name
def generate_posterior_plotting_df(posterior_df):
    posterior_plot_df = posterior_df.copy()
    posterior_plot_df = posterior_plot_df.groupby("D")["density"].mean().reset_index()
    posterior_plot_df['density'] = posterior_plot_df['density'] / posterior_plot_df['density'].sum()
    posterior_plot_df['condition'] = np.unique([re.sub('_[0-9]*_traj.csv' , '' , f) for f in posterior_df['file']])[0]
    return(posterior_plot_df)




## take MLE_df and scaled posterior_plot_df and generate a histogram and bar graph
## The safe usage of this function is to pass only one condition. could likely be modified
## to accept many conditions, and plot many plots into a single PDF, but that hasn't been implemented yet
## lightgray lines correspond to file-wise PDFs, black line is mean PDF
def generate_plots(MLE_df , posterior_plot_df , all_posteriors , settings , showPlot = True , conditions = []):

    n = len(conditions)
    if n == 0:
        conditions = MLE_df['condition'].unique()

    bin_labels = settings['plot']['barGraphLabels']
    barGraphs = pd.DataFrame(index=bin_labels)

    #print(f"Working on conditions: {conditions}")

    for condition in conditions:
        sm_df = posterior_plot_df[posterior_plot_df['condition'] == condition]
        sm_MLE = MLE_df[MLE_df['condition'] == condition]
        sm_all_posterior = all_posteriors[all_posteriors['condition']==condition]

        normalized_df = sm_all_posterior.copy()
        normalized_df['density_norm'] = normalized_df.groupby('file')['density'].transform(lambda x: x / x.sum())

        # Create a new figure for each condition
        #fig, ax = plt.subplots(figsize=(8, 2.5))
        fig, axes = plt.subplots(
            1, 2,                   # 1 row, 2 columns
            figsize=(9, 2.5 * len(conditions)),        # total figure size = 8 + 1 width, 2.5 height
            gridspec_kw={'width_ratios': [8, 1]}  # relative widths
        )

        ax_hist, ax_bar = axes

        ax_hist.hist(np.log10(sm_MLE['MLE_D']) , bins = np.linspace(-2,2,settings['plot']['nbins']) , density = True , color="#D28F8F");

        for f, group in normalized_df.groupby('file'):
            ax_hist.plot(np.log10(group['D']), 50 * group['density_norm'], linewidth=1.5, color='dimgray' , alpha = 0.4)

        ax_hist.plot(np.log10(sm_df['D']) , 50 * sm_df['density'] , linewidth = 1.5 , color = 'black')


        tick_positions = [-2, -1, 0, 1, 2]
        tick_labels = [f"{10**v:.2f}" if 10**v < 1 else f"{10**v:.0f}" for v in tick_positions]
        ax_hist.set_xticks(tick_positions, tick_labels);

        minor_ticks = []
        for base in [0.01, 0.1, 1, 10]:
            for i in range(2, 10):
                tick = base * i
                if tick <= 100:
                    minor_ticks.append(tick)
        minor_ticks = np.log10(minor_ticks)
        ax_hist.set_xticks(minor_ticks, minor=True)
        ax_hist.tick_params(axis='x', which='minor', length=4, color='gray')

        ax_hist.set_ylim(settings['plot']['ylim'][0] , settings['plot']['ylim'][1])

        ax_hist.set_title(f"{condition} ; n_jumps = {settings['saspt']['splitsize']} ; {len(sm_MLE['MLE_D'])} particles", pad=20)
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Probability Density")

        for spine in ['top', 'right']:
            ax_hist.spines[spine].set_visible(False)

        counts, _ = np.histogram(sm_MLE['MLE_D'], bins=settings['plot']['barGraphBreaks']);
        barGraphs[condition] = counts

        barGraphs = barGraphs.T
        barGraphs = barGraphs.div(barGraphs.sum(axis=1), axis=0)
        barGraphs.plot(kind="bar", stacked=True, ax=ax_bar, edgecolor='black')
        #ax_bar.set_xlabel("Condition")
        ax_bar.set_ylabel("Fraction")
        if len(barGraphs) == 1:
            ax_bar.set_xticklabels([]) ## if there is only one condition, suppress title
        ax_bar.legend(title="Population", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save to PDF with condition in filename
        fig.savefig(f"{settings['io']['plot_directory']}/{condition}_hist.pdf", dpi = 300 , format="pdf", bbox_inches='tight')

        if showPlot:
            plt.show()

        plt.close(fig)  # Close to avoid memory buildup

        
        
def plot_survival_by_metadata(settings, metadata_fields = ['send' , 'rec'] , subset = [] , min_detections=2, xlim=120 , title = ''):
    # Build a mapping from field -> value -> list of files
    metadata_groups = {field: defaultdict(list) for field in metadata_fields}

    file_list = [os.path.join(settings['io']['traj_directory'] , fi) for fi in os.listdir(settings['io']['traj_directory']) if fi.endswith('csv')]
    file_pattern = [re.sub(r"_\d{3}_traj\.csv$", "", f) for f in file_list]
    file_pattern = list(dict.fromkeys(file_pattern))

    for f in file_pattern:
        metadata = extract_metadata(f, metadata_fields)
        for field in metadata_fields:
            if field in metadata:
                metadata_groups[field][metadata[field]].append(f)

    for field in metadata_groups:
        if subset:
            for key, files in metadata_groups[field].items():
                # For this key, collect all subset-matching files *in subset order*
                filtered_files = []
                for sub in subset:
                    matches = [f for f in files if sub in f]
                    filtered_files.extend(matches)
                metadata_groups[field][key] = filtered_files
    for field in metadata_groups:
        keys_to_delete = [key for key, files in metadata_groups[field].items() if not files]
        for key in keys_to_delete:
            del metadata_groups[field][key]

    # Generate one plot per unique metadata value
    for field in metadata_fields:
        for value, pattern in metadata_groups[field].items():
            plt.figure(figsize=(8, 5))
            for p in pattern:
                matching_files  = glob.glob(p + '*.csv')
                dat = pd.DataFrame()
                for f in matching_files:
                    newDat = pd.read_csv(f);
                    newDat['trajectory'] = f"{f}::" + newDat['trajectory'].astype(str)
                    dat = pd.concat([dat , newDat] , ignore_index = True)
                df_relative = dat.copy()
                df_relative["frame_offset"] = df_relative["frame"] - df_relative.groupby("trajectory")["frame"].transform("min")
                df_relative = df_relative[df_relative['frame_offset'] >= (min_detections - 1)]
                df_relative = df_relative.groupby("trajectory").filter(lambda g: g["I0"].mean() > settings['quot']['track']['min_I0'])

                survival_counts = df_relative.groupby("frame_offset")["trajectory"].nunique().sort_index()
                total_trajectories = df_relative["trajectory"].nunique()
                survival_prob = survival_counts / total_trajectories

                                # For each time point, compute survival fraction + CI
                results = []
                for frame_offset, n_alive in survival_counts.items():
                    frac = n_alive / total_trajectories
                    ci_low, ci_high = wilson_ci(n_alive, total_trajectories)
                    results.append((frame_offset, frac, ci_low, ci_high))

                survival_df = pd.DataFrame(results, columns=["relative_frame", "survival_prob", "ci_low", "ci_high"])
                survival_df["time"] = survival_df["relative_frame"] * settings['quot']['track']['frame_interval']

    #                 survival_df = survival_prob.reset_index().rename(columns={
    #                     "frame_offset": "relative_frame",
    #                     "trajectory": "survival_probability"
    #                 })
    #                 survival_df['time'] = survival_df['relative_frame'] * settings['quot']['track']['frame_interval']

                label = os.path.basename(p)
                #plt.plot(survival_df['time'], survival_df['survival_probability'], label=label)
                plt.fill_between(survival_df["time"], survival_df["ci_low"], survival_df["ci_high"], alpha=0.3)
                plt.plot(survival_df["time"], survival_df["survival_prob"], label=label)

            if title == '':
                plotname = f"survival_by_{field}_{value}"
            else:
                plotname = f"survival_by_{field}_{value}_{title}"
            plotfile = os.path.join(settings['io']['plot_directory'] , f"{plotname}.pdf")
            plt.xlabel("Time")
            plt.ylabel("Survival Probability")
            plt.title(f"Survival Curves for {field} = {value}")
            plt.xlim(0, xlim)
            plt.ylim(0, 1)
            plt.legend(fontsize='small')
            plt.tight_layout()
            plt.savefig(plotfile)
            plt.show()
      
      

### take a rolling MLE data frame, and fit an HMM to the states defined in
### settings['plot']. Return the data frame with columns appended for naive and posterior state
### and print HMM parameters to a file in the ['io']['plot_directory']/HMM_output
def fitHMM(rollingMLE , settings , condition , min_obs = 7):
    np.set_printoptions(suppress=True, precision=4)

    nPopulations = len(settings['plot']['barGraphLabels'])

    # dynamic_MLE_df = rollingMLE.copy()
    # dynamic_MLE_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # dynamic_MLE_df.dropna(inplace=True)
    # filtered_df = dynamic_MLE_df.groupby("ur_trajectory").filter(lambda group: len(group) >= min_obs).dropna(subset=["MLE_D"])

    # Clean invalid values only in the MLE_D column
    dynamic_MLE_df = rollingMLE.copy()
    dynamic_MLE_df['MLE_D'] = dynamic_MLE_df['MLE_D'].replace([np.inf, -np.inf], np.nan)

    # Identify which trajectories are long enough
    valid_trajs = (
        dynamic_MLE_df.groupby("ur_trajectory")['frame']
        .transform('count') >= min_obs
    )

    # Make a filtered view, but keep all rows in dynamic_MLE_df
    filtered_df = dynamic_MLE_df[valid_trajs & dynamic_MLE_df['MLE_D'].notna()].copy()
        
    grouped = filtered_df.sort_values(['ur_trajectory', 'frame']).groupby('ur_trajectory')

    data = [group['MLE_D'].values for _, group in grouped if len(group) >= min_obs]

    # Concatenate all trajectories into one long sequence
    X = np.concatenate(data).reshape(-1, 1)
    X = np.log10(X + 1e-10) ## log-transform diffusion rates, i.e. [-2 , 2]
    lengths = [len(traj) for traj in data]

    print("Running HMM")

    #######
    ### i don't understand this -- CHAD did it
    class BoundedMeanHMM(GaussianHMM):
        def __init__(self, n_components=1, mean_bounds=None, **kwargs):
            super().__init__(n_components=n_components, **kwargs)
            self.mean_bounds = mean_bounds

        def _do_mstep(self, stats):
            super()._do_mstep(stats)
            if self.mean_bounds is not None:
                for i, (low, high) in enumerate(self.mean_bounds):
                    self.means_[i, :] = np.clip(self.means_[i, :], low, high)

    ## convert boundaries into allowable ranges for HMM
    eps = 1e-10
    boundaries = [b if b > 0 else eps for b in settings['plot']['barGraphBreaks']]
    zones = list(zip(boundaries[:-1], boundaries[1:]))  # pair consecutive boundaries
    log_zones = [(np.log10(a), np.log10(b)) for a, b in zones]

    naive_pop = np.full_like(X, fill_value=np.nan, dtype=int)
    # Loop over zones
    for i, (low, high) in enumerate(zones):
        mask = (X >= np.log10(low)) & (X < np.log10(high))
        naive_pop[mask] = i


    model = BoundedMeanHMM(
        n_components=nPopulations,
        covariance_type='full',
        mean_bounds=log_zones,
        n_iter = 500 , 
        tol = 1e-8
    )
    model.fit(X)

    ### extract the posterior allocations and add them to dynamic_MLE_df
    gamma = model.predict_proba(X)
    fractions_soft = gamma.mean(axis=0)
    posterior_pop = gamma.argmax(axis = 1)

    
    # posterior_pop should be a 1D array with length == len(X)
    posterior_pop = np.array(posterior_pop)

    # Keep track of where we are in posterior_pop
    start_idx = 0

    # We'll fill a new column in dynamic_MLE_df
    dynamic_MLE_df['posterior_pop'] = np.nan
    dynamic_MLE_df['naive_pop'] = np.nan

    # Iterate over each trajectory (as you did when creating `data`)
    grouped = filtered_df.sort_values(['ur_trajectory', 'frame']).groupby('ur_trajectory')
    for traj_id, group in grouped:
        traj_len = len(group)
        if traj_len <= min_obs:
            continue  # skip short trajectories (matches your original filter)

        # Assign posterior population for this trajectory
        dynamic_MLE_df.loc[group.index, 'posterior_pop'] = posterior_pop[start_idx:start_idx + traj_len]
        dynamic_MLE_df.loc[group.index, 'naive_pop'] = naive_pop[start_idx:start_idx + traj_len]

        start_idx += traj_len
                
        

    ## write the HMM parameters to a file
    #######
    means = model.means_.flatten()  # assuming GaussianHMM from hmmlearn
    sort_order = np.argsort(means)  # ascending order
    #model.startprob_ = model.startprob_[sort_order]
    model.startprob_ = fractions_soft[sort_order]
    model.transmat_ = model.transmat_[sort_order][:, sort_order]
    model.means_ = model.means_[sort_order]
    model.covars_ = model.covars_[sort_order]  # only if using full or diag covariances


    output_dirname = os.path.join(settings['io']['plot_directory'] , 'HMM_output')
    os.makedirs(output_dirname, exist_ok=True)

    output_file = f"{output_dirname}/{condition}_HMM.txt"
    with open(output_file, "w") as f:
        print(f"{condition}" , file = f)
        print("", file = f)
        print("Transition matrix:", file=f)
        print(model.transmat_, file=f)
        print("", file=f)

        print("Means of each state:", file=f)
        print(10**(model.means_), file=f)
        print("", file=f)

        print("Occupancy of each state:", file=f)
        print(model.startprob_, file=f)
        print("", file=f)

        # Uncomment if you want to include covariances
        # print("Covariances:", file=f)
        # print(model.covars_, file=f)
        # print("", file=f)

        print("State lifetimes", file=f)
        transmat = model.transmat_
        lifetimes = 1 / (1 - np.diag(transmat)) * settings['saspt']['frame_interval']

        for i, tau in enumerate(lifetimes):
            print(f"State {i}: expected lifetime = {tau:.4f} seconds", file=f)
    
    return(dynamic_MLE_df)


def run_lengths_to_survival_curve(run_lengths, max_lag=None , min_run = 0):
    """
    Convert a list of run lengths into a survival curve.
    
    Parameters:
    - run_lengths: list or array of integers
    - max_lag: maximum Δt to compute (optional)
    
    Returns:
    - survival_curve: numpy array of survival fractions
    """
    run_lengths = np.array(run_lengths)
    run_lengths = run_lengths[run_lengths > min_run]
    if len(run_lengths) == 0:
        return np.array([])
    
    if max_lag is None:
        max_lag = run_lengths.max()
    
    survival_curve = np.zeros(max_lag)
    
    for dt in range(max_lag):
        # Fraction of runs that last at least dt+1
        survival_curve[dt] = np.sum(run_lengths > dt) / len(run_lengths)
    
    return survival_curve

def get_runs(pops, pop):
    """Return lengths of consecutive runs in population `pop`."""
    is_pop = (pops == pop).astype(int)
    if len(is_pop) == 0:
        return []
    padded = np.pad(is_pop, (1,1), mode='constant')  # pad to detect edges
    diff = np.diff(padded)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    run_lengths = run_ends - run_starts
    
    return run_lengths



def plot_survival_HMM(rolling_MLE , settings , cond , plot_pop , max_lag = 50 , showPlot = False , min_run = 0):
    import matplotlib.pyplot as plt
    import os

    # Make sure posterior_pop is integer
    df = rolling_MLE.copy()
    df = df.dropna(subset=['posterior_pop'])
    df['posterior_pop'] = df['posterior_pop'].astype(int)

    populations = sorted(df['posterior_pop'].unique())

    # Initialize dictionary to hold run lengths
    run_lengths_dict = {pop: [] for pop in populations}

    # Iterate over each trajectory
    for traj_id, traj in df.groupby('ur_trajectory'):
        traj = traj.sort_values('frame')
        pops = traj['posterior_pop'].values

        for pop in populations:
            run_lengths = get_runs(pops, pop)
            run_lengths_dict[pop].extend(run_lengths)

    # Compute survival curve for the specified population
    run_lengths = run_lengths_dict[plot_pop]
    survival_curve = run_lengths_to_survival_curve(run_lengths, max_lag=max_lag , min_run = min_run)

    # Create a new figure explicitly
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(len(survival_curve)) * settings['saspt']['frame_interval'] * 1000, 
            survival_curve, color='blue')

    ax.set_ylim(0,1.05)
    ax.set_xlim(0 , max_lag * settings['saspt']['frame_interval'] * 1000)
    ax.set_xlabel('Time offset Δt (ms)')
    ax.set_ylabel('Fraction of runs surviving')
    ax.set_title(f'{cond} population: {plot_pop}')

    # Save figure
    output_dirname = f"{settings['io']['plot_directory']}/survival_curves/"
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname , f"{cond}_pop={plot_pop}_fast_survival.pdf")
    fig.savefig(output_filename, bbox_inches='tight')

    # Only show if requested
    if showPlot:
        plt.show()
    plt.close(fig)  # close the figure to prevent auto-display


    
    
    ## post-analysis data visualization including...

## QC movies
## Bleaching analysis

## from a data frame of trajectories given by all_traj, extract the trajectories matching
## the name ur_trajectory, and return a padded data frame to pass to movie-making code
def isolate_traj(all_traj , ur_trajectory , half_size = 15 , max_frame = 500):
    this_traj = all_traj[all_traj['ur_trajectory'] == ur_trajectory].copy()
        
    this_traj_center = int(np.mean(this_traj['y'])) , int(np.mean(this_traj['x']))
    timeBox = np.min(this_traj['frame']) - 5 , np.max(this_traj['frame']) + 5
    if timeBox[0] < 0:
        timeBox = 0 , timeBox[1]
    if timeBox[1] > 500:
        timeBox = timeBox[0] , 500
    #print(timeBox)

    spaceBox = {'xmin' : (this_traj_center[1] - half_size) , 
                          'xmax' : (this_traj_center[1] + half_size + 1) ,
                          'ymin' : (this_traj_center[0] - half_size) , 
                          'ymax' : (this_traj_center[0] + half_size + 1)}

    this_traj['og_frame'] = this_traj['frame'] ## store the global frame number where the trajectory started
    this_traj['rel_frame'] = this_traj['frame'] -  np.min(timeBox)
        
    first_frame = this_traj['frame'].min()
    last_frame = this_traj['frame'].max()
    # Generate the new frame numbers
    new_frames = list(range(first_frame - 5, first_frame)) + \
                 list(range(last_frame + 1, last_frame + 6))
    new_frames = [x for x in new_frames if x > 0 & x < max_frame]

    # Create a DataFrame of NaN rows with the same columns
    nan_rows = pd.DataFrame(np.nan, index=range(len(new_frames)), columns=this_traj.columns)
    traj_id = this_traj['ur_trajectory'].iloc[0]

    # Assign the new frame numbers
    nan_rows['frame'] = new_frames
    nan_rows['og_frame'] = new_frames
    #nan_rows['rel_frame'] = new_frames
    nan_rows['ur_trajectory'] = traj_id
    nan_rows['condition'] = this_traj['condition'].iloc[0]
    nan_rows['rel_frame'] = nan_rows['frame'] - np.min(timeBox) ## frame relative to the bounding box time interval start
    
    bounding_box = {'timeBox' : timeBox , 'spaceBox' : spaceBox}
    
    # Concatenate and sort
    this_traj = pd.concat([nan_rows, this_traj], ignore_index=True).sort_values('frame', ignore_index=True)
    return(this_traj , bounding_box)



## colorized version
def annotate_trajectory_building_tail(box, traj_df, spaceBox, upscale=8, spot_radius=8, lwd=4):
    """
    Annotate trajectory with subpixel spots and draw all prior jumps between adjacent timepoints.
    Each frame shows the trajectory up to that time point, with line and spot colors based on MLE_D.

    Color is directly mapped to MLE_D in the fixed range [0.01, 100], no normalization.

    Parameters:
    - box: 3D numpy array (T, Y, X)
    - traj_df: DataFrame with 'frame', 'x', 'y', 'ur_trajectory', 'MLE_D'
    - spaceBox: dict with 'xmin', 'xmax', 'ymin', 'ymax'
    - upscale: scale factor for subpixel rendering
    - spot_radius: radius of spot circle to draw
    - lwd: line width for trajectory segments

    Returns:
    - upsized_input: list of upscaled input images
    - annotated_frames: list of RGB arrays with annotations
    """

    annotated_frames = []
    upsized_input = []

    # Sort and group the detections
    traj_df = traj_df.sort_values(['ur_trajectory', 'frame'])
    grouped = traj_df.groupby('ur_trajectory')

    # Precompute upscaled trajectory lines
    trajectory_segments_by_frame = {t: [] for t in range(len(box))}

    # Colormap and fixed value range
    cmap = cm.viridis
    vmin, vmax = -2, 2

    # Helper: map MLE_D to colormap without normalization
    def mle_to_color(val):
        val_clipped = np.clip(np.log10(val), vmin, vmax)
        fraction = (val_clipped - vmin) / (vmax - vmin)
        return tuple(int(255*c) for c in cmap(fraction)[:3])

    # Precompute segments with colors
    for _, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            f_prev = group.loc[i-1, 'rel_frame']
            f_curr = group.loc[i, 'rel_frame']
            if f_curr == f_prev + 1:
                x1, y1 = (group.loc[i-1, 'x'] - spaceBox['xmin']) * upscale, (group.loc[i-1, 'y'] - spaceBox['ymin']) * upscale
                x2, y2 = (group.loc[i, 'x'] - spaceBox['xmin']) * upscale, (group.loc[i, 'y'] - spaceBox['ymin']) * upscale

                val = group.loc[i, 'MLE_D']
                if not np.isnan(val):
                    color = mle_to_color(val)
                    for t in range(f_curr, len(box)):
                        trajectory_segments_by_frame[t].append(((x1, y1), (x2, y2), color))

    # Render each frame
    for t in range(len(box)):
        height, width = box[t].shape  # dynamically get size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        img = Image.fromarray(frame, 'RGB')
        img_up = img.resize((img.width * upscale, img.height * upscale), resample=Image.BICUBIC)
        draw = ImageDraw.Draw(img_up)

        boximg = box[t]
        boximg = Image.fromarray(boximg)
        #boximg = boximg.convert("RGB")
        boximg_up = boximg.resize((boximg.width * upscale, boximg.height * upscale), resample=Image.BICUBIC)

        # Draw all previous displacement segments
        for x1y1, x2y2, color in trajectory_segments_by_frame[t]:
            x1, y1 = x1y1
            x2, y2 = x2y2
            if np.isnan([x1, x2, y1, y2]).any():
                continue
            draw.line([(x1, y1), (x2, y2)], fill=color, width=lwd)

        # Draw current spots colored by MLE_D
        current_detections = traj_df[traj_df['rel_frame'] == t]
        for _, row in current_detections.iterrows():
            x = (row['x'] - spaceBox['xmin']) * upscale
            y = (row['y'] - spaceBox['ymin']) * upscale
            bbox = [x - spot_radius, y - spot_radius, x + spot_radius, y + spot_radius]

            if not np.isnan(row['MLE_D']):
                ellipse_color = mle_to_color(row['MLE_D'])
                draw.ellipse(bbox, outline=ellipse_color, width=lwd)

        annotated_frames.append(np.array(img_up))
        upsized_input.append(np.array(boximg_up))

    return upsized_input, annotated_frames




def save_overlay_movie(box_resized, box_annotated, output_path='overlay_output.mp4', fps=20):
    frames_rgb = []

    ## ensure the image is 8-bit and invert BW
    box_resized = box_resized - np.min(box_resized)
    box_resized = (box_resized / np.max(box_resized) * 255).astype(np.uint8)
    box_resized = [255 - frame for frame in box_resized]
    
    #box_resized = box_resized - np.min(box_resized)
    #box_resized = (box_resized / np.max(box_resized) * 255).astype(np.uint8)
    
    #print(np.percentile(box_resized , 2))
    #print(np.percentile(box_resized , 99))
    box_resized = box_resized - np.percentile(box_resized , 2)
    box_resized = (box_resized / np.percentile(box_resized , 99) * 255)
    box_resized = np.clip(box_resized , 0 , 255).astype(np.uint8)
    #print(np.min(box_resized))
    #print(np.max(box_resized))
    
    for gray, blue in zip(box_resized, box_annotated):
        # Ensure frames are 2D arrays
        gray = np.squeeze(gray)
        blue = np.squeeze(blue)

        ## scale intensity on a per-frame basis
        #gray = gray - np.min(gray)
        #gray = (gray / np.max(gray) * 255).astype(np.uint8)
        
        # Create RGB channels
        rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

        # Grayscale in all channels
        rgb[..., 0] = gray  # Red
        rgb[..., 1] = gray  # Green
        rgb[..., 2] = gray  # Blue

        # Overlay blue mask where blue > 0
        rgb[..., 0][blue.sum(axis = 2) > 0] = blue[...,0][blue.sum(axis = 2) > 0]     # Remove red
        rgb[..., 1][blue.sum(axis = 2) > 0] = blue[...,1][blue.sum(axis = 2) > 0]      # Remove green
        rgb[..., 2][blue.sum(axis = 2) > 0] = blue[...,2][blue.sum(axis = 2) > 0]    # Add blue at annotation locations

        frames_rgb.append(rgb)

    # Save as MP4
    imageio.mimsave(output_path, frames_rgb, fps=fps, codec='libx264', quality=8)
    print(f"Movie saved to {output_path}")


    

def save_overlay_movie_monochrome(box_resized, box_annotated, output_path='overlay_output.mp4', fps=20):
    frames_rgb = []

    box_resized = [255 - frame for frame in box_resized]
    box_resized = box_resized - np.min(box_resized)
    box_resized = (box_resized / np.max(box_resized) * 255).astype(np.uint8)
    
    for gray, blue in zip(box_resized, box_annotated):
        # Ensure frames are 2D arrays
        gray = np.squeeze(gray)
        blue = np.squeeze(blue)

        # Create RGB channels
        rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

        # Grayscale in all channels
        #rgb[..., 0] = gray  # Red
        #rgb[..., 1] = gray  # Green
        #rgb[..., 2] = gray  # Blue
        rgb = gray

        # Overlay blue mask where blue > 0
        rgb[..., 0][blue.sum(axis = 2) > 0] = 48     # Remove red
        rgb[..., 1][blue.sum(axis = 2) > 0] = 92      # Remove green
        rgb[..., 2][blue.sum(axis = 2) > 0] = 222    # Add blue at annotation locations

        frames_rgb.append(rgb)

    # Save as MP4
    imageio.mimsave(output_path, frames_rgb, fps=fps, codec='libx264', quality=8)
    print(f"Movie saved to {output_path}")

    
    
def save_overlay_tiff(box, box_anotations, output_path='overlay_output.tif', fps=20):
    
    box = box - np.min(box)
    box = (box / np.max(box) * 255).astype(np.uint8)
    
    box_gray = np.dot(box[...,:3], [0.2989, 0.5870, 0.1140]) ## standard weights for de-mixing RGB
    annotations_binary = [
        np.where(np.any(frame != 0, axis=2), 255, 0).astype(np.uint8)
        for frame in box_anotations
    ]
    
    combined_channels = np.stack([np.stack(box_gray, axis=0), np.stack(annotations_binary, axis=0)], axis=-1)

    imwrite(
        output_movie,
        combined_channels,
        photometric='minisblack',  # grayscale
        planarconfig='contig'      # channels stored together per pixel
    )
    print(f"Movie saved to {output_path}")


def compute_runs_df(traj_df, tol=0.1, column='MLE_D'):
    """
    Compute consecutive runs of roughly constant D_MLE in a trajectory DataFrame.
    
    Parameters:
    - traj_df: DataFrame for a single trajectory
    - tol: max difference to consider values in the same state
    - column: the column containing MLE values
    
    Returns:
    - List of (state_value, run_length)
    """
    traj = traj_df[column].values
    runs = []
    if len(traj) == 0:
        return runs

    current_val = traj[0]
    run_len = 1
    for val in traj[1:]:
        if abs(val - current_val) <= tol:
            run_len += 1
        else:
            runs.append((current_val, run_len))
            current_val = val
            run_len = 1
    runs.append((current_val, run_len))
    return runs

def trajectory_score_df(traj_df, run_tol=0.1, min_run_len=5, column='MLE_D'):
    """
    Compute simple metrics for filtering visually interesting trajectories.
    
    Returns:
    - length: number of frames
    - num_long_runs: number of runs longer than min_run_len
    - state_var: standard deviation of D_MLE
    """
    runs = compute_runs_df(traj_df, tol=run_tol, column=column)
    long_runs = [r for r in runs if r[1] >= min_run_len]
    num_long_runs = len(long_runs)
    state_var = traj_df[column].std()
    length = len(traj_df)
    return length, num_long_runs, state_var



import numpy as np
import pandas as pd

def bleaching_curve(img, traj_df, signal_size=3, background_size=7, post_frames=10):
    """
    Measure signal and background around detected positions in an image stack.

    Parameters:
    - img: np.ndarray of shape (T, Y, X)
    - traj_df: DataFrame with columns ['x', 'y', 'frame']
    - signal_size: width of signal square (pixels)
    - background_size: width of background square (pixels)
    - post_frames: number of frames to continue measuring after trajectory ends

    Returns:
    - traj_df with new columns ['signal', 'background', 'rel_frame']
    """

    half_signal = signal_size // 2
    half_background = background_size // 2
    results = []

    # Drop NaN positions
    traj_valid = traj_df.dropna(subset=['x', 'y', 'frame']).copy()

    # Get last frame and position for post-measurement
    if len(traj_valid) > 0:
        last_x = traj_valid['x'].iloc[-1]
        last_y = traj_valid['y'].iloc[-1]
        last_frame = int(traj_valid['frame'].iloc[-1])

        # Add extra frames at the same position
        post_rows = []
        for f in range(last_frame + 1, min(last_frame + 1 + post_frames, img.shape[0])):
            post_rows.append({
                'x': last_x, 'y': last_y, 'frame': f,
                'ur_trajectory': traj_valid['ur_trajectory'].iloc[-1],
                'condition': traj_valid['condition'].iloc[-1]
            })
        if post_rows:
            traj_valid = pd.concat([traj_valid, pd.DataFrame(post_rows)], ignore_index=True)

    # Measure intensity per frame
    for _, row in traj_valid.iterrows():
        x = int(round(row['x']))
        y = int(round(row['y']))
        f = int(row['frame'])

        if np.isnan(x) or np.isnan(y) or f >= img.shape[0]:
            signal = np.nan
            background = np.nan
        else:
            frame_img = img[f]
            h, w = frame_img.shape

            # Define region bounds safely (clipped to image)
            y1s, y2s = np.clip([y - half_signal, y + half_signal + 1], 0, h)
            x1s, x2s = np.clip([x - half_signal, x + half_signal + 1], 0, w)
            y1b, y2b = np.clip([y - half_background, y + half_background + 1], 0, h)
            x1b, x2b = np.clip([x - half_background, x + half_background + 1], 0, w)

            # Extract patches
            signal_patch = frame_img[y1s:y2s, x1s:x2s]
            background_patch = frame_img[y1b:y2b, x1b:x2b]

            signal = np.mean(signal_patch)
            background = np.median(background_patch)

        result = row.to_dict()
        result['signal'] = signal
        result['background'] = background
        results.append(result)

    df_out = pd.DataFrame(results)

    # Compute rel_frame relative to trajectory end (0 = last frame)
    if not df_out.empty:
        non_nan = df_out[df_out['I0'].notna()]
        last_frame = non_nan['frame'].max()
        df_out['rel_frame'] = df_out['frame'] - last_frame

    return df_out


def normalize_signal_by_baseline(df, pre_window=10):
    """
    For each trajectory, divide 'signal' by the mean signal in the pre_window frames
    before rel_frame = 0 (i.e., rel_frame < 0).
    """
    df = df.copy()

    def normalize(traj):
        # baseline: frames where rel_frame < 0 and >= -pre_window
        mask = (traj['rel_frame'] < 0) & (traj['rel_frame'] >= -pre_window)
        baseline = traj.loc[mask, 'corrected_signal'].mean()
        if pd.isna(baseline) or baseline == 0:
            # avoid division by zero
            traj['signal_norm'] = traj['corrected_signal']
        else:
            traj['signal_norm'] = traj['corrected_signal'] / baseline
        return traj

    df = df.groupby('ur_trajectory', group_keys=False).apply(normalize)
    return df


def aggregate_bleaching_data(rollingMLE , settings , n_particles = 200 , min_obs = 7):

    counts = rollingMLE['ur_trajectory'].value_counts()
    valid_trajectories = counts[counts >= min_obs].index
    rollingMLE_filtered = rollingMLE[rollingMLE['ur_trajectory'].isin(valid_trajectories)].copy()
    bc = pd.DataFrame()
    prev_open_image = ''

    unique_trajs = np.unique(rollingMLE_filtered["ur_trajectory"])
    n = min(n_particles, len(unique_trajs))
    #n = n_particles
    sampled_ids = np.random.choice(unique_trajs, size=n, replace=False)
    sampled_ids = sorted(sampled_ids)
    sampled_trajs = rollingMLE_filtered[rollingMLE_filtered['ur_trajectory'].isin(sampled_ids)]
    sampled_trajs = sampled_trajs.sort_values(by=['ur_trajectory', 'frame'], ignore_index=True)

    # iterate over each sampled trajectory and run your code
    for traj_id in sampled_ids:
        this_traj , boundingBox = isolate_traj(rollingMLE_filtered , traj_id)

        try:
            file_pattern = this_traj['ur_trajectory'].replace(r'_traj::\d+', '.nd2', regex=True)[0]
            image_file = glob.glob(os.path.join(settings['io']['data_directory'] , file_pattern))[0]

            ## only reload the image if neccessary
            if image_file == prev_open_image:
                pass
                #print('skipping open new image')
            else:
                #print(f'Old image:: {prev_open_image} ... new image:: {image_file}')
                img = pims.open(image_file)
                img_stack = np.array(img)
                prev_open_image = image_file                
            bc = pd.concat([bc , bleaching_curve(img , this_traj)])

        except:
           print('Caught an error')
           pass
    return bc


## can handle many conditions, but as currently implemented is only called on one condition at a time
def plot_bleaching_curves(settings , bc ):
    
    plot_xlim = settings['plot']['bleach_xlim'] * 1000
    plot_xlim = (settings['saspt']['frame_interval'] * -30 * 1000) , (settings['saspt']['frame_interval'] * 10 * 1000)
    
    output_dirname = os.path.join(settings['io']['plot_directory'] , 'bleaching_curves')
    os.makedirs(output_dirname, exist_ok=True)

    bc['corrected_signal'] = bc['signal'] - bc['background']
    # Apply to bc
    bc_norm = normalize_signal_by_baseline(bc, pre_window=10)
    bc_norm['signal_smooth'] = bc_norm.groupby('ur_trajectory')['signal_norm'] \
                                      .transform(lambda x: x.rolling(3, min_periods=1).mean())
    median_fluorescence = (
        bc_norm
        .groupby(['condition', 'rel_frame'], as_index=False)
        .agg(median_signal=('signal_norm', 'median'))
    )

    for cond in bc_norm['condition'].unique():
        output_filename = os.path.join(output_dirname, f"{cond}.pdf")
        print(f"Writing to {output_filename}")

        # Filter data for this condition
        df_cond = bc_norm[bc_norm['condition'] == cond]
        df_median = median_fluorescence[median_fluorescence['condition'] == cond]
        
        df_cond['rel_frame'] = df_cond['rel_frame'] * settings['saspt']['frame_interval'] * 1000
        df_median['rel_frame'] = df_median['rel_frame'] * settings['saspt']['frame_interval'] * 1000

        # Create figure
        plt.figure(figsize=(8, 6))

        # Plot each trajectory separately
        for traj_id, traj_data in df_cond.groupby('ur_trajectory'):
            plt.plot(
                traj_data['rel_frame'],
                traj_data['signal_smooth'] ,
                color = 'darkgray' , 
                alpha = 0.25
            )
        plt.plot(
            df_median['rel_frame'],
            df_median['median_signal'] ,
            color = 'black'
        )   

        # Axis limits
        plt.xlim(plot_xlim[0], plot_xlim[1])
        plt.ylim(-0.5, 3.5)

        # Labels and title
        plt.xlabel("Time rel bleaching (ms)")
        plt.ylabel("Signal")
        plt.title(f"Condition: {cond}")

        # Save figure
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()




## take a rollingMLE object and display movies of the tracked particles
def generate_movies(rollingMLE , settings , min_obs = 8 , overlay = True):
    counts = rollingMLE['ur_trajectory'].value_counts()
    valid_trajectories = counts[counts >= min_obs].index

    rollingMLE_filtered = rollingMLE[rollingMLE['ur_trajectory'].isin(valid_trajectories)].copy()
    condition = rollingMLE_filtered['ur_trajectory'].str.replace(r'_\d+_traj::\d+', '', regex=True).unique()[0]
    
    unique_trajs = np.unique(rollingMLE_filtered["ur_trajectory"])
    n = min(100, len(unique_trajs))

    sampled_trajs = np.random.choice(unique_trajs, size=n, replace=False)

    output_dirname = os.path.join(settings['io']['movie_directory'] , condition)
    os.makedirs(output_dirname, exist_ok=True)
    prev_open_image = ''

    # iterate over each sampled trajectory and run your code
    for traj_id in sampled_trajs:
        this_traj , boundingBox = isolate_traj(rollingMLE_filtered , traj_id)
        output_filename = f"{this_traj['ur_trajectory'][0]}.mp4"
        output_filename = os.path.join(output_dirname , output_filename)

        #print(f"Writing to     {output_filename}")

        try:
            file_pattern = this_traj['ur_trajectory'].replace(r'_traj::\d+', '.nd2', regex=True)[0]
            image_file = glob.glob(os.path.join(settings['io']['data_directory'] , file_pattern))[0]

             ## only reload the image if neccessary
            if image_file == prev_open_image:
                pass
                #print('skipping open new image')
            else:
                #print(f'Old image:: {prev_open_image} ... new image:: {image_file}')
                img = pims.open(image_file)
                #half_size = 15  # since 15x15 has half-width 7
                img_stack = np.array(img)
                prev_open_image = image_file                


            ## isolte bounding box around the particle
            box = img_stack[boundingBox['timeBox'][0] : boundingBox['timeBox'][1], boundingBox['spaceBox']['ymin'] : boundingBox['spaceBox']['ymax'] ,
                                 boundingBox['spaceBox']['xmin'] : boundingBox['spaceBox']['xmax']]

            ## generate a render-able image (resized as needed) and a printable annotation layer matching the resolution of the image
            box_resized , box_resized_anotations = annotate_trajectory_building_tail(box , this_traj , boundingBox['spaceBox'] , upscale=8, spot_radius = 12 , lwd = 3)

            if overlay == True:
                save_overlay_movie(box_resized , box_resized_anotations , output_filename)
            else:
                box_resized_annotations = np.zeros(box_resized_annotations)
                save_overlay_movie(box_resized , box_resized_anotations , output_filename)

                #output_movie = os.path.join(settings['io']['movie_directory'] , 'movie.tif')
                #save_overlay_tiff(box_resized , box_resized_anotations , output_movie) ## this is 24mb... save these sparingly
        except:
            print('error found')
            pass