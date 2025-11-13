import yaml
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import re
import toml
import time
import shutil
import glob

#import imagej
#import jpype
#import scyjava

import tifffile
import nd2reader
import pims
import subprocess

from quot.read import read_config
from quot.core import track_files
from saspt import StateArrayParameters, StateArray, make_likelihood, RBME

#from hmmlearn.hmm import GaussianHMM




## take settings as input, and return a list of input images that need to be processed
## an image needs to be processed if it does not have corresponding output files (_traj.csv , _posterio.csv , _mle.csv , _rollingMLE.csv)
## in the proper directories
## if the client specifices force=True , return all input images in the settings-specified tree
## output will always be grouped by condition (key = condition ; value = list of files)
def identify_missing_filewise(settings , force = False):

    nd2_files_by_condition = group_files_by_metadata(os.path.join(settings['io']['home_directory'] , settings['io']['data_filepath']))

    if force == True:
        return nd2_files_by_condition
    
    ### Identify missing file-wise outputs
    target_files = {}
    for condition, file_list in nd2_files_by_condition.items():
        target_list = []
        for f in file_list:
            base = os.path.splitext(f)[0]  # remove .nd2 extension

            target = settings['io']['traj_directory'] + '/' + base + "_traj.csv"
            target_list.append(target)
            target = settings['io']['post_directory'] + '/' + base + "_posterior.csv"
            target_list.append(target)
            target = settings['io']['MLE_directory'] + '/' + base + "_MLE.csv"
            target_list.append(target)
            target = settings['io']['rolling_window_directory'] + '/' + base + "_rollingMLE.csv"
            target_list.append(target)
        target_files[condition] = target_list


    ### Check which files are missing
    missing_files = {}
    for key, file_list in target_files.items():
        missing = [f for f in file_list if not os.path.exists(f)]
        if missing:
            missing_files[key] = missing

    if missing_files:
        print("\nMissing files.")
        #for key, files in missing_files.items():
        #    for f in files:
        #        print(f"  {key}: {f}")
    else:
        print("\nAll files exist.\n")


    ### Convert missing CSVs to corresponding .nd2 filenames (preserving condition structure)
    nd2_files_by_condition = {}

    for key, files in missing_files.items():
        nd2_set = set()
        for f in files:
            base = os.path.basename(f)
            base = re.sub(r'(_\d+)(?:_.*)?\.csv$', r'\1.nd2', base)  # keep _### and replace suffix with .nd2
            nd2_set.add(base)
        nd2_files_by_condition[key] = sorted(nd2_set)

    ### Display results
    print("\nRun file-wise operations for .nd2 files (grouped by condition):")
    for key, files in nd2_files_by_condition.items():
        print(f"\n{key}:")
        for f in files:
            print(f"\t{f}")
    
    return nd2_files_by_condition













def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin(i) for i in obj]
    elif isinstance(obj, (np.float64, np.float32, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    else:
        return obj

def group_files_by_metadata(directory, extension=".nd2"):
    files_by_metadata = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not filename.endswith(extension):
            continue

        # Remove trailing index like _001.nd2
        match = re.match(r"(.*)_\d{3}" + re.escape(extension) + r"$", filename)
        if match:
            metadata_key = match.group(1)
            files_by_metadata[metadata_key].append(filename)

    return dict(files_by_metadata)



def group_quotOutput_by_metadata(directory, extension="_trajs.csv"):
    files_by_metadata = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not filename.endswith(extension):
            continue

        # Remove trailing index like _001.nd2
        match = re.match(r"(.*)_\d{3}" + re.escape(extension) + r"$", filename)
        if match:
            metadata_key = match.group(1)
            files_by_metadata[metadata_key].append(filename)

    return dict(files_by_metadata)




## this module reads in YAML settings
def safe_eval(expr):
    """Evaluate limited expressions using only NumPy."""
    allowed_names = {"np": np}
    return eval(expr, {"__builtins__": {}}, allowed_names)

def recursive_parse(value):
    """Recursively evaluate NumPy expressions in nested structures."""
    if isinstance(value, str) and value.startswith("np."):
        try:
            return safe_eval(value)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {value}") from e
    elif isinstance(value, dict):
        return {k: recursive_parse(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursive_parse(v) for v in value]
    else:
        return value

    
def save_dict_to_yaml(data, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
'''
        ## core tracking code -- requires an imageJ instance called ij as an argument
        ## but handles loading of java classes in function scope
        ## could likely pass **kwargs to avoid re-loading these classes
def run_fiji(ij , rows , global_settings , filename):
    
    
    # Import Java classes used by TrackMate
    TrackMate = jpype.JClass('fiji.plugin.trackmate.TrackMate')
    Model = jpype.JClass('fiji.plugin.trackmate.Model')
    Settings = jpype.JClass('fiji.plugin.trackmate.Settings')
    SelectionModel = jpype.JClass('fiji.plugin.trackmate.SelectionModel')
    Logger = jpype.JClass('java.util.logging.Logger')
    LogLevel = jpype.JClass('java.util.logging.Level')
    Integer = jpype.JClass('java.lang.Integer')
    Double = jpype.JClass('java.lang.Double')
    FeatureFilter = jpype.JClass('fiji.plugin.trackmate.features.FeatureFilter')
    LogDetectorFactory = jpype.JClass('fiji.plugin.trackmate.detection.LogDetectorFactory')
    SimpleLAPTrackerFactory = jpype.JClass('fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory')

    ImageReader = jpype.JClass('loci.formats.ImageReader')
    ServiceFactory = jpype.JClass('loci.common.services.ServiceFactory')
    OMEXMLServiceImpl = jpype.JClass('loci.formats.services.OMEXMLServiceImpl')
    
    LoggerFactory = jpype.JClass("org.slf4j.LoggerFactory")
    Logger = LoggerFactory.getLogger("loci.formats")
    
    LogbackLevel = jpype.JClass("ch.qos.logback.classic.Level")
    Logger.setLevel(LogbackLevel.ERROR)

    image_path = os.path.join(global_settings['io']['data_filepath'] , filename)
    imp = ij.io().open(image_path)
    img_plus = ij.py.to_imageplus(imp)

    # Create the settings object
    settings = Settings(img_plus) ## inherit imp from img_plus
    model = Model() ## create the tracker model

    # Configure detector settings (LoG example)
    
    ####
    ## get metadata without accessing another function
    ####
    service = OMEXMLServiceImpl()
    metadata = service.createOMEXMLMetadata()
    
    reader = ImageReader()
    reader.setMetadataStore(metadata)
    reader.setId(image_path)  # your .nd2 file

        # Now read metadata from series 0
    size_x = metadata.getPixelsSizeX(0).getValue()
    size_y = metadata.getPixelsSizeY(0).getValue()
    size_z = metadata.getPixelsSizeZ(0).getValue()
    size_c = metadata.getPixelsSizeC(0).getValue()
    size_t = metadata.getPixelsSizeT(0).getValue()

    px = metadata.getPixelsPhysicalSizeX(0)
    py = metadata.getPixelsPhysicalSizeY(0)

    # Read the delta time between plane 0 and plane 1
    acq_time_0 = metadata.getPlaneDeltaT(0, 0)  # (series, plane index)
    acq_time_1 = metadata.getPlaneDeltaT(0, 1)

    dt = acq_time_1.value() - acq_time_0.value()
    t_unit = acq_time_1.unit().getSymbol()
        
    meta = {'size_x' : size_x , 'size_y' : size_y , 'n_channels' : size_c , 
            'size_t' : size_t , 'px_size' : py.value() , 'px_size_units' : py.unit().getSymbol() , 
            't_interval' : dt , 't_unit' : t_unit}
        
    print(meta)
    reader.close()
    ##############
    
    # Assume img_plus is your ImagePlus instance
    w = img_plus.getWidth()
    h = img_plus.getHeight()
    
    ## pre-load parameters for spot filter
    width_lower_bound = 0.4
    width_upper_bound = w * settings.dx - 0.4
    height_lower_bound = 0.4
    height_upper_bound = w * settings.dy - 0.4

    # Set detector factory
    settings.detectorFactory = LogDetectorFactory()
    settings.trackerFactory = SimpleLAPTrackerFactory()

    detector_settings = settings.detectorSettings
    detector_settings['RADIUS'] = global_settings['trackmate']['radius']  # radius of spots in pixels
    detector_settings['TARGET_CHANNEL'] = Integer(global_settings['trackmate']['channel'])  # channel to use (1-based)
    detector_settings['THRESHOLD'] = global_settings['trackmate']['quality']
    detector_settings['DO_MEDIAN_FILTERING'] = False
    detector_settings['DO_SUBPIXEL_LOCALIZATION'] = True

    # Configure tracker settings (simple LAP tracker)
    tracker_settings = settings.trackerSettings
    tracker_settings['LINKING_MAX_DISTANCE'] = global_settings['trackmate']['linking_distance']
    tracker_settings['GAP_CLOSING_MAX_DISTANCE'] = 0.0
    tracker_settings['MAX_FRAME_GAP'] = Integer(0)
    tracker_settings['ALLOW_GAP_CLOSING'] = False
    tracker_settings['ALLOW_TRACK_SPLITTING'] = False
    tracker_settings['SPLITTING_MAX_DISTANCE'] = 0.0
    tracker_settings['ALLOW_TRACK_MERGING'] = False
    tracker_settings['MERGING_MAX_DISTANCE'] = 0.0
    tracker_settings['ALTERNATIVE_LINKING_COST_FACTOR'] = 1.0
    tracker_settings['CUTOFF_PERCENTILE'] = 0.0
    tracker_settings['BLOCKING_VALUE'] = Double.POSITIVE_INFINITY

    # Configure spot filters - filter on x position to remove weird underflow at edges
    filter1 = FeatureFilter('POSITION_X', width_lower_bound , True) ## cutting off left size because of high density in 0pp condition
    settings.addSpotFilter(filter1)
    filter3 = FeatureFilter('POSITION_X', width_upper_bound , False)
    settings.addSpotFilter(filter3)
    filter4 = FeatureFilter('POSITION_Y', height_lower_bound , True)
    settings.addSpotFilter(filter4)
    filter5 = FeatureFilter('POSITION_Y', height_upper_bound , False) ## 31.9 for 292px
    settings.addSpotFilter(filter5)

    # Create TrackMate instance
    trackmate = TrackMate(model, settings)

    # Run detection
    if not trackmate.checkInput():
        print('Error: input check failed.')
        trackmate.getErrorMessage()
        exit(1)

    if not trackmate.process():
        print('Error: processing failed.')
        trackmate.getErrorMessage()
        exit(1)

    spots = model.getSpots()
    track_model = model.getTrackModel()

    iterator = spots.iterator(False)  # <-- Iterate over all spots across all frames
    while iterator.hasNext():
        spot = iterator.next()
        x = spot.getDoublePosition(0)  # X coordinate
        y = spot.getDoublePosition(1)  # Y coordinate
        frame = int(spot.getFeature('FRAME'))  # Frame number
        track_id = track_model.trackIDOf(spot)  # Associated Track ID
        if track_id is not None:
            track_id = f"{filename}::{track_id}"
        rows.append((x, y, frame, track_id))

#     # extract metadata to save in next step (before closing img_plus)
#     calibration = img_plus.getCalibration()
#     pixel_width = calibration.pixelWidth
#     n_frames = img_plus.getImageStackSize()
#     img_width = img_plus.getWidth()
#     img_height = img_plus.getHeight()

#     meta = {'n_frames' : n_frames ,
#                 'img_width' : img_width , 
#                 'img_height' : img_height , 
#                'pixel_width' : pixel_width , 
#                'time_increment' : time_increment}
    
    img_plus.close()
    jpype.java.lang.System.gc() 
    
    return(meta) ## and update rows pointer in-place


## get the nd2 embeded metadata for the file at image_path
## relies on classes loaded by the caller and passed in ij_args
def get_nd2_metadata(ij , image_path ,  OMEXMLServiceImpl , ImageReader):
    # Import Java classes used by TrackMate

    # Path to file (already in variable 'image_path')
    # Set up OME metadata service and reader
    service = OMEXMLServiceImpl()
    metadata = service.createOMEXMLMetadata()

    reader = ImageReader()
    reader.setMetadataStore(metadata)
    reader.setId(image_path)  # your .nd2 file

        # Now read metadata from series 0
    size_x = metadata.getPixelsSizeX(0).getValue()
    size_y = metadata.getPixelsSizeY(0).getValue()
    size_z = metadata.getPixelsSizeZ(0).getValue()
    size_c = metadata.getPixelsSizeC(0).getValue()
    size_t = metadata.getPixelsSizeT(0).getValue()

    px = metadata.getPixelsPhysicalSizeX(0)
    py = metadata.getPixelsPhysicalSizeY(0)

    # Read the delta time between plane 0 and plane 1
    acq_time_0 = metadata.getPlaneDeltaT(0, 0)  # (series, plane index)
    acq_time_1 = metadata.getPlaneDeltaT(0, 1)

    dt = acq_time_1.value() - acq_time_0.value()
    t_unit = acq_time_1.unit().getSymbol()
        
    meta = {'size_x' : size_x , 'size_y' : size_y , 'n_channels' : size_c , 
            'size_t' : size_t , 'px_size' : py.value() , 'px_size_units' : py.unit().getSymbol() , 
            't_interval' : dt , 't_unit' : t_unit}
    return(meta)


def initialize_imagej():
    ij = imagej.init('~/software/Fiji.app' , mode='headless')

    OMEXMLServiceImpl = jpype.JClass('loci.formats.services.OMEXMLServiceImpl')
    ImageReader = jpype.JClass('loci.formats.ImageReader')

    return(ij , OMEXMLServiceImpl , ImageReader)

'''

from pathlib import Path


def aggregate_csv(directory_string , filter_string = "*.csv"):
    
    # Path to the directory with your CSV files
    csv_dir = Path(directory_string)

    # List of all .csv files in the directory
    csv_files = sorted(csv_dir.glob(filter_string))

    # Load and concatenate all CSVs
    df_list = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(df_list, ignore_index=True)
    return(master_df)




'''
## set up the various fields of the metadata objects
## this has become a bit messy and could be re-written to make fewer objects
def prep_analysis(analysis_file = 'analyis_settings.yaml'):

    # Load and parse YAML
    with open("analysis_settings.yaml", "r") as f:
        raw_config = yaml.safe_load(f)

    trackmate_settings = recursive_parse(raw_config.get("trackmate_settings", {}))
    saspt_settings = recursive_parse(raw_config.get("saspt_settings", {}))
    home_directory = recursive_parse(raw_config.get("home_directory" , {}))
    data_filepath = recursive_parse(raw_config.get("data_filepath", {}))
    quot_settings_path = recursive_parse(raw_config.get("quot_settings_path", {}))
    analysis_relative_filepath = recursive_parse(raw_config.get("analysis_relative_filepath", {}))

    trackmate_settings['srcDir'] = os.path.join(home_directory , data_filepath)

    # Format directory name
    dir_name = f"trackmate_output_q={str(trackmate_settings['quality']).replace('.','p')}"
    output_path = os.path.join(home_directory , dir_name)
    trackmate_settings['dstDir'] = output_path
    trackmate_settings['MLE_directory'] = os.path.join(output_path , "MLE_directory")
    trackmate_settings['posterior_directory'] = os.path.join(output_path , "posterior_directory")
    trackmate_settings['home_directory'] = home_directory

    # Create the directory if it doesn't exist
    os.makedirs(trackmate_settings['dstDir'], exist_ok=True)
    os.makedirs(trackmate_settings['MLE_directory'], exist_ok=True)
    os.makedirs(trackmate_settings['posterior_directory'], exist_ok=True)

    grouped_files = group_files_by_metadata(os.path.join(trackmate_settings['srcDir']))

    # Print the result
    for key, file_list in grouped_files.items():
        print(f"{key}:")
        print(f"  - {len(file_list)} files")

    condition_file = os.path.join(trackmate_settings['srcDir'] , "fileIndex.yaml")

    save_dict_to_yaml(grouped_files, condition_file)
    print("")
    print(f"Saved to {condition_file}")
    
    ## use the median pixel value to set the background for the spot localization
    key, val_list = next(iter(grouped_files.items()))
    img = pims.open(os.path.join(trackmate_settings['srcDir'] ,  val_list[0]))

    frame = img[0]
    bg_level = np.median(frame)
    print(f"Imaging background calculated to be : {bg_level}")

    quot_settings = read_config(quot_settings_path)
    quot_settings['localize']['camera_bg'] = float(bg_level)
    quot_settings['detect']['t'] = trackmate_settings['quality']
    saspt_settings['frame_interval']  = 1 / img.frame_rate
    quot_settings['track']['frame_interval'] = saspt_settings['frame_interval']
    trackmate_settings['background'] = bg_level
    quot_settings['track']['pixel_size_um'] = img.calibration
    saspt_settings['pixel_size_um'] = quot_settings['track']['pixel_size_um']

    with open("./current_quot_settings.toml", "w") as f: ## helper for SLURM job -- don't overwrite the orignal settings
        toml.dump(to_builtin(quot_settings), f)

    with open("./current_saspt_settings.toml", "w") as f: ## helper for SLURM job -- don't overwrite the orignal settings
        toml.dump(to_builtin(saspt_settings), f)


    yaml_settings_file = os.path.join(trackmate_settings['dstDir'] , 'called_settings.toml')
    settings_to_dump = {'trackmate' : trackmate_settings , 
                        'saspt' : saspt_settings , 
                        'quot' : quot_settings}
    with open(yaml_settings_file, "w") as f: ## helper for SLURM job -- don't overwrite the orignal settings
        toml.dump(to_builtin(settings_to_dump), f)
    
    return(trackmate_settings , saspt_settings , quot_settings, grouped_files)
'''


def is_job_running(job_id):
    result = subprocess.run(["squeue", "--job", str(job_id)],
                            capture_output=True, text=True)
    return str(job_id) in result.stdout

def count_jobs_running(job_ids):
    job_ids = set(str(jid) for jid in job_ids)  # ensure string type
    result = subprocess.run(["squeue", "--noheader", "--format=%i"],
                            capture_output=True, text=True)
    running_ids = set(result.stdout.strip().splitlines())
    still_running = job_ids & running_ids
    return len(still_running)


def get_running_job_ids():
    result = subprocess.run(["squeue", "--noheader", "--format=%i"],
                            capture_output=True, text=True)
    return set(result.stdout.strip().splitlines())

def keys_with_all_jobs_finished(job_dict):
    running = get_running_job_ids()
    finished_keys = []
    for key, job_ids in job_dict.items():
        job_ids = set(str(j) for j in job_ids)
        if job_ids.isdisjoint(running):  # no overlap = all finished
            finished_keys.append(key)
    return finished_keys


def print_nested_dict(d, indent=0, file=None):
    should_close = False

    if isinstance(file, str):
        file = open(file, "w")
        should_close = True

    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key} ::", file=file)
            print_nested_dict(value, indent + 2, file=file)
        else:
            print(" " * indent + f"{key} :: {value}", file=file)

    if should_close:
        file.close()
        
## update diciontary recursively, i.e. update values based on settings_override.yaml
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
            
            
def extract_metadata(file_path, keys):
    """Extract key=value pairs from a filename."""
    metadata = {}
    for key in keys:
        match = re.search(rf"{key}=([^_]+)", file_path)
        if match:
            metadata[key] = match.group(1)
    return metadata


import numpy as np
from scipy.stats import norm


## this is "wilson's sore interval" based on binomial distribution
## assumes n_alive are randomly drawn from n_total, which seems like a weird assumption
def wilson_ci(k, n, confidence=0.95):
    """k = successes, n = total trials"""
    if n == 0:
        return (0.0, 0.0)

    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denominator = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return lower, upper


def split_image_stack(img, tile_size=110, overlap=10):
    """
    Split a 3D image (T, Y, X) into overlapping tiles of shape ≤ (tile_size, tile_size, T).
    Includes trailing edge tiles and ensures no gaps between tiles.
    """
    T, Y, X = img.shape
    step = tile_size - overlap
    tiles = []
    positions = []

    # Compute x start positions
    x_starts = list(range(0, X - tile_size + 1, step))
    if x_starts and x_starts[-1] + tile_size < X:
        x_starts.append(x_starts[-1] + step)
    elif not x_starts:
        x_starts = [0]

    # Compute y start positions
    y_starts = list(range(0, Y - tile_size + 1, step))
    if y_starts and y_starts[-1] + tile_size < Y:
        y_starts.append(y_starts[-1] + step)
    elif not y_starts:
        y_starts = [0]

    for y_start in y_starts:
        for x_start in x_starts:
            x_end = x_start + tile_size
            y_end = y_start + tile_size

            # Slice and transpose to get shape (tile_height, tile_width, T)
            tile = img[:, y_start:y_end, x_start:x_end]
            tiles.append(tile)
            positions.append((x_start, y_start))

    return tiles, positions



def save_tiles(tiles, positions, out_dir, nd2_filename):
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(nd2_filename))[0]
    saved_basenames = []

    for tile, (x, y) in zip(tiles, positions):
        fname = f"{base_name}_x={x}_y={y}.tif"
        fpath = os.path.join(out_dir, fname)
        tifffile.imwrite(fpath, tile.astype(np.uint16))  # adjust dtype as needed
        saved_basenames.append(fname)

    return saved_basenames



def extract_offsets(filename):
    """Extract x and y offset values from the filename."""
    match = re.search(r'_x=(\d+)_y=(\d+)', filename)
    if match:
        x_offset = int(match.group(1))
        y_offset = int(match.group(2))
        return x_offset, y_offset
    else:
        raise ValueError(f"Could not extract x/y offsets from filename: {filename}")
        
def update_csv_with_offsets(filename):
    # Step 1: extract offsets
    x_offset, y_offset = extract_offsets(filename)

    # Step 2: load the file
    df = pd.read_csv(filename)

    # Step 3: add offsets to relevant columns
    for col in ['x', 'x_detect']:
        if col in df.columns:
            df[col] += x_offset
    for col in ['y', 'y_detect']:
        if col in df.columns:
            df[col] += y_offset

    return df  # or optionally: df.to_csv(...)
    
    
### takes a CSV of trajectories, splits into sub-trajectories
### and runs SASPT on each sub-trajectory, and then re-aggregates the data
### and saves a detections table, where each detection is described by
### an instantaneous diffusion rate (except at trajectory ends)
def analyze_rolling_windows(input_csv, settings):
    window_size = settings['saspt']['splitsize'] ## use the same window size for SASPT and for HMM
    if window_size % 2 == 1: ## window size is odd / make symmetric window
        half_window = window_size // 2
        left_window = half_window
        right_window = half_window
    else: ## window size is even // make left-shifted window
        left_window = int(window_size / 2)-1
        right_window = int(window_size / 2)

    
    print('Successfully initialized half-windows')
    print(f'left window ::   {left_window}\nright window ::   {right_window}')
    
    basename = os.path.basename(input_csv).replace('.csv', '')  # remove .csv extension

    df = pd.read_csv(input_csv)
    df['ur_trajectory'] = df['trajectory'].apply(lambda t: f"{basename}::{t}")
    
    counts = df['trajectory'].value_counts()
    valid_traj = counts[counts >= 0].index #### keep all trajectories in rolling windows analysis
    df = df[df['trajectory'].isin(valid_traj)]
    
    all_rows = []
    new_traj_id = 0

    for i, group in df.groupby('ur_trajectory'):
        group = group.sort_values('frame').reset_index(drop=True)
        for i in range(left_window, len(group) - right_window):
            window = group.iloc[i - left_window : i + right_window + 1].copy()
            window['focal_point'] = False
            window.iloc[left_window, window.columns.get_loc('focal_point')] = True
            window['trajectory'] = new_traj_id  # re-index trajectory
            all_rows.append(window)
            new_traj_id += 1

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame(columns=df.columns.tolist() + ['focal_point'])
        trajectory_counts = out_df['ur_trajectory'].value_counts()
        out_df['ur_length'] = out_df['ur_trajectory'].map(trajectory_counts)

    SA = StateArray.from_detections(out_df , **settings['saspt'])
    marginal_D = SA.posterior_assignment_probabilities.sum(axis=1) ## marginalize over errors

    print("extracting posterior and formatting export")
    print("")
    MLE_D = settings['saspt']['diff_coefs'][marginal_D.argmax(axis = 0)] ## for every sub-trajectory, lookup it's MLE diffusion rate

    detections_df = SA.trajectories.detections
    MLE_D = pd.DataFrame(MLE_D , columns = ['MLE_D'])
    MLE_D['trajectory'] = range(MLE_D.shape[0])
    detections_df = pd.merge(detections_df , MLE_D , on='trajectory')
    detections_df = detections_df[detections_df['focal_point'] == True]
    merge_key = ['y' , 'x' , 'I0' , 'frame' , 'ur_trajectory']
    new = pd.merge(
        df,
        detections_df[merge_key + ['MLE_D']],
        on=merge_key,
        how='left'
    )
    new = new[merge_key + ['MLE_D']]
    
    output_file = basename.replace('_traj' , '_rollingMLE.csv')
    output_file = os.path.join(settings['io']['rolling_window_directory'] , output_file)
    new.to_csv(output_file , index = False)
    
    return new


def executeSASPT(input_csv , settings):
    print(f"Sending posterior output to : {settings['io']['post_directory']}")
    print(f"Sending MLE output to : {settings['io']['MLE_directory']}")
    print("")

    print(f"Loading spots from : {input_csv}")
    print("")
    spots = pd.read_csv(input_csv)

    ## filter for trajectories with at least 4 detections
    counts = spots['trajectory'].value_counts()
    valid_traj = counts[counts >= 0].index ### include all trajecories at this step
    spots = spots[spots['trajectory'].isin(valid_traj)]
    ## filter spots with low mean intensity
    spots = spots.groupby("trajectory").filter(lambda g: g["I0"].mean() > settings['quot']['track']['min_I0'])

    print("running SASPT")
    SA = StateArray.from_detections(spots , **settings['saspt'])
    marginal_D = SA.posterior_assignment_probabilities.sum(axis=1) ## marginalize over errors

    print("extracting posterior and formatting export")
    MLE_D = settings['saspt']['diff_coefs'][marginal_D.argmax(axis = 0)] ## for every sub-trajectory, lookup it's MLE diffusion rate
    soft_D = marginal_D.sum(axis=1)
    softD_df = pd.DataFrame({"D" : (settings['saspt']['diff_coefs']) , "density" : soft_D , "file" : os.path.basename(input_csv)})

    #### generate output file markup // can't really remember why this is neccessary or useful, but just go with it...
    detections_df = SA.trajectories.detections
    MLE_D = pd.DataFrame(MLE_D , columns = ['MLE_D'])
    MLE_D['trajectory'] = range(MLE_D.shape[0])
    detections_df = pd.merge(detections_df , MLE_D , on='trajectory')

    useful_trajname = os.path.basename(os.path.dirname(input_csv)) ## quality string
    useful_trajname = f"{useful_trajname}::{os.path.basename(input_csv)}" ## filename string

    MLE_df = SA.trajectories.detections.drop_duplicates(subset='trajectory')
    MLE_df = pd.merge(MLE_df , MLE_D , on='trajectory')
    MLE_df = MLE_df[['orig_trajectory' , 'track_length', 'MLE_D']]
    MLE_df['source_file'] = input_csv
    MLE_df["orig_trajectory"] = useful_trajname + "::" + MLE_df["orig_trajectory"].astype(str)

    print("exporting data")
    print("")

    basename = os.path.basename(input_csv)
    MLE_filename = basename.replace('_traj' , '_MLE')
    MLE_df.to_csv(os.path.join(settings['io']['MLE_directory'] , MLE_filename) , index = False)
    posterior_filename = basename.replace('_traj' , '_posterior')
    softD_df.to_csv(os.path.join(settings['io']['post_directory'] , posterior_filename), index = False)
    




def split_and_reform(settings , pkl_path):
    data_dir = os.path.join(settings['io']['home_directory'] , settings['io']['data_filepath'])
    grouped_files = glob.glob(os.path.join(data_dir,"*.nd2")) ## FOR THIS ANALYSIS I DON'T ACTUALLY WANT THE FILES GROUPED WHEN I DO MAP/REDUCE ;; GROUP LATER
    grouped_files = {item: [item] for item in grouped_files}
     # Print the result
    job_ids = {}
    for key, file_list in grouped_files.items():
        job_ids[key] = []
        print(f"{key}:")
        print(f"  - {len(file_list)} files")
        print(f"    starting execution")
        for f in file_list:
            #filename = os.path.join(data_dir , f)
            filename = f
            frames = pims.open(filename)
            # Convert to a NumPy array (T, Y, X)
            stack = np.array(frames)
            tiles, positions = split_image_stack(stack , tile_size = settings['io']['split_dimension'] , overlap = 10)
            split_filenames = save_tiles(tiles, positions, out_dir=settings['io']['split_directory'], nd2_filename=f"{f}")

            for f_s in split_filenames:            
                filename = os.path.join(settings['io']['split_directory'] , f_s)
                output_basename = os.path.basename(filename)
                output_basename = output_basename.replace('.tif' , '_splitTraj.csv')
                out_csv = os.path.join(settings['io']['split_traj_directory'] , output_basename)
                #print(f"Processing file: {f}")            
                sbatch_command = [
                    "sbatch",
                    f"{settings['io']['code_directory']}/runQUOT.sh" ,       # Your SLURM script
                    "-c" ,  f"{settings['io']['code_directory']}",
                    "-i" ,  f"{filename}",
                    "-o" ,  f"{out_csv}",
                    "-s" , f"{pkl_path}" , 
                    "-m" , f"track"
                ]
                result = subprocess.run(sbatch_command, capture_output=True, text=True)
                job_ids[key].append(int(result.stdout.strip().split()[-1]))

    ## in this block, re-aggregate the files after all sub-tiles have been processed
    all_processes = job_ids.keys()
    already_finished = set()
    while True:
        newly_finished = set(keys_with_all_jobs_finished(job_ids)) - already_finished
        for key in newly_finished:
            print(f"{key} has just finished. Re-aggregating trajectories")            
            split_corename = os.path.basename(key)
            split_corename = split_corename.replace(".nd2", "*")
            matching_files = glob.glob(os.path.join(settings['io']['split_traj_directory'], split_corename))

            global_trajectory_offset = 0
            updated_dataframes = []

            for filename in matching_files:
                df = update_csv_with_offsets(filename)

                if "trajectory" in df.columns and not df["trajectory"].dropna().empty:
                    df["trajectory"] += global_trajectory_offset
                    global_trajectory_offset = df["trajectory"].max() + 1

                updated_dataframes.append(df)

            merged_df = pd.concat(updated_dataframes, ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=["frame", "x_detect", "y_detect"]) ## drop any rows that have identical frame,x_detect,y_detect

            csv_filename = split_corename.replace("*" , "_traj.csv")
            merged_df.to_csv(os.path.join(settings['io']['traj_directory'] , csv_filename) , index = False)

            for fr in matching_files:
                try:
                    os.remove(fr)
                except FileNotFoundError:
                    print(f"File not found, skipping: {fr}")
                except Exception as e:
                    print(f"Error removing {fr}: {e}")

            already_finished.add(key)

        if already_finished == set(job_ids.keys()):
            print("Re-aggregated trajectories.")
            break

        time.sleep(5)  # adjust polling interval as needed


    ## in this block, proceed with the rest of runQUOT, i.e. perform SASPT and dyanmics modules
    traj_dir = os.path.join(settings['io']['home_directory'] , settings['io']['traj_directory'])
    grouped_files = glob.glob(os.path.join(traj_dir,"*.csv")) ## FOR THIS ANALYSIS I DON'T ACTUALLY WANT THE FILES GROUPED WHEN I DO MAP/REDUCE ;; GROUP LATER
    grouped_files = {item: [item] for item in grouped_files}
     # Print the result
    saspt_job_ids = {}
    for key, file_list in grouped_files.items():
        saspt_job_ids[key] = []
        print(f"{key}:")
        print(f"  - {len(file_list)} files")
        print(f"    Picking up with SASPT and rolling windows on reformed trajectories")
        for f in file_list:

            #filename = os.path.join(settings['io']['traj_directory'] , f)

            print(f"Check this file, i think the path has been malformed in the past-- {f}")
            
            #print(f"Processing file: {f}")            
            sbatch_command = [
                "sbatch",
                f"{settings['io']['code_directory']}/runQUOT.sh" ,       # Your SLURM script
                "-c" ,  f"{settings['io']['code_directory']}",
                "-i" ,  f"{f}",
                "-o" ,  f"{f}",
                "-s" , f"{pkl_path}" , 
                "-m" , f"skipQuot"
            ]
            result = subprocess.run(sbatch_command, capture_output=True, text=True)
            saspt_job_ids[key].append(int(result.stdout.strip().split()[-1]))
        
    all_processes = saspt_job_ids.keys()
    already_finished = set()
    while True:
        newly_finished = set(keys_with_all_jobs_finished(saspt_job_ids)) - already_finished
        if already_finished == set(saspt_job_ids.keys()):
            print("All jobs finished.")
            break
        time.sleep(5)  # adjust polling interval as needed
    shutil.rmtree(settings['io']['split_traj_directory'], ignore_errors=True)
    shutil.rmtree(settings['io']['split_directory'], ignore_errors=True)
    

    
