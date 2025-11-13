import numpy as np
from saspt import RBME
import os
import pims
import glob


def get_default_settings():
    settings = {}
    settings['io'] = {}
    settings['quot'] = {}
    settings['saspt'] = {}
    settings['plot'] = {}
    
    settings['io']['home_directory'] = ""
    settings['io']['data_filepath'] = "raw_data"
    settings['io']['split_dimension'] = 0 ## pixel dimension to split image into -- this is useful for dense / large images, i.e. for slow SPT
    settings['io']['forceQUOT'] = True ## pixel dimension to split image into -- this is useful for dense / large images, i.e. for slow SPT
    settings['io']['plotOnly'] = False ## pixel dimension to split image into -- this is useful for dense / large images, i.e. for slow SPT

    settings['quot']['filter'] = {}
    settings['quot']['filter']['start'] = 0 
    settings['quot']['filter']['method'] = 'identity'
    settings['quot']['filter']['chunk_size'] = 100    
    
    settings['quot']['detect'] = {}
    settings['quot']['detect']['method'] = 'llr'
    settings['quot']['detect']['k'] = 2.0 ## this is related to expected spot size i think
    settings['quot']['detect']['w'] = 15 ## window dimension over which to look for spots
    settings['quot']['detect']['t'] = 20.00 ## likelihood ratio testing the model that the window contains a spot vs doesn't
    
    '''
    ## try LoG filtereing
    settings['quot']['detect'] = {}
    settings['quot']['detect']['method'] = 'log'
    settings['quot']['detect']['k'] = 1 ## this is related to expected spot size i think
    settings['quot']['detect']['w'] = 11 ## window dimension over which to look for spots
    settings['quot']['detect']['t'] = 20 ## likelihood ratio testing the model that the window contains a spot vs doesn't
    
    settings['quot']['detect'] = {}
    settings['quot']['detect']['method'] = 'hess_det_var'
    settings['quot']['detect']['k'] = 1.5 ## this is related to expected spot size i think
    settings['quot']['detect']['w0'] = 15 ## window dimension over which to look for spots
    settings['quot']['detect']['w1'] = 9 ## window dimension over which to look for spots
    settings['quot']['detect']['t'] = 20.0 ## likelihood ratio testing the model that the window contains a spot vs doesn't
    '''
    
    settings['quot']['localize'] = {}
    settings['quot']['localize']['method'] = 'ls_int_gaussian'
    settings['quot']['localize']['window_size'] = 15 ## window size of 7 was bad. 11 is decent. 15 was best by a hair.
    settings['quot']['localize']['sigma'] = 2.5 
    settings['quot']['localize']['ridge'] = 0.00001
    settings['quot']['localize']['max_iter'] = 30
    settings['quot']['localize']['damp'] = 1 # higher damp means faster convergence
    #settings['quot']['localize']['camera_gain'] = 100
    settings['quot']['localize']['camera_bg'] = 100
    
    settings['quot']['track'] = {}
    settings['quot']['track']['method'] = 'euclidean' ## conservative is too conservative. euclidean should match trackmate
    settings['quot']['track']['pixel_size_um'] = 0.11
    settings['quot']['track']['frame_interval'] = 0.006 ## frame interval in seconds
    settings['quot']['track']['search_radius'] = 1.6 ## linking distance in microns
    settings['quot']['track']['max_blinks'] = 0
    settings['quot']['track']['min_I0'] = 0.0 ## "spot intensity above background." might be a hard parameter to tune
    settings['quot']['track']['scale'] = 1.0
    
    '''
    settings['trackmate'] = {}
    settings['trackmate']['radius'] = 0.2
    settings['trackmate']['quality'] = 14.0
    settings['trackmate']['linking_distance'] = 1.6
    settings['trackmate']['channel'] = 1
    '''

    settings['saspt']['likelihood_type'] = RBME
    settings['saspt']['pixel_size_um'] = 0.11
    settings['saspt']['frame_interval'] = 0.006
    settings['saspt']['focal_depth'] = 0.1
    settings['saspt']['progress_bar'] = 'False'
    settings['saspt']['splitsize'] = 5
    settings['saspt']['sample_size'] = 100000
    settings['saspt']['num_workers'] = 4
    settings['saspt']['diff_coefs'] = np.power(10,(np.linspace(-2,2,125)))
    settings['saspt']['loc_errors'] = np.linspace(0.025 , 0.028 , 5)
    settings['saspt']['start_frame'] = 0
    
    settings['plot']['xlim'] = (-2,2)
    settings['plot']['bleach_xlim'] = (-300,50)
    settings['plot']['ylim'] = [ 0 , 5 ]
    settings['plot']['nbins'] = 100
    settings['plot']['tick_positions'] = (-2, -1, 0, 1, 2)
    settings['plot']['mult_on_sd'] = 5.0
    settings['plot']['barGraphBreaks'] = [-10000 , 10000]
    settings['plot']['barGraphLabels'] = ['None']
    
    return(settings)


def update_default_settings_for_analysis(settings , home_directory):
    
    settings['io']['home_directory'] = home_directory
    settings['io']['data_directory'] = os.path.join(settings['io']['home_directory'] , settings['io']['data_filepath'])

    traj_filestring = f"tracking_output_q={str(float(settings['quot']['detect']['t'])).replace('.','p')}"
    settings['io']['analysis_directory'] = os.path.join(settings['io']['home_directory'] , traj_filestring)
    settings['io']['traj_directory'] = os.path.join(settings['io']['analysis_directory'] , 'trajectories')
    settings['io']['post_directory'] = os.path.join(settings['io']['analysis_directory'] , 'posterior')
    settings['io']['MLE_directory'] = os.path.join(settings['io']['analysis_directory'] , 'MLE')
    settings['io']['plot_directory'] = os.path.join(settings['io']['analysis_directory'] , 'plots')
    #settings['io']['split_directory'] = os.path.join(settings['io']['analysis_directory'] , 'split_directory')
    #settings['io']['split_traj_directory'] = os.path.join(settings['io']['analysis_directory'] , 'split_trajectories')
    settings['io']['rolling_window_directory'] = os.path.join(settings['io']['analysis_directory'] , 'rolling_windows')
    settings['io']['movie_directory'] = os.path.join(settings['io']['analysis_directory'] , 'movies')

    settings['quot']['filter']['start'] = 0

    os.makedirs(settings['io']['analysis_directory'], exist_ok=True)
    os.makedirs(settings['io']['traj_directory'], exist_ok=True)
    os.makedirs(settings['io']['post_directory'], exist_ok=True)
    os.makedirs(settings['io']['MLE_directory'], exist_ok=True)
    os.makedirs(settings['io']['plot_directory'], exist_ok=True)
    #os.makedirs(settings['io']['split_directory'] , exist_ok=True)
    #os.makedirs(settings['io']['split_traj_directory'] , exist_ok=True)
    os.makedirs(settings['io']['rolling_window_directory'] , exist_ok=True)
    os.makedirs(settings['io']['movie_directory'] , exist_ok=True)
    
    return(settings)


def update_settings_with_image_metadata(settings , cond = None):
    
    
    #arbitrary_image = os.listdir(settings['io']['data_directory'])[0]
    
    if cond == None:
        arbitrary_image = glob.glob(f"{settings['io']['data_directory']}/*.nd2")[0]
    else:
        arbitrary_image = glob.glob(f"{settings['io']['data_directory']}/*{cond}*.nd2")[0]
    
    #img = pims.open(os.path.join(settings['io']['data_directory'] ,arbitrary_image))
    img = pims.open(arbitrary_image)
    frame = img[0]
    bg_level = np.median(frame)

    dt = 1 / img.frame_rate
    
    settings['quot']['localize']['camera_bg'] = bg_level
    settings['quot']['track']['frame_interval'] = dt
    settings['saspt']['frame_interval'] = dt
    
    return(settings)