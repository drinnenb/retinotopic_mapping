# -*- coding: utf-8 -*-
"""
Example script to test that everything is working. Running this script is a
good first step for trying to debug your experimental setup and is also a
great tool to familiarize yourself with the parameters that are used to
generate each specific stimulus.

!!!IMPORTANT!!!
Note that once you are displaying stimulus, if you want to stop the code from
running all you need to do is press either one of the 'Esc' or 'q' buttons.
"""

import numpy as np
import matplotlib.pyplot as plt
import StimulusRoutines as stim
from MonitorSetup import Monitor, Indicator
from DisplayStimulus import DisplaySequence
import json

"""
To get up and running quickly before performing any experiments it is 
sufficient to setup two monitors -- one for display and one for your python 
environment. If you don't have two monitors at the moment it is doable with
only one. 

Edit the following block of code with your own monitors respective parameters.
Since this script is for general debugging and playing around with the code, 
we will arbitrarily populate variables that describe the geometry of where 
the mouse will be located during an experiment. All we are interested in 
here is just making sure that we can display stimulus on a monitor and learning
how to work with the different stimulus routines.
"""


# ======================== monitor parameters ==================================
mon_resolution = (1080,1920) #enter your monitors resolution (height, width)
mon_width_cm = 34.5 #enter your monitors width in cm
mon_height_cm = 19.5 #enter your monitors height in cm
mon_refresh_rate = 60  #enter your monitors rate in Hz

# The following variables correspond to the geometry of your setup don't worry about them for now.
mon_C2T_cm = mon_height_cm / 2.  # center (projection point from mouse eye to the monitor) to monitor top edge in cm
mon_C2A_cm = mon_width_cm / 2.  # center (projection point from mouse eye to the monitor) to monitor anterior edge in cm
mon_center_coordinates = (0., 60.) # the visual coordinates of center (altitude, azimuth)
mon_dis_cm = 11. # cm from mouse eye to the monitor
mon_downsample_rate = 10 # downsample rate of the displayed images relative to the monitor resolution.
# the both numbers in mon_resolution should be divisble by this number
# ======================== monitor parameters ==================================

# ======================== indicator parameters ================================
ind_width_cm = 0.5
ind_height_cm = 0.5
ind_position = 'northwest'
ind_is_sync = 'True'
ind_freq = .5
# ======================== indicator parameters ================================

# ============================ generic stimulus parameters ======================
pregap_dur = 2.
postgap_dur = 3.
background = 0.4 #color from -1 to 1; when from 0 to 1, midgrey value is at 0.7 for this type of monitor
coordinate = 'degree'
# ===============================================================================

# ============================ LocallySparseNoise ===============================
lsn_subregion = None
lsn_min_distance = 25.
lsn_grid_space = (5., 5.)
lsn_probe_size = (5., 5.)
lsn_probe_orientation = 0.
lsn_probe_frame_num = 15 #duration of stimulus in frame rate units (60Hz, 15 corresponds to 250 ms)
lsn_sign = 'ON-OFF'
lsn_iteration = 115
lsn_is_include_edge = True
# ===============================================================================

# ============================ DisplaySequence ====================================
ds_log_dir = r'D:\data'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1 #10
ds_mouse_id = 'M68'
ds_user_id = 'USER'
ds_psychopy_mon = 'testMonitor'
ds_is_by_index = True
ds_is_interpolate = False
ds_is_triggered = False #set to true if you want to start stimulus with trigger
ds_trigger_event = "positive_edge" #trigger when turning on
ds_trigger_NI_dev = 'Dev1' # device and lines to where trigger input is connected to
ds_trigger_NI_port = 1
ds_trigger_NI_line = 1
ds_is_sync_pulse = True # wheter to send out a sync pulse, digital signal that is true for a few ms at beginning of each frame
ds_sync_pulse_NI_dev = 'Dev1' # which device/port to use for sync pulse, connect this to another DAQ to record sync pulses
ds_sync_pulse_NI_port = 1
ds_sync_pulse_NI_line = 0
ds_display_screen = 1 # 0 is normally main display, 1 and 2 are secondary displays
ds_initial_background_color = 0.4 #color from -1 to 1; when from 0 to 1, midgrey value is at 0.7 for this type of monitor
ds_save_sequence = False #if it should save display patterns as tiff stack
# =================================================================================


# Initialize Monitor object
mon = Monitor(resolution=mon_resolution, dis=mon_dis_cm, mon_width_cm=mon_width_cm, mon_height_cm=mon_height_cm,
              C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm, center_coordinates=mon_center_coordinates,
              downsample_rate=mon_downsample_rate)

# plot warpped monitor coordinates
mon.plot_map()
plt.show()

# initialize Indicator object
ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm, position=ind_position, is_sync=ind_is_sync,
                freq=ind_freq)


# initialize LocallySparseNoise object
lsn = stim.LocallySparseNoise(monitor=mon, indicator=ind, pregap_dur=pregap_dur,
                              postgap_dur=postgap_dur, coordinate=coordinate,
                              background=background, subregion=lsn_subregion,
                              grid_space=lsn_grid_space, sign=lsn_sign,
                              probe_size=lsn_probe_size, probe_orientation=lsn_probe_orientation,
                              probe_frame_num=lsn_probe_frame_num, iteration=lsn_iteration,
                              is_include_edge=lsn_is_include_edge, min_distance=lsn_min_distance)

# initialize DisplaySequence object
ds = DisplaySequence(log_dir=ds_log_dir, backupdir=ds_backupdir,
                     identifier=ds_identifier, display_iter=ds_display_iter,
                     mouse_id=ds_mouse_id, user_id=ds_user_id,
                     psychopy_mon=ds_psychopy_mon, is_by_index=ds_is_by_index,
                     is_interpolate=ds_is_interpolate, is_triggered=ds_is_triggered,
                     trigger_event=ds_trigger_event, trigger_NI_dev=ds_trigger_NI_dev,
                     trigger_NI_port=ds_trigger_NI_port, trigger_NI_line=ds_trigger_NI_line,
                     is_sync_pulse=ds_is_sync_pulse, sync_pulse_NI_dev=ds_sync_pulse_NI_dev,
                     sync_pulse_NI_port=ds_sync_pulse_NI_port,
                     sync_pulse_NI_line=ds_sync_pulse_NI_line,
                     display_screen=ds_display_screen,
                     initial_background_color=ds_initial_background_color,is_save_sequence= True)

# display
# =============================== display =========================================
ds.set_stim(lsn)
ds.trigger_display()
plt.show()
# =================================================================================