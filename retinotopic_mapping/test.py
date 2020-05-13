import numpy as np
import StimulusRoutines as stim
from DisplayStimulus import DisplaySequence
from MonitorSetup import Monitor, Indicator
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mon = Monitor(resolution=(1200, 1920), dis=15., mon_width_cm=52., mon_height_cm=32.)
    #mon.plot_map()
    
    ind = Indicator(mon)
    print(ind.get_size_pixel())
    ds = DisplaySequence(log_dir='C:/data', is_by_index=True,display_screen=1)
    fc = stim.FlashingCircle(monitor=mon, indicator=ind, coordinate='degree', center=(0., 60.),
                            radius=10., is_smooth_edge=False, smooth_width_ratio=0.2,
                            smooth_func=stim.blur_cos, color=1., flash_frame_num=60,
                            pregap_dur=2., postgap_dur=3., background=-1., midgap_dur=1.,
                            iteration=100)
    ds.set_stim(fc)
    ds.trigger_display()
    plt.show()
    