"""
Contains various stimulus routines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
#import h5py
from tools import ImageAnalysis as ia
#from tools import FileTools as ft

try:
    import skimage.external.tifffile as tf
except ImportError:
    import tifffile as tf


def in_hull(p, hull):
    """
    Determine if points in `p` are in `hull`

    `p` should be a `NxK` coordinate matrix of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    Parameters
    ----------
    p : array
        NxK coordinate matrix of N points in K dimensions
    hull :
        either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay
        triangulation will be computed

    Returns
    -------
    is_in_hull : ndarray of int
        Indices of simplices containing each point. Points outside the
        triangulation get the value -1.
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def get_warped_probes(deg_coord_alt, deg_coord_azi, probes, width,
                      height, ori=0., background_color=0.):
    """
    Generate a frame (matrix) with multiple probes defined by 'porbes', `width`,
    `height` and orientation in degrees. visual degree coordinate of each pixel is
    defined by deg_coord_azi, and deg_coord_alt

    Parameters
    ----------
    deg_coord_alt : ndarray
        2d array of warped altitude coordinates of monitor pixels
    deg_coord_azi : ndarray
        2d array of warped azimuth coordinates of monitor pixels
    probes : tuple or list
        each element of probes represents a single probe (center_alt, center_azi, sign)
    width : float
         width of the square in degrees
    height : float
         height of the square in degrees
    ori : float
        angle in degree, should be [0., 180.]
    foreground_color : float, optional
         color of the noise pixels, takes values in [-1,1] and defaults to `1.`
    background_color : float, optional
         color of the background behind the noise pixels, takes values in
         [-1,1] and defaults to `0.`
    Returns
    -------
    frame : ndarray
         the warped s
    """

    frame = np.ones(deg_coord_azi.shape, dtype=np.float32) * background_color

    # if ori < 0. or ori > 180.:
    #      raise ValueError, 'ori should be between 0 and 180.'

    ori_arc = (ori % 360.) * 2 * np.pi / 360.

    for probe in probes:
        dis_width = np.abs(np.cos(ori_arc) * (deg_coord_azi - probe[1]) +
                           np.sin(ori_arc) * (deg_coord_alt - probe[0]))

        dis_height = np.abs(np.cos(ori_arc + np.pi / 2) * (deg_coord_azi - probe[1]) +
                            np.sin(ori_arc + np.pi / 2) * (deg_coord_alt - probe[0]))

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # fig1 = ax1.imshow(dis_width)
        # ax1.set_title('width')
        # f.colorbar(fig1, ax=ax1)
        # fig2 = ax2.imshow(dis_height)
        # ax2.set_title('height')
        # f.colorbar(fig2, ax=ax2)
        # plt.show()

        frame[np.logical_and(dis_width <= width / 2.,
                             dis_height <= height / 2.)] = probe[2]

    return frame


def blur_cos(dis, sigma):
    """
    return a smoothed value [0., 1.] given the distance to center (with sign)
    and smooth width. this is using cosine curve to smooth edge

    parameters
    ----------
    dis : ndarray
        array that store the distance from the current pixel to blurred band center
    sigma : float
        definition of the width of blurred width, here is the length represent
        half cycle of the cosin function

    returns
    -------
    blurred : float
        blurred value
    """
    dis_f = dis.astype(np.float32)
    sigma_f = abs(float(sigma))

    blur_band = (np.cos((dis_f - (sigma_f / -2.)) * np.pi / sigma_f) + 1.) / 2.

    # plt.imshow(blur_band)
    # plt.show()

    blur_band[dis_f < (sigma_f / -2.)] = 1.
    blur_band[dis_f > (sigma_f / 2.)] = 0.

    # print blur_band.dtype

    return blur_band


def get_circle_mask(map_alt, map_azi, center, radius, is_smooth_edge=False,
                    blur_ratio=0.2, blur_func=blur_cos, is_plot=False):
    """
    Generate a binary mask of a circle with given `center` and `radius`

    The binary mask is generated on a map with coordinates for each pixel
    defined by `map_x` and `map_y`

    Parameters
    ----------
    map_alt  : ndarray
        altitude coordinates for each pixel on a map
    map_azi  : ndarray
        azimuth coordinates for each pixel on a map
    center : tuple
        coordinates (altitude, azimuth) of the center of the binary circle mask
    radius : float
        radius of the binary circle mask
    is_smooth_edge : bool
        if True, use 'blur_ratio' and 'blur_func' to smooth circle edge
    blur_ratio : float, option, default 0.2
        the ratio between blurred band width to radius, should be smaller than 1
        the middle of blurred band is the circle edge
    blur_func : function object to blur edge
    is_plot : bool

    Returns
    -------
    circle_mask : ndarray (dtype np.float32) with same shape as map_alt and map_azi
        if is_smooth_edge is True
            weighted circle mask, with smoothed edge
        if is_smooth_edge is False
            binary circle mask, takes values in [0.,1.]
    """

    if map_alt.shape != map_azi.shape:
        raise ValueError('map_alt and map_azi should have same shape.')

    if len(map_alt.shape) != 2:
        raise ValueError('map_alt and map_azi should be 2-d.')

    dis_mat = np.sqrt((map_alt - center[0]) ** 2 + (map_azi - center[1]) ** 2)
    # plt.imshow(dis_mat)
    # plt.show()

    if is_smooth_edge:
        sigma = radius * blur_ratio
        circle_mask = blur_func(dis=dis_mat - radius, sigma=sigma)
    else:
        circle_mask = np.zeros(map_alt.shape, dtype=np.float32)
        circle_mask[dis_mat <= radius] = 1.

    if is_plot:
        plt.imshow(circle_mask)
        plt.show()

    return circle_mask


def get_grating(alt_map, azi_map, dire=0., spatial_freq=0.1,
                center=(0., 60.), phase=0., contrast=1.):
    """
    Generate a grating frame with defined spatial frequency, center location,
    phase and contrast

    Parameters
    ----------
    azi_map : ndarray
        x coordinates for each pixel on a map
    alt_map : ndarray
        y coordinates for each pixel on a map
    dire : float, optional
        orientation angle of the grating in degrees, defaults to 0.
    spatial_freq : float, optional
        spatial frequency (cycle per unit), defaults to 0.1
    center : tuple, optional
        center coordinates of circle {alt, azi}
    phase : float, optional
        defaults to 0.
    contrast : float, optional
        defines contrast. takes values in [0., 1.], defaults to 1.

    Returns
    -------
    frame :
        a frame as floating point 2-d array with grating, value range [0., 1.]
    """

    if azi_map.shape != alt_map.shape:
        raise ValueError('map_alt and map_azi should have same shape.')

    if len(azi_map.shape) != 2:
        raise ValueError('map_alt and map_azi should be 2-d.')

    axis_arc = ((dire + 90.) * np.pi / 180.) % (2 * np.pi)

    map_azi_h = np.array(azi_map, dtype=np.float32)
    map_alt_h = np.array(alt_map, dtype=np.float32)

    distance = (np.sin(axis_arc) * (map_azi_h - center[1]) -
                np.cos(axis_arc) * (map_alt_h - center[0]))

    grating = np.sin(distance * 2 * np.pi * spatial_freq - phase)

    grating = grating * contrast  # adjust contrast

    grating = (grating + 1.) / 2.  # change the scale of grating to be [0., 1.]

    return grating


# def get_sparse_loc_num_per_frame(min_alt, max_alt, min_azi, max_azi, minimum_dis):
#     """
#     given the subregion of visual space and the minmum distance between the probes
#     within a frame (definition of sparseness), return generously how many probes
#     will be presented of a given frame
#
#     Parameters
#     ----------
#     min_alt : float
#         minimum altitude of display region, in visual degrees
#     max_alt : float
#         maximum altitude of display region, in visual degrees
#     min_azi : float
#         minimum azimuth of display region, in visual degrees
#     max_azi : float
#         maximum azimuth of display region, in visual degrees
#     minimum_dis : float
#         minimum distance allowed among probes within a frame
#
#     returns
#     -------
#     probe_num_per_frame : uint
#         generously how many probes will be presented in a given frame
#     """
#     if min_alt >= max_alt:
#         raise ValueError('min_alt should be less than max_alt.')
#
#     if min_azi >= max_azi:
#         raise ValueError('min_azi should be less than max_azi.')
#
#     min_alt = float(min_alt)
#     max_alt = float(max_alt)
#     min_azi = float(min_azi)
#     max_azi = float(max_azi)
#
#     area_tot = (max_alt - min_alt) * (max_azi - min_azi)
#     area_circle = np.pi * (minimum_dis ** 2)
#     probe_num_per_frame = int(np.ceil((2.0 * (area_tot / area_circle))))
#     return probe_num_per_frame


def get_grid_locations(subregion, grid_space, monitor_azi, monitor_alt, is_include_edge=True,
                       is_plot=False):
    """
    generate all the grid points in display area (covered by both subregion and
    monitor span), designed for SparseNoise and LocallySparseNoise stimuli.

    Parameters
    ----------
    subregion : list, tuple or np.array
        the region on the monitor that will display the sparse noise,
        [min_alt, max_alt, min_azi, max_azi], all floats
    grid_space : tuple or list of two floats
        grid size of probes to be displayed, [altitude, azimuth]
    monitor_azi : 2-d array
        array mapping monitor pixels to azimuth in visual space
    monitor_alt : 2-d array
        array mapping monitor pixels to altitude in visual space
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    is_plot : bool

    Returns
    -------
    grid_locations : n x 2 array,
        refined [alt, azi] pairs of probe centers going to be displayed
    """

    rows = np.arange(subregion[0],
                     subregion[1] + grid_space[0],
                     grid_space[0])
    columns = np.arange(subregion[2],
                        subregion[3] + grid_space[1],
                        grid_space[1])

    azis, alts = np.meshgrid(columns, rows)

    grid_locations = np.transpose(np.array([alts.flatten(), azis.flatten()]))

    left_alt = monitor_alt[:, 0]
    right_alt = monitor_alt[:, -1]
    top_azi = monitor_azi[0, :]
    bottom_azi = monitor_azi[-1, :]

    left_azi = monitor_azi[:, 0]
    right_azi = monitor_azi[:, -1]
    top_alt = monitor_alt[0, :]
    bottom_alt = monitor_alt[-1, :]

    left_azi_e = left_azi - grid_space[1]
    right_azi_e = right_azi + grid_space[1]
    top_alt_e = top_alt + grid_space[0]
    bottom_alt_e = bottom_alt - grid_space[0]

    all_alt = np.concatenate((left_alt, right_alt, top_alt, bottom_alt))
    all_azi = np.concatenate((left_azi, right_azi, top_azi, bottom_azi))

    all_alt_e = np.concatenate((left_alt, right_alt, top_alt_e, bottom_alt_e))
    all_azi_e = np.concatenate((left_azi_e, right_azi_e, top_azi, bottom_azi))

    monitorPoints = np.array([all_alt, all_azi]).transpose()
    monitorPoints_e = np.array([all_alt_e, all_azi_e]).transpose()

    # get the grid points within the coverage of monitor
    if is_include_edge:
        grid_locations = grid_locations[in_hull(grid_locations, monitorPoints_e)]
    else:
        grid_locations = grid_locations[in_hull(grid_locations, monitorPoints)]

    # grid_locations = np.array([grid_locations[:, 1], grid_locations[:, 0]]).transpose()

    if is_plot:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(monitorPoints[:, 1], monitorPoints[:, 0], '.r', label='monitor')
        ax.plot(monitorPoints_e[:, 1], monitorPoints_e[:, 0], '.g', label='monitor_e')
        ax.plot(grid_locations[:, 1], grid_locations[:, 0], '.b', label='grid')
        ax.legend()
        plt.show()

    return grid_locations


class Stim(object):
    """
    generic class for visual stimulation. parent class for individual
    stimulus routines.

    Parameters
    ----------
    monitor : monitor object
         the monitor used to display stimulus in the experiment
    indicator : indicator object
         the indicator used during stimulus
    background : float, optional
        background color of the monitor screen when stimulus is not being
        presented, takes values in [-1,1] and defaults to `0.` (grey)
    coordinate : str {'degree', 'linear'}, optional
        determines the representation of pixel coordinates on monitor,
        defaults to 'degree'
    pregap_dur : float, optional
        duration of gap period before stimulus, measured in seconds, defaults
        to `2.`
    postgap_dur : float, optional
        duration of gap period after stimulus, measured in seconds, defaults
        to `3.`
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 pregap_dur=2., postgap_dur=3.):
        """
        Initialize visual stimulus object
        """

        self.monitor = monitor
        self.indicator = indicator

        if background < -1. or background > 1.:
            raise ValueError('parameter "background" should be a float within [-1., 1.].')
        else:
            self.background = float(background)

        if coordinate not in ['degree', 'linear']:
            raise ValueError('parameter "coordinate" should be either "degree" or "linear".')
        else:
            self.coordinate = coordinate

        if pregap_dur >= 0.:
            self.pregap_dur = float(pregap_dur)
        else:
            raise ValueError('pregap_dur should be no less than 0.')

        if postgap_dur >= 0.:
            self.postgap_dur = float(postgap_dur)
        else:
            raise ValueError('postgap_dur should be no less than 0.')

        self.clear()

    @property
    def pregap_frame_num(self):
        return int(self.pregap_dur * self.monitor.refresh_rate)

    @property
    def postgap_frame_num(self):
        return int(self.postgap_dur * self.monitor.refresh_rate)

    def generate_frames(self):
        """
        place holder of function "generate_frames" for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def generate_movie(self):
        """
        place holder of function 'generate_movie' for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. '
              'See documentation in the respective stimulus. \n'
              'It is possible that full sequence generation is not'
              'implemented in this particular stimulus. Try '
              'generate_movie_by_index() function to see if indexed '
              'sequence generation is implemented.')

    def _generate_frames_for_index_display(self):
        """
        place holder of function _generate_frames_for_index_display()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def _generate_display_index(self):
        """
        place holder of function _generate_display_index()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def generate_movie_by_index(self):
        """
        place holder of function generate_movie_by_index()
        for each specific stimulus
        """
        print('Nothing executed! This is a place holder function. \n'
              'See documentation in the respective stimulus.')

    def clear(self):
        if hasattr(self, 'frames'):
            del self.frames
        if hasattr(self, 'frames_unique'):
            del self.frames_unique
        if hasattr(self, 'index_to_display'):
            del self.index_to_display

        # for StaticImages
        if hasattr(self, 'images_wrapped'):
            del self.images_wrapped
        if hasattr(self, 'images_dewrapped'):
            del self.images_dewrapped
        if hasattr(self, 'altitude_wrapped'):
            del self.altitude_wrapped
        if hasattr(self, 'azimuth_wrapped'):
            del self.azimuth_wrapped
        if hasattr(self, 'altitude_dewrapped'):
            del self.altitude_dewrapped
        if hasattr(self, 'azimuth_dewrapped'):
            del self.azimuth_dewrapped

    def set_monitor(self, monitor):
        self.monitor = monitor
        self.clear()

    def set_indicator(self, indicator):
        self.indicator = indicator
        self.clear()

    def set_pregap_dur(self, pregap_dur):
        if pregap_dur >= 0.:
            self.pregap_dur = float(pregap_dur)
        else:
            raise ValueError('pregap_dur should be no less than 0.')
        self.clear()

    def set_postgap_dur(self, postgap_dur):
        if postgap_dur >= 0.:
            self.postgap_dur = float(postgap_dur)
        else:
            raise ValueError('postgap_dur should be no less than 0.')

    def set_background(self, background):
        if background < -1. or background > 1.:
            raise ValueError('parameter "background" should be a float within [-1., 1.].')
        else:
            self.background = float(background)
        self.clear()

    def set_coordinate(self, coordinate):
        if coordinate not in ['degree', 'linear']:
            raise ValueError('parameter "coordinate" should be either "degree" or "linear".')
        self.coordinate = coordinate

class SparseNoise(Stim):
    """
    generate sparse noise stimulus integrates flashing indicator for photodiode

    This stimulus routine presents quasi-random noise in a specified region of
    the monitor. The `background` color can be customized but defaults to a
    grey value. Can specify the `subregion` of the monitor where the pixels
    will flash on and off (black and white respectively)

    Parameters
    ----------
    monitor : monitor object
        contains display monitor information
    indicator : indicator object
        contains indicator information
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    grid_space : 2-tuple of floats, optional
        first coordinate is altitude, second coordinate is azimuth
    probe_size : 2-tuple of floats, optional
        size of flicker probes. First coordinate defines the width, and
        second coordinate defines the height
    probe_orientation : float, optional
        orientation of flicker probes
    probe_frame_num : int, optional
        number of frames for each square presentation
    subregion : list or tuple
        the region on the monitor that will display the sparse noise,
        list or tuple, [min_alt, max_alt, min_azi, max_azi]
    sign : {'ON-OFF', 'ON', 'OFF'}, optional
        determines which pixels appear in the `subregion`, defaults to
        `'ON-Off'` so that both on and off pixels appear. If `'ON` selected
        only on pixels (white) are displayed in the noise `subregion while if
        `'OFF'` is selected only off (black) pixels are displayed in the noise
    iteration : int, optional
        number of times to present stimulus, defaults to `1`
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    """

    def __init__(self, monitor, indicator, background=0., coordinate='degree',
                 grid_space=(10., 10.), probe_size=(10., 10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):
        """
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        super(SparseNoise, self).__init__(monitor=monitor,
                                          indicator=indicator,
                                          background=background,
                                          coordinate=coordinate,
                                          pregap_dur=pregap_dur,
                                          postgap_dur=postgap_dur)

        self.stim_name = 'SparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.probe_orientation = probe_orientation

        if probe_frame_num >= 2.:
            self.probe_frame_num = int(probe_frame_num)
        else:
            raise ValueError('SparseNoise: probe_frame_num should be no less than 2.')

        self.is_include_edge = is_include_edge
        self.frame_config = ('is_display', 'probe center (altitude, azimuth)',
                             'polarity (-1 or 1)', 'indicator color [-1., 1.]')

        if subregion is None:
            if self.coordinate == 'degree':
                self.subregion = [np.amin(self.monitor.deg_coord_y),
                                  np.amax(self.monitor.deg_coord_y),
                                  np.amin(self.monitor.deg_coord_x),
                                  np.amax(self.monitor.deg_coord_x)]
            if self.coordinate == 'linear':
                self.subregion = [np.amin(self.monitor.lin_coord_y),
                                  np.amax(self.monitor.lin_coord_y),
                                  np.amin(self.monitor.lin_coord_x),
                                  np.amax(self.monitor.lin_coord_x)]
        else:
            self.subregion = subregion

        self.sign = sign
        if iteration >= 1:
            self.iteration = int(iteration)
        else:
            raise ValueError('iteration should be no less than 1.')

        self.clear()

    def _get_grid_locations(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array,
            refined [alt, azi] pairs of probe centers going to be displayed
        """

        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_azi = self.monitor.deg_coord_x
            monitor_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_azi = self.monitor.lin_coord_x
            monitor_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        grid_locations = get_grid_locations(subregion=self.subregion, grid_space=self.grid_space,
                                            monitor_azi=monitor_azi, monitor_alt=monitor_alt,
                                            is_include_edge=self.is_include_edge, is_plot=is_plot)

        return grid_locations

    def _generate_grid_points_sequence(self):
        """
        generate pseudorandomized grid point sequence. if ON-OFF, consecutive
        frames should not present stimulus at same location

        Returns
        -------
        all_grid_points : list
            list of the form [grid_point, sign]
        """

        grid_points = self._get_grid_locations()

        if self.sign == 'ON':
            grid_points = [[x, 1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'OFF':
            grid_points = [[x, -1] for x in grid_points]
            random.shuffle(grid_points)
            return grid_points
        elif self.sign == 'ON-OFF':
            all_grid_points = [[x, 1] for x in grid_points] + [[x, -1] for x in grid_points]
            random.shuffle(all_grid_points)
            # remove coincident hit of same location by continuous frames
            print('removing coincident hit of same location with continuous frames:')
            while True:
                iteration = 0
                coincident_hit_num = 0
                for i, grid_point in enumerate(all_grid_points[:-3]):
                    if (all_grid_points[i][0] == all_grid_points[i + 1][0]).all():
                        all_grid_points[i + 1], all_grid_points[i + 2] = all_grid_points[i + 2], all_grid_points[i + 1]
                        coincident_hit_num += 1
                iteration += 1
                print('iteration:' + iteration + '  continous hits number:' + coincident_hit_num)
                if coincident_hit_num == 0:
                    break

            return all_grid_points

    def generate_frames(self):
        """
        function to generate all the frames needed for SparseNoise stimulus

        returns a list of information of all frames as a list of tuples

        Information contained in each frame:
             first element - int
                  when stimulus is displayed value is equal to 1, otherwise
                  equal to 0,
             second element - tuple,
                  retinotopic location of the center of current square,[alt, azi]
             third element -
                  polarity of current square, 1 -> bright, -1-> dark
             forth element - color of indicator
                  if synchronized : value equal to 0 when stimulus is not
                       begin displayed, and 1 for onset frame of stimulus for
                       each square, -1 for the rest.
                  if non-synchronized: values alternate between -1 and 1
                       at defined frequency

             for gap frames the second and third elements should be 'None'
        """

        frames = []
        if self.probe_frame_num == 1:
            indicator_on_frame = 1
        elif self.probe_frame_num > 1:
            indicator_on_frame = self.probe_frame_num // 2
        else:
            raise ValueError('`probe_frame_num` should be an int larger than 0!')

        indicator_off_frame = self.probe_frame_num - indicator_on_frame

        frames += [[0., None, None, -1.]] * self.pregap_frame_num

        for i in range(self.iteration):

            iter_grid_points = self._generate_grid_points_sequence()

            for grid_point in iter_grid_points:
                frames += [[1., grid_point[0], grid_point[1], 1.]] * indicator_on_frame
                frames += [[1., grid_point[0], grid_point[1], -1.]] * indicator_off_frame

        frames += [[0., None, None, -1.]] * self.postgap_frame_num

        if not self.indicator.is_sync:
            indicator_frame = self.indicator.frame_num
            for m in range(len(frames)):
                if np.floor(m // indicator_frame) % 2 == 0:
                    frames[m][3] = 1.
                else:
                    frames[m][3] = -1.

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def _generate_frames_for_index_display(self):
        """ compute the information that defines the frames used for index display"""
        if self.indicator.is_sync:
            frames_unique = []

            gap = [0., None, None, -1.]
            frames_unique.append(gap)
            grid_points = self._get_grid_locations()
            for grid_point in grid_points:
                if self.sign == 'ON':
                    frames_unique.append([1., grid_point, 1., 1.])
                    frames_unique.append([1., grid_point, 1., -1.])
                elif self.sign == 'OFF':
                    frames_unique.append([1., grid_point, -1., 1.])
                    frames_unique.append([1., grid_point, -1., -1])
                elif self.sign == 'ON-OFF':
                    frames_unique.append([1., grid_point, 1., 1.])
                    frames_unique.append([1., grid_point, 1., -1.])
                    frames_unique.append([1., grid_point, -1., 1.])
                    frames_unique.append([1., grid_point, -1., -1])
                else:
                    raise ValueError('SparseNoise: Do not understand "sign", should '
                                     'be one of "ON", "OFF" and "ON-OFF".')

            frames_unique = tuple([tuple(f) for f in frames_unique])

            return frames_unique
        else:
            raise NotImplementedError("method not available for non-sync indicator")

    @staticmethod
    def _get_probe_index_for_one_iter_on_off(frames_unique):
        """
        get shuffled probe indices from frames_unique generated by
        self._generate_frames_for_index_display(), only for 'ON-OFF' stimulus

        the first element of frames_unique should be gap frame, the following
        frames should be [
                          (probe_i_ON, indictor_ON),
                          (probe_i_ON, indictor_OFF),
                          (probe_i_OFF, indictor_ON),
                          (probe_i_OFF, indictor_OFF),
                          ]

        it is designed such that no consecutive probes will hit the same visual
        field location

        return list of integers, indices of shuffled probe
        """

        if len(frames_unique) % 4 == 1:
            probe_num = (len(frames_unique) - 1) // 2
        else:
            raise ValueError('number of frames_unique should be 4x + 1')

        probe_locations = [f[1] for f in frames_unique[1::2]]
        probe_ind = np.arange(probe_num)
        np.random.shuffle(probe_ind)

        is_overlap = True
        while is_overlap:
            is_overlap = False
            for i in range(probe_num - 1):
                probe_loc_0 = probe_locations[probe_ind[i]]
                probe_loc_1 = probe_locations[probe_ind[i + 1]]
                if np.array_equal(probe_loc_0, probe_loc_1):
                    # print('overlapping probes detected. ind_{}:loc{}; ind_{}:loc{}'
                    #       .format(i, probe_loc_0, i + 1, probe_loc_1))
                    # print ('ind_{}:loc{}'.format((i + 2) % probe_num,
                    #                              probe_locations[(i + 2) % probe_num]))
                    ind_temp = probe_ind[i + 1]
                    probe_ind[i + 1] = probe_ind[(i + 2) % probe_num]
                    probe_ind[(i + 2) % probe_num] = ind_temp
                    is_overlap = True

        return probe_ind

    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """

        frames_unique = self._generate_frames_for_index_display()
        probe_on_frame_num = self.probe_frame_num // 2
        probe_off_frame_num = self.probe_frame_num - probe_on_frame_num

        if self.sign == 'ON' or self.sign == 'OFF':

            if len(frames_unique) % 2 == 1:
                probe_num = (len(frames_unique) - 1) / 2
            else:
                raise ValueError('SparseNoise: number of unique frames is not correct. Should be odd.')

            index_to_display = []

            index_to_display += [0] * self.pregap_frame_num

            for iter in range(self.iteration):

                probe_sequence = np.arange(probe_num)
                np.random.shuffle(probe_sequence)

                for probe_ind in probe_sequence:
                    index_to_display += [probe_ind * 2 + 1] * probe_on_frame_num
                    index_to_display += [probe_ind * 2 + 2] * probe_off_frame_num

            index_to_display += [0] * self.postgap_frame_num

        elif self.sign == 'ON-OFF':
            if len(frames_unique) % 4 != 1:
                raise ValueError('number of frames_unique should be 4x + 1')

            index_to_display = []

            index_to_display += [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                probe_inds = self._get_probe_index_for_one_iter_on_off(frames_unique)

                for probe_ind in probe_inds:
                    index_to_display += [probe_ind * 2 + 1] * probe_on_frame_num
                    index_to_display += [probe_ind * 2 + 2] * probe_off_frame_num

            index_to_display += [0] * self.postgap_frame_num

        else:
            raise ValueError('SparseNoise: Do not understand "sign", should '
                             'be one of "ON", "OFF" and "ON-OFF".')

        return frames_unique, index_to_display

    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """
        self.frames_unique, self.index_to_display = self._generate_display_index()

        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        full_seq = self.background * \
                   np.ones((num_unique_frames, num_pixels_width, num_pixels_height), dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
                curr_probes = ([frame[1][0], frame[1][1], frame[2]],)
                # print type(curr_probes)
                disp_mat = get_warped_probes(deg_coord_alt=coord_alt,
                                             deg_coord_azi=coord_azi,
                                             probes=curr_probes,
                                             width=self.probe_size[0],
                                             height=self.probe_size[1],
                                             ori=self.probe_orientation,
                                             background_color=self.background)

                full_seq[i] = disp_mat

            full_seq[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[3]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        SNdict = dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        full_dict = {'stimulation': SNdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict

    def generate_movie(self):
        """
        generate movie for display frame by frame
        """

        self.frames = self.generate_frames()

        if self.coordinate == 'degree':
            coord_x = self.monitor.deg_coord_x
            coord_y = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_x = self.monitor.lin_coord_x
            coord_y = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        indicator_width_min = (self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = (self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = (self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = (self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        full_seq = np.ones((len(self.frames),
                            self.monitor.deg_coord_x.shape[0],
                            self.monitor.deg_coord_x.shape[1]),
                           dtype=np.float32) * self.background

        for i, curr_frame in enumerate(self.frames):
            if curr_frame[0] == 1:  # not a gap

                curr_probes = ([curr_frame[1][0], curr_frame[1][1], curr_frame[2]],)

                if i == 0:  # first frame and (not a gap)
                    curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                      deg_coord_azi=coord_x,
                                                      probes=curr_probes,
                                                      width=self.probe_size[0],
                                                      height=self.probe_size[1],
                                                      ori=self.probe_orientation,
                                                      background_color=self.background)
                else:  # (not first frame) and (not a gap)
                    if self.frames[i - 1][1] is None:  # (not first frame) and (not a gap) and (new square from gap)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          background_color=self.background)
                    elif (curr_frame[1] != self.frames[i - 1][1]).any() or (curr_frame[2] != self.frames[i - 1][2]):
                        # (not first frame) and (not a gap) and (new square from old square)
                        curr_disp_mat = get_warped_probes(deg_coord_alt=coord_y,
                                                          deg_coord_azi=coord_x,
                                                          probes=curr_probes,
                                                          width=self.probe_size[0],
                                                          height=self.probe_size[1],
                                                          ori=self.probe_orientation,
                                                          background_color=self.background)

                # assign current display matrix to full sequence
                full_seq[i] = curr_disp_mat

            # add sync square for photodiode
            full_seq[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = curr_frame[3]

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: ' +
                       str(int(100 * (i + 1) / len(self.frames))) + '%')

        # generate log dictionary
        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        SNdict = dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        full_dict = {'stimulation': SNdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict


class LocallySparseNoise(Stim):
    """
    generate locally sparse noise stimulus integrates flashing indicator for
    photodiode

    This stimulus routine presents quasi-random noise in a specified region of
    the monitor. The `background` color can be customized but defaults to a
    grey value. Can specify the `subregion` of the monitor where the pixels
    will flash on and off (black and white respectively)

    Different from SparseNoise stimulus which presents only one probe at a time,
    the LocallySparseNoise presents multiple probes simultaneously to speed up
    the sampling frequency. The sparsity of probes is defined by minimum distance
    in visual degree: in any given frame, the centers of any pair of two probes
    will have distance larger than minimum distance in visual degrees. The
    method generate locally sparse noise here insures, for each iteration, all
    the locations in the subregion will be sampled once and only once.

    Parameters
    ----------
    monitor : monitor object
        contains display monitor information
    indicator : indicator object
        contains indicator information
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    min_distance : float, default 20.
        the minimum distance in visual degree for any pair of probe centers
        in a given frame
    grid_space : 2-tuple of floats, optional
        first coordinate is altitude, second coordinate is azimuth
    probe_size : 2-tuple of floats, optional
        size of flicker probes. First coordinate defines the width, and
        second coordinate defines the height
    probe_orientation : float, optional
        orientation of flicker probes
    probe_frame_num : int, optional
        number of frames for each square presentation
    subregion : list or tuple
        the region on the monitor that will display the sparse noise,
        list or tuple, [min_alt, max_alt, min_azi, max_azi]
    sign : {'ON-OFF', 'ON', 'OFF'}, optional
        determines which pixels appear in the `subregion`, defaults to
        `'ON-Off'` so that both on and off pixels appear. If `'ON` selected
        only on pixels (white) are displayed in the noise `subregion while if
        `'OFF'` is selected only off (black) pixels are displayed in the noise
    iteration : int, optional
        number of times to present stimulus with random order, the total number
        a paticular probe will be displayded will be iteration * repeat,
        defaults to `1`
    repeat : int, optional
        number of repeat of whole sequence, the total number a paticular probe
        will be displayded will be iteration * repeat, defaults to `1`
    is_include_edge : bool, default True,
        if True, the displayed probes will cover the edge case and ensure that
        the entire subregion is covered.
        If False, the displayed probes will exclude edge case and ensure that all
        the centers of displayed probes are within the subregion.
    """

    def __init__(self, monitor, indicator, min_distance=20., background=0., coordinate='degree',
                 grid_space=(10., 10.), probe_size=(10., 10.), probe_orientation=0.,
                 probe_frame_num=6, subregion=None, sign='ON-OFF', iteration=1, repeat=1,
                 pregap_dur=2., postgap_dur=3., is_include_edge=True):
        """
        Initialize sparse noise object, inherits Parameters from Stim object
        """

        super(LocallySparseNoise, self).__init__(monitor=monitor, indicator=indicator,
                                                 background=background, coordinate=coordinate,
                                                 pregap_dur=pregap_dur, postgap_dur=postgap_dur)

        self.stim_name = 'LocallySparseNoise'
        self.grid_space = grid_space
        self.probe_size = probe_size
        self.min_distance = float(min_distance)
        self.probe_orientation = probe_orientation

        self.is_include_edge = is_include_edge
        self.frame_config = ('is_display', 'probes ((altitude, azimuth, sign), ...)',
                             'iteration', 'indicator color [-1., 1.]')

        if probe_frame_num >= 2:
            self.probe_frame_num = int(probe_frame_num)
        else:
            raise ValueError('SparseNoise: probe_frame_num should be no less than 2.')

        self.is_include_edge = is_include_edge

        if subregion is None:
            if self.coordinate == 'degree':
                self.subregion = [np.amin(self.monitor.deg_coord_y),
                                  np.amax(self.monitor.deg_coord_y),
                                  np.amin(self.monitor.deg_coord_x),
                                  np.amax(self.monitor.deg_coord_x)]
            if self.coordinate == 'linear':
                self.subregion = [np.amin(self.monitor.lin_coord_y),
                                  np.amax(self.monitor.lin_coord_y),
                                  np.amin(self.monitor.lin_coord_x),
                                  np.amax(self.monitor.lin_coord_x)]
        else:
            self.subregion = subregion

        self.sign = sign

        if iteration >= 1:
            self.iteration = int(iteration)
        else:
            raise ValueError('iteration should be no less than 1.')

        if repeat >= 1:
            self.repeat = int(repeat)
        else:
            raise ValueError('repeat should be no less than 1.')

        self.clear()

    def _get_grid_locations(self, is_plot=False):
        """
        generate all the grid points in display area (covered by both subregion and
        monitor span)

        Returns
        -------
        grid_points : n x 2 array,
            refined [azi, alt] pairs of probe centers going to be displayed
        """

        # get all the visual points for each pixels on monitor
        if self.coordinate == 'degree':
            monitor_azi = self.monitor.deg_coord_x
            monitor_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            monitor_azi = self.monitor.lin_coord_x
            monitor_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. Should be either "linear" or "degree".'.
                             format(self.coordinate))

        grid_locations = get_grid_locations(subregion=self.subregion, grid_space=self.grid_space,
                                            monitor_azi=monitor_azi, monitor_alt=monitor_alt,
                                            is_include_edge=self.is_include_edge, is_plot=is_plot)

        return grid_locations

    def _generate_all_probes(self):
        """
        return all possible (grid location + sign) combinations within the subregion,
        return a list of probe parameters, each element in the list is
        [center_altitude, center_azimuth, sign]
        """
        grid_locs = self._get_grid_locations()

        grid_locs = list([list(gl) for gl in grid_locs])

        if self.sign == 'ON':
            all_probes = [gl + [1.] for gl in grid_locs]
        elif self.sign == 'OFF':
            all_probes = [gl + [-1.] for gl in grid_locs]
        elif self.sign == 'ON-OFF':
            all_probes = [gl + [1.] for gl in grid_locs] + [gl + [-1.] for gl in grid_locs]
        else:
            raise ValueError('LocallySparseNoise: Cannot understand self.sign, should be '
                             'one of "ON", "OFF", "ON-OFF".')
        return all_probes

    def _generate_probe_locs_one_frame(self, probes):
        """
        given the available probes, generate a sublist of the probes for a single frame,
        all the probes in the sublist will have their visual space distance longer than
        self.min_distance. This function will also update input probes, remove the
        elements that have been selected into the sublist.

        parameters
        ----------
        probes : list of all available probes
            each elements is [center_altitude, center_azimuth, sign] for a particular probe
        min_dis : float
            minimum distance to reject probes too close to each other

        returns
        -------
        probes_one_frame : list of selected probes fo one frame
            each elements is [center_altitude, center_azimuth, sign] for a selected probe
        """

        np.random.shuffle(probes)
        probes_one_frame = []

        probes_left = list(probes)

        for probe in probes:

            # print len(probes)

            is_overlap = False

            for probe_frame in probes_one_frame:
                # print probe
                # print probe_frame
                curr_dis = ia.distance([probe[0], probe[1]], [probe_frame[0], probe_frame[1]])
                if curr_dis <= self.min_distance:
                    is_overlap = True
                    break

            if not is_overlap:
                probes_one_frame.append(probe)
                probes_left.remove(probe)

        return probes_one_frame, probes_left

    def _generate_probe_sequence_one_iteration(self, all_probes, is_redistribute=True):
        """
        given all probes to be displayed and minimum distance between any pair of two probes
        return frames of one iteration that ensure all probes will be present once

        parameters
        ----------
        all_probes : list
            all probes to be displayed, each element (center_alt, center_azi, sign). ideally
            outputs of self._generate_all_probes()
        is_redistribute : bool
            redistribute the probes among frames after initial generation or not.
            redistribute will use self._redistribute_probes() and try to minimize the difference
            of probe numbers among different frames

        returns
        -------
        frames : tuple
            each element of the frames tuple represent one display frame, the element itself
            is a tuple of the probes to be displayed in this particular frame
        """

        all_probes_cpy = list(all_probes)

        frames = []

        while len(all_probes_cpy) > 0:
            curr_frames, all_probes_cpy = self._generate_probe_locs_one_frame(probes=all_probes_cpy)
            frames.append(curr_frames)

        if is_redistribute:
            frames = self._redistribute_probes(frames=frames)

        frames = tuple(tuple(f) for f in frames)

        return frames

    def _redistribute_one_probe(self, frames):

        # initiate is_moved variable
        is_moved = False

        # reorder frames from most probes to least probes
        new_frames = sorted(frames, key=lambda frame: len(frame))
        probe_num_most = len(new_frames[-1])

        # the indices of frames in new_frames that contain most probes
        frame_ind_most = []

        # the indices of frames in new_frames that contain less probes
        frame_ind_less = []

        for frame_ind, frame in enumerate(new_frames):
            if len(frame) == probe_num_most:
                frame_ind_most.append(frame_ind)
            elif len(frame) <= probe_num_most - 2:  # '-1' means well distributed
                frame_ind_less.append(frame_ind)

        # constructing a list of probes that potentially can be moved
        # each element is [(center_alt, center_azi, sign), frame_ind]
        probes_to_be_moved = []
        for frame_ind in frame_ind_most:
            frame_most = new_frames[frame_ind]
            for probe in frame_most:
                probes_to_be_moved.append((probe, frame_ind))

        # loop through probes_to_be_moved to see if any of them will fit into
        # frames with less probes, once find a case, break the loop and return
        for probe, frame_ind_src in probes_to_be_moved:
            frame_src = new_frames[frame_ind_src]
            for frame_ind_dst in frame_ind_less:
                frame_dst = new_frames[frame_ind_dst]
                if self._is_fit(probe, frame_dst):
                    frame_src.remove(probe)
                    frame_dst.append(probe)
                    is_moved = True
                    break
            if is_moved:
                break

        return is_moved, new_frames

    def _is_fit(self, probe, probes):
        """
        test if a given probe will fit a group of probes without breaking the
        sparcity

        parameters
        ----------
        probe : list or tuple of three floats
            (center_alt, center_azi, sign)
        probes : list of probes
            [(center_alt, center_zai, sign), (center_alt, center_azi, sign), ...]

        returns
        -------
        is_fit : bool
            the probe will fit or not
        """

        is_fit = True
        for probe2 in probes:
            if ia.distance([probe[0], probe[1]], [probe2[0], probe2[1]]) <= self.min_distance:
                is_fit = False
                break
        return is_fit

    def _redistribute_probes(self, frames):
        """
        attempt to redistribute probes among frames for one iteration of display
        the algorithm is to pick a probe from the frames with most probes to the
        frames with least probes and do it iteratively until it can not move
        anymore and the biggest difference of probe numbers among all frames is
        no more than 1 (most evenly distributed).

        the algorithm is implemented by self._redistribute_probes() function,
        this is just to roughly massage the probes among frames, but not the
        attempt to find the best solution.

        parameters
        ----------
        frames : list
            each element of the frames list represent one display frame, the element
            itself is a list of the probes (center_alt, center_azi, sign) to be
            displayed in this particular frame

        returns
        -------
        new_frames : list
            same structure as input frames but with redistributed probes
        """

        new_frames = list(frames)
        is_moved = True
        probe_nums = [len(frame) for frame in new_frames]
        probe_nums.sort()
        probe_diff = probe_nums[-1] - probe_nums[0]

        while is_moved and probe_diff > 1:

            is_moved, new_frames = self._redistribute_one_probe(new_frames)
            probe_nums = [len(frame) for frame in new_frames]
            probe_nums.sort()
            probe_diff = probe_nums[-1] - probe_nums[0]
        else:
            if not is_moved:
                # print ('redistributing probes among frames: no more probes can be moved.')
                pass
            if probe_diff <= 1:
                # print ('redistributing probes among frames: probes already well distributed.')
                pass

        return new_frames

    def _generate_frames_for_index_display(self):
        """
        compute the information that defines the frames used for index display

        parameters
        ----------
        all_probes : list
            all probes to be displayed, each element (center_alt, center_azi, sign). ideally
            outputs of self._generate_all_probes()

        returns
        -------
        frames_unique : tuple
        """
        all_probes = self._generate_all_probes()

        frames_unique = []

        gap = [0., None, None, -1.]
        frames_unique.append(gap)
        for i in range(self.iteration):
            probes_iter = self._generate_probe_sequence_one_iteration(all_probes=all_probes,
                                                                      is_redistribute=True)
            for probes in probes_iter:
                frames_unique.append([1., probes, i, 1.])
                frames_unique.append([1., probes, i, -1.])

        frames_unique = tuple([tuple(f) for f in frames_unique])

        return frames_unique

    def _generate_display_index(self):
        """
        compute a list of indices corresponding to each frame to display.
        """

        if self.indicator.is_sync:

            frames_unique = self._generate_frames_for_index_display()
            if len(frames_unique) % 2 == 1:
                display_num = (len(frames_unique) - 1) / 2  # number of each unique display frame
            else:
                raise ValueError('LocallySparseNoise: number of unique frames is not correct. Should be odd.')

            probe_on_frame_num = self.probe_frame_num // 2
            probe_off_frame_num = self.probe_frame_num - probe_on_frame_num

            index_to_display = []

            for display_ind in np.arange(display_num):
                index_to_display += [display_ind * 2 + 1] * probe_on_frame_num
                index_to_display += [display_ind * 2 + 2] * probe_off_frame_num

            index_to_display = index_to_display * self.repeat

            index_to_display = [0] * self.pregap_frame_num + index_to_display + [0] * self.postgap_frame_num

            return frames_unique, index_to_display

        else:
            raise NotImplementedError("method not available for non-sync indicator")

    def generate_movie_by_index(self):

        self.frames_unique, self.index_to_display = self._generate_display_index()

        num_unique_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        if self.coordinate == 'degree':
            coord_azi = self.monitor.deg_coord_x
            coord_alt = self.monitor.deg_coord_y
        elif self.coordinate == 'linear':
            coord_azi = self.monitor.lin_coord_x
            coord_alt = self.monitor.lin_coord_y
        else:
            raise ValueError('Do not understand coordinate system: {}. '
                             'Should be either "linear" or "degree".'.
                             format(self.coordinate))

        indicator_width_min = int(self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = int(self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = int(self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = int(self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        full_seq = self.background * \
                   np.ones((num_unique_frames, num_pixels_width, num_pixels_height), dtype=np.float32)

        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1.:
                disp_mat = get_warped_probes(deg_coord_alt=coord_alt,
                                             deg_coord_azi=coord_azi,
                                             probes=frame[1],
                                             width=self.probe_size[0],
                                             height=self.probe_size[1],
                                             ori=self.probe_orientation,
                                             background_color=self.background)

                full_seq[i] = disp_mat

            full_seq[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[3]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        SNdict = dict(self.__dict__)
        SNdict.pop('monitor')
        SNdict.pop('indicator')
        full_dict = {'stimulation': SNdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict


class FlashingCircle(Stim):
    """
    Generate flashing circle stimulus.
    Stimulus routine presents a circle centered at the position `center`
    with given `radius`.
    Parameters
    ----------
    monitor : monitor object
        contains display monitor information
    indicator : indicator object
        contains indicator information
    coordinate : str from {'degree','linear'}, optional
        specifies coordinates, defaults to 'degree'
    background : float, optional
        color of background. Takes values in [-1,1] where -1 is black and 1
        is white
    pregap_dur : float, optional
        amount of time (in seconds) before the stimulus is presented, defaults
        to `2.`
    postgap_dur : float, optional
        amount of time (in seconds) after the stimulus is presented, defaults
        to `3.`
    center : 2-tuple, optional
        center coordinate (altitude, azimuth) of the circle in degrees, defaults to (0.,60.).
    radius : float, optional
        radius of the circle, defaults to `10.`
    is_smooth_edge : bool
        True, smooth circle edge with smooth_width_ratio and smooth_func
        False, do not smooth edge
    smooth_width_ratio : float, should be smaller than 1.
        the ratio between smooth band width and radius, circle edge is the middle
        of smooth band
    smooth_func : function object
        this function take to inputs
            first, ndarray storing the distance from each pixel to smooth band center
            second, smooth band width
        returns smoothed mask with same shape as input ndarray
    color : float, optional
        color of the circle, takes values in [-1,1], defaults to `-1.`
    iteration : int, optional
        total number of flashes, defaults to `1`.
    flash_frame : int, optional
        number of frames that circle is displayed during each presentation
        of the stimulus, defaults to `3`.
    """

    def __init__(self, monitor, indicator, coordinate='degree', center=(0., 60.),
                 radius=10., is_smooth_edge=False, smooth_width_ratio=0.2,
                 smooth_func=blur_cos, color=-1., flash_frame_num=3,
                 pregap_dur=2., postgap_dur=3., background=0., midgap_dur=1.,
                 iteration=1):

        """
        Initialize `FlashingCircle` stimulus object.
        """

        super(FlashingCircle, self).__init__(monitor=monitor,
                                             indicator=indicator,
                                             background=background,
                                             coordinate=coordinate,
                                             pregap_dur=pregap_dur,
                                             postgap_dur=postgap_dur)

        self.stim_name = 'FlashingCircle'
        self.center = center
        self.radius = float(radius)
        self.color = float(color)
        self.flash_frame_num = int(flash_frame_num)
        self.frame_config = ('is_display', 'indicator color [-1., 1.]')
        self.is_smooth_edge = is_smooth_edge
        self.smooth_width_ratio = float(smooth_width_ratio)
        self.smooth_func = smooth_func
        self.midgap_dur = float(midgap_dur)
        self.iteration = int(iteration)

        if self.pregap_frame_num + self.postgap_frame_num == 0:
            raise ValueError('pregap_frame_num + postgap_frame_num should be larger than 0.')

        self.clear()

    def set_flash_frame_num(self, flash_frame_num):
        self.flash_frame_num = flash_frame_num
        self.clear()

    def set_color(self, color):
        self.color = color
        self.clear()

    def set_center(self, center):
        self.center = center
        self.clear()

    def set_radius(self, radius):
        self.radius = radius
        self.clear()

    @property
    def midgap_frame_num(self):
        return int(self.midgap_dur * self.monitor.refresh_rate)

    def generate_frames(self):
        """
        function to generate all the frames needed for the stimulation.
        Information contained in each frame:
           first element :
                during a gap, the value is equal to 0 and during display the
                value is equal to 1
           second element :
                corresponds to the color of indicator
                if indicator.is_sync is True, during stimulus the value is
                equal to 1., whereas during a gap the value isequal to -1.;
                if indicator.is_sync is False, indicator color will alternate
                between 1. and -1. at the frequency as indicator.freq
        Returns
        -------
        frames : list
            list of information defining each frame.
        """

        frames = [[0, -1.]] * self.pregap_frame_num

        for iter in range(self.iteration):

            if self.indicator.is_sync:
                frames += [[0, -1.]] * self.midgap_frame_num
                frames += [[1, 1.]] * self.flash_frame_num
            else:
                frames += [[0, -1.]] * self.midgap_frame_num
                frames += [[1, -1.]] * self.flash_frame_num

        frames += [[0, -1.]] * self.postgap_frame_num

        frames = frames[self.midgap_frame_num:]

        if not self.indicator.is_sync:
            for frame_ind in xrange(frames.shape[0]):
                # mark unsynchronized indicator
                if np.floor(frame_ind // self.indicator.frame_num) % 2 == 0:
                    frames[frame_ind, 1] = 1.
                else:
                    frames[frame_ind, 1] = -1.

        frames = [tuple(x) for x in frames]

        return tuple(frames)

    def _generate_frames_for_index_display(self):
        """
        frame structure: first element: is_gap (0:False; 1:True).
                         second element: indicator color [-1., 1.]
        """
        if self.indicator.is_sync:
            gap = (0., -1.)
            flash = (1., 1.)
            frames = (gap, flash)
            return frames
        else:
            raise NotImplemente("method not available for non-sync indicator")

    def _generate_display_index(self):
        """ compute a list of indices corresponding to each frame to display. """
        if self.indicator.is_sync:

            index_to_display = [0] * self.pregap_frame_num

            for iter in range(self.iteration):
                index_to_display += [0] * self.midgap_frame_num
                index_to_display += [1] * self.flash_frame_num

            index_to_display += [0] * self.postgap_frame_num
            index_to_display = index_to_display[self.midgap_frame_num:]

            return index_to_display
        else:
            raise NotImplementedError("method not available for non-sync indicator")

    def generate_movie_by_index(self):
        """ compute the stimulus movie to be displayed by index. """

        # compute unique frame parameters
        self.frames_unique = self._generate_frames_for_index_display()
        self.index_to_display = self._generate_display_index()

        num_frames = len(self.frames_unique)
        num_pixels_width = self.monitor.deg_coord_x.shape[0]
        num_pixels_height = self.monitor.deg_coord_x.shape[1]

        full_sequence = self.background * np.ones((num_frames,
                                                   num_pixels_width,
                                                   num_pixels_height),
                                                   dtype=np.float32)

        indicator_width_min = int(self.indicator.center_width_pixel
                               - self.indicator.width_pixel / 2)
        indicator_width_max = int(self.indicator.center_width_pixel
                               + self.indicator.width_pixel / 2)
        indicator_height_min = int(self.indicator.center_height_pixel
                                - self.indicator.height_pixel / 2)
        indicator_height_max = int(self.indicator.center_height_pixel
                                + self.indicator.height_pixel / 2)

        # background = self.background * np.ones((num_pixels_width,
        #                                         num_pixels_height),
        #                                        dtype=np.float32)

        if self.coordinate == 'degree':
            map_azi = self.monitor.deg_coord_x
            map_alt = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_azi = self.monitor.lin_coord_x
            map_alt = self.monitor.lin_coord_y
        else:
            raise LookupError("`coordinate` not in {'linear','degree'}")

        circle_mask = get_circle_mask(map_alt=map_alt, map_azi=map_azi,
                                      center=self.center, radius=self.radius,
                                      is_smooth_edge=self.is_smooth_edge,
                                      blur_ratio=self.smooth_width_ratio,
                                      blur_func=self.smooth_func).astype(np.float32)
        # plt.imshow(circle_mask)
        # plt.show()
        # print(indicator_width_min)
        # print(indicator_width_max)
        for i, frame in enumerate(self.frames_unique):
            if frame[0] == 1:
                full_sequence[i][np.where(circle_mask==1)] = self.color

            full_sequence[i, indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = frame[1]

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        NFdict.pop('smooth_func')
        full_dict = {'stimulation': NFdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_sequence, full_dict

    def generate_movie(self):
        """
        generate movie frame by frame.
        """

        self.frames = self.generate_frames()

        full_seq = np.zeros((len(self.frames), self.monitor.deg_coord_x.shape[0],
                             self.monitor.deg_coord_x.shape[1]),
                            dtype=np.float32)

        indicator_width_min = (self.indicator.center_width_pixel -
                               (self.indicator.width_pixel / 2))
        indicator_width_max = (self.indicator.center_width_pixel +
                               (self.indicator.width_pixel / 2))
        indicator_height_min = (self.indicator.center_height_pixel -
                                (self.indicator.height_pixel / 2))
        indicator_height_max = (self.indicator.center_height_pixel +
                                (self.indicator.height_pixel / 2))

        background = np.ones((np.size(self.monitor.deg_coord_x, 0),
                              np.size(self.monitor.deg_coord_x, 1)),
                             dtype=np.float32) * self.background

        if self.coordinate == 'degree':
            map_azi = self.monitor.deg_coord_x
            map_alt = self.monitor.deg_coord_y

        elif self.coordinate == 'linear':
            map_azi = self.monitor.lin_coord_x
            map_alt = self.monitor.lin_coord_y
        else:
            raise LookupError("`coordinate` not in {'linear','degree'}")

        circle_mask = get_circle_mask(map_alt=map_alt, map_azi=map_azi,
                                      center=self.center, radius=self.radius,
                                      is_smooth_edge=self.is_smooth_edge,
                                      blur_ratio=self.smooth_width_ratio,
                                      blur_func=self.smooth_func).astype(np.float32)

        for i in range(len(self.frames)):
            curr_frame = self.frames[i]

            if curr_frame[0] == 0:
                curr_FC_seq = background
            else:
                curr_FC_seq = ((circle_mask * self.color) +
                               ((-1 * (circle_mask - 1)) * background))

            curr_FC_seq[indicator_height_min:indicator_height_max,
            indicator_width_min:indicator_width_max] = curr_frame[1]

            full_seq[i] = curr_FC_seq

            if i in range(0, len(self.frames), len(self.frames) / 10):
                print('Generating numpy sequence: '
                       + str(int(100 * (i + 1) / len(self.frames))) + '%')

        mondict = dict(self.monitor.__dict__)
        indicator_dict = dict(self.indicator.__dict__)
        indicator_dict.pop('monitor')
        NFdict = dict(self.__dict__)
        NFdict.pop('monitor')
        NFdict.pop('indicator')
        NFdict.pop('smooth_func')
        full_dict = {'stimulation': NFdict,
                     'monitor': mondict,
                     'indicator': indicator_dict}

        return full_seq, full_dict


