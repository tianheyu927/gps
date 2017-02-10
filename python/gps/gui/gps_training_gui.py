"""
GPS Training GUI

The GPS Training GUI is used to interact with the GPS algorithm during training.
It contains the below seven functionalities:

Action Panel                contains buttons for stop, reset, go, fail
Action Status Textbox       displays action status
Algorithm Status Textbox    displays algorithm status
Cost Plot                   displays costs after each iteration
Algorithm Output Textbox    displays algorithm output after each iteration
3D Trajectory Visualizer    displays 3D trajectories after each iteration
Image Visualizer            displays images received from a rostopic

For more detailed documentation, visit: rll.berkeley.edu/gps/gui
"""
import time
import threading
import sys

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from gps.gui.config import config
from gps.gui.action_panel import Action, ActionPanel
from gps.gui.textbox import Textbox
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D
from gps.gui.image_visualizer import ImageVisualizer
from gps.gui.util import buffered_axis_limits, load_data_from_npz

from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, RGB_IMAGE, RGB_IMAGE_SIZE, IMAGE_FEAT

from gps.gui.line_plot import LinePlotter, ScatterPlot

NUM_DEMO_PLOTS = 5

# Needed for typechecks
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.behavior_cloning import BehaviorCloning

class GPSTrainingGUI(object):

    def __init__(self, hyperparams, gui_on=True):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        if 'target_filename' in self._hyperparams:
            self._target_filename = self._hyperparams['target_filename']
        else:
            self._target_filename = None

        self.gui_on = gui_on # whether or not to draw

        # GPS Training Status.
        self.mode = config['initial_mode']  # Modes: run, wait, end, request, process.
        self.request = None                 # Requests: stop, reset, go, fail, None.
        self.err_msg = None
        self._colors = {
            'run': 'cyan',
            'wait': 'orange',
            'end': 'red',

            'stop': 'red',
            'reset': 'yellow',
            'go': 'green',
            'fail': 'magenta',
        }
        self._first_update = True

        # Actions.
        actions_arr = [
            Action('stop',  'stop',  self.request_stop,  axis_pos=0),
            Action('reset', 'reset', self.request_reset, axis_pos=1),
            Action('go',    'go',    self.request_go,    axis_pos=2),
            Action('fail',  'fail',  self.request_fail,  axis_pos=3),
        ]

        # Setup figure.
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        for key in plt.rcParams:
            if key.startswith('keymap.'):
                plt.rcParams[key] = ''

        self._fig = plt.figure(figsize=config['figsize'])
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                wspace=0, hspace=0)

        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(18, 8)
        self._gs_action_panel           = self._gs[0:1,  0:8]
        self._gs_action_output          = self._gs[1:2,  0:4]
        self._gs_status_output          = self._gs[2:3,  0:4]
        self._gs_cost_plotter           = self._gs[1:3,  4:8]
        self._gs_gt_cost_plotter        = self._gs[9:11,  4:8]
        self._gs_demo_cost_plotter      = self._gs[7:9,  4:8]
        self._gs_scatter_cost_plotter   = self._gs[7:9,  0:4]
        self._gs_algthm_output          = self._gs[3:7,  0:4]
        if config['image_on']:
            self._gs_traj_visualizer    = self._gs[11:18, 0:4]
            self._gs_image_visualizer   = self._gs[11:18, 4:8]
        elif config['fp_on']:
            self._gs_traj_visualizer = self._gs[8:16, 0:4]
            self._gs_fp_visualizer = self._gs[8:16, 4:8]
        else:
            self._gs_traj_visualizer    = self._gs[11:18, 0:8]

        # Create GUI components.
        self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 1, 4, actions_arr)
        self._scatter_cost_plotter = ScatterPlot(self._fig, self._gs_scatter_cost_plotter, xlabel='Ground_Truth',
                                                ylabel='Learned_Cost', gui_on=gui_on)
        self._action_output = Textbox(self._fig, self._gs_action_output, border_on=True, gui_on=gui_on)
        self._status_output = Textbox(self._fig, self._gs_status_output, border_on=False, gui_on=gui_on)
        self._algthm_output = Textbox(self._fig, self._gs_algthm_output,
                max_display_size=config['algthm_output_max_display_size'],
                log_filename=self._log_filename,
                fontsize=config['algthm_output_fontsize'],
                font_family='monospace',
                gui_on=gui_on)
        self._cost_plotter = MeanPlotter(self._fig, self._gs_cost_plotter,
                color='blue', label='mean cost', gui_on=gui_on)
        self._gt_cost_plotter = MeanPlotter(self._fig, self._gs_gt_cost_plotter,
                color='red', label='ground truth cost', gui_on=gui_on)
        self._demo_cost_plotter = LinePlotter(self._fig, self._gs_demo_cost_plotter,
                                         color='blue', label='demo cost', num_plots=NUM_DEMO_PLOTS*3, gui_on=gui_on)
        self._traj_visualizer = Plotter3D(self._fig, self._gs_traj_visualizer,
                num_plots=self._hyperparams['conditions'], gui_on=gui_on)
        if config['image_on']:
            self._image_visualizer = ImageVisualizer(self._fig,
                    self._gs_image_visualizer, cropsize=config['image_size'],
                    rostopic=config['image_topic'], show_overlay_buttons=True)
        if config['fp_on']:
            self._fp_visualizer = plt.subplot(self._gs_fp_visualizer)

        # Setup GUI components.
        self._algthm_output.log_text('\n')
        self.set_output_text(self._hyperparams['info'])
        if config['initial_mode'] == 'run':
            self.run_mode()
        else:
            self.wait_mode()

        # Setup 3D Trajectory Visualizer plot titles and legends
        for m in range(self._hyperparams['conditions']):
            self._traj_visualizer.set_title(m, 'Condition %d' % (m))
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='green', label='Trajectory Samples')
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='blue', label='Policy Samples')
        self._traj_visualizer.add_legend(linestyle='None', marker='x',
                color=(0.5, 0, 0), label='LG Controller Means')
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='red', label='LG Controller Distributions')

        if self.gui_on:
            self._fig.canvas.draw()

        # Display calculating thread
        def display_calculating(delay, run_event):
            while True:
                if not run_event.is_set():
                    run_event.wait()
                if run_event.is_set():
                    self.set_status_text('Calculating.')
                    time.sleep(delay)
                if run_event.is_set():
                    self.set_status_text('Calculating..')
                    time.sleep(delay)
                if run_event.is_set():
                    self.set_status_text('Calculating...')
                    time.sleep(delay)

        self._calculating_run = threading.Event()
        self._calculating_thread = threading.Thread(target=display_calculating,
                args=(1, self._calculating_run))
        self._calculating_thread.daemon = True
        self._calculating_thread.start()

    # GPS Training functions
    def request_stop(self, event=None):
        self.request_mode('stop')

    def request_reset(self, event=None):
        self.request_mode('reset')

    def request_go(self, event=None):
        self.request_mode('go')

    def request_fail(self, event=None):
        self.request_mode('fail')

    def request_mode(self, request):
        """
        Sets the request mode (stop, reset, go, fail). The request is read by
        gps_main before sampling, and the appropriate action is taken.
        """
        self.mode = 'request'
        self.request = request
        self.set_action_text(self.request + ' requested')
        self.set_action_bgcolor(self._colors[self.request], alpha=0.2)

    def process_mode(self):
        """
        Completes the current request, after it is first read by gps_main.
        Displays visual confirmation that the request was processed,
        displays any error messages, and then switches into mode 'run' or 'wait'.
        """
        self.mode = 'process'
        self.set_action_text(self.request + ' processed')
        self.set_action_bgcolor(self._colors[self.request], alpha=1.0)
        if self.err_msg:
            self.set_action_text(self.request + ' processed' + '\nERROR: ' +
                                 self.err_msg)
            self.err_msg = None
            time.sleep(1.0)
        else:
            time.sleep(0.5)
        if self.request in ('stop', 'reset', 'fail'):
            self.wait_mode()
        elif self.request == 'go':
            self.run_mode()
        self.request = None

    def wait_mode(self):
        self.mode = 'wait'
        self.set_action_text('waiting')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def run_mode(self):
        self.mode = 'run'
        self.set_action_text('running')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def end_mode(self):
        self.mode = 'end'
        self.set_action_text('ended')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def estop(self, event=None):
        self.set_action_text('estop: NOT IMPLEMENTED')

    # GUI functions
    def set_action_text(self, text):
        self._action_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels
        self._gt_cost_plotter.draw_ticklabels()

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels
        self._gt_cost_plotter.draw_ticklabels()

    def set_status_text(self, text):
        self._status_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels
        self._gt_cost_plotter.draw_ticklabels()

    def set_output_text(self, text):
        self._algthm_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels
        self._gt_cost_plotter.draw_ticklabels()

    def append_output_text(self, text):
        self._algthm_output.append_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels
        self._gt_cost_plotter.draw_ticklabels()

    def start_display_calculating(self):
        self._calculating_run.set()

    def stop_display_calculating(self):
        self._calculating_run.clear()

    def set_image_overlays(self, condition):
        """
        Sets up the image visualizer with what images to overlay if
        "overlay_initial_image" or "overlay_target_image" is pressed.
        """
        if not config['image_on'] or not self._target_filename:
            return
        initial_image = load_data_from_npz(self._target_filename,
                config['image_overlay_actuator'], str(condition),
                'initial', 'image', default=None)
        target_image  = load_data_from_npz(self._target_filename,
            config['image_overlay_actuator'], str(condition),
                'target',  'image', default=None)
        self._image_visualizer.set_initial_image(initial_image,
                alpha=config['image_overlay_alpha'])
        self._image_visualizer.set_target_image(target_image,
                alpha=config['image_overlay_alpha'])

    # Iteration update functions
    def update(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists, eval_pol_gt = False,
               ioc_demo_losses=None, ioc_sample_losses=None, ioc_dist_cost=None, ioc_demo_dist_cost=None,
               ioc_demo_true_losses=None, ioc_sample_true_losses=None, ioc_demo_target_losses=None,
               ioc_sample_target_losses=None):
        """
        After each iteration, update the iteration data output, the cost plot,
        and the 3D trajectory visualizations (if end effector points exist).
        """
        if self._first_update:
            self._output_column_titles(algorithm)
            self._first_update = False

        if config['fp_on'] and not algorithm._hyperparams['bc']:
            if RGB_IMAGE in agent.obs_data_types:
                img = []
                samples = []
                images = []

                for sample_list in traj_sample_lists:
                    samples.append(sample_list.get_samples()[0])
                    size = np.array(samples[0].get(RGB_IMAGE_SIZE))
                    img = samples[0].get(RGB_IMAGE, 0) #fixed image
                    img = img.reshape(size)

                fps = []
                for sample in samples:
                    fp = sample.get(IMAGE_FEAT, 0)
                    #fp = fp.reshape(-1, 2).T.reshape(-1)
                    fps.append(fp)
                    #fp1 = sample.get(IMAGE_FEAT, 1)
                    #fp2 = sample.get(IMAGE_FEAT, 2)
                    #img = sample.get(RGB_IMAGE, 0)
                self._update_feature_visualization(img, fps)
                    #images = sample.get(RGB_IMAGE)
                #for image in images:
                   # img.append(image.reshape(size).transpose((2, 1, 0)))
                   # feature_points = algorithm.policy_opt.fp_vals
                   # idx = np.random.randint(len(img))
        if not algorithm._hyperparams.get('bc', False):
            costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        else:
            costs = 'N/A'
        if algorithm._hyperparams['ioc']:
            gt_costs = [np.mean(np.sum(algorithm.prev[m].cgt, axis=1)) for m in range(algorithm.M)]
            self._update_iteration_data(itr, algorithm, gt_costs, pol_sample_lists, eval_pol_gt)
            self._gt_cost_plotter.update(gt_costs, t=itr)
        else:
            self._update_iteration_data(itr, algorithm, costs, pol_sample_lists)
        if not algorithm._hyperparams.get('bc', False):
            self._cost_plotter.update(costs, t=itr)
            print "Learned cost: %.2f" %np.mean(costs)
        if END_EFFECTOR_POINTS in agent.x_data_types and not algorithm._hyperparams.get('bc', False):
            self._update_trajectory_visualizations(algorithm, agent,
                    traj_sample_lists, pol_sample_lists)

        if ioc_demo_losses:
            self._update_ioc_demo_plot(ioc_demo_losses, ioc_sample_losses)
        if ioc_dist_cost:
            self._update_scatter_plot(ioc_dist_cost, ioc_demo_dist_cost)
        if ioc_demo_true_losses:
            demo_losses = (ioc_demo_true_losses, ioc_demo_target_losses)
            sample_losses = (ioc_sample_true_losses, ioc_sample_target_losses)
            self._update_scatter_plot(demo_losses, sample_losses)

        if self.gui_on:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events() # Fixes bug in Qt4Agg backend

    def _output_column_titles(self, algorithm, policy_titles=False):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        self.set_output_text(self._hyperparams['experiment_name'])
        if isinstance(algorithm, AlgorithmMDGPS) or isinstance(algorithm, AlgorithmBADMM) or isinstance(algorithm, BehaviorCloning):
            condition_titles = '%3s | %8s %12s' % ('', '', '')
            # itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
            itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_fk_cost')
        else:
            condition_titles = '%3s | %8s' % ('', '')
            itr_data_fields  = '%3s | %8s' % ('itr', 'avg_cost')
        for m in range(algorithm.M):
            if not algorithm._hyperparams.get('bc', False):
                condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
                itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            else:
                condition_titles += ' | '
                itr_data_fields  += ' | '

            #if algorithm._hyperparams['ioc'] and not algorithm._hyperparams['learning_from_prior']:
            #    condition_titles += ' %8s' % ('')
            #    itr_data_fields  += ' %8s' % ('kl_div')
            if 'target_end_effector' in algorithm._hyperparams:
                condition_titles += ' %8s' % ('')
                itr_data_fields  += ' %8s' % ('mean_dist')

            if isinstance(algorithm, AlgorithmBADMM):
                condition_titles += ' %8s %8s %8s' % ('', '', '')
                itr_data_fields  += ' %8s %8s %8s' % ('pol_cost', 'kl_div_i', 'kl_div_f')
            elif isinstance(algorithm, AlgorithmMDGPS) or isinstance(algorithm, BehaviorCloning):
                condition_titles += ' %8s' % ('')
                # itr_data_fields  += ' %8s' % ('pol_cost')
                itr_data_fields  += ' %8s' % ('fk_cost')
        self.append_output_text(condition_titles)
        self.append_output_text(itr_data_fields)

    def _update_iteration_data(self, itr, algorithm, costs, pol_sample_lists, eval_pol_gt=False):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        if not algorithm._hyperparams.get('bc', False):
            avg_cost = np.mean(costs)
        else:
            avg_cost = 'N/A'
        pol_costs = [-123 for _ in range(algorithm.M)]
        if pol_sample_lists is not None:
            # test_idx = algorithm._hyperparams['test_conditions']
            # # import pdb; pdb.set_trace()
            # # pol_sample_lists is a list of singletons
            # samples = [sl[0] for sl in pol_sample_lists]
            # if not eval_pol_gt:
            #     if 'global_cost' in algorithm._hyperparams and algorithm._hyperparams['global_cost'] and \
            #             type(algorithm.cost) != list:
            #         pol_costs = [np.sum(algorithm.cost.eval(s)[0])
            #                 for s in samples]
            #     else:
            #         pol_costs = [np.sum(algorithm.cost[idx].eval(s)[0])
            #                 for s, idx in zip(samples, test_idx)]
            # else:
            #     assert algorithm._hyperparams['ioc']
            #     pol_costs = [np.sum(algorithm.gt_cost[idx].eval(s)[0])
            #             for s, idx in zip(samples, test_idx)]
            # if not algorithm._hyperparams.get('bc', False):
            #     itr_data = '%3d | %8.2f %12.2f' % (itr, avg_cost, np.mean(pol_costs))
            # else:
            #     itr_data = '%3d | %8s %12.2f' % (itr, avg_cost, np.mean(pol_costs))
            fk_costs = [np.mean(np.sum(algorithm.prev[m].cfk, axis=1)) for m in range(algorithm.M)]
            itr_data = '%3d | %8.2f %12.2f' % (itr, avg_cost, np.mean(fk_costs))
        else:
            test_idx = None
            itr_data = '%3d | %8.2f' % (itr, avg_cost)

        for m in range(algorithm.M):
            if not algorithm._hyperparams.get('bc', False):
                cost = costs[m]
                step = algorithm.prev[m].step_mult * algorithm.base_kl_step
                entropy = 2*np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar,
                        axis1=1, axis2=2)))
                itr_data += ' | %8.2f %8.2f %8.2f' % (cost, step, entropy)
                #if algorithm._hyperparams['ioc'] and not algorithm._hyperparams['learning_from_prior']:
                #    itr_data += ' %8.2f' % (algorithm.kl_div[itr][m])
            else:
                itr_data += ' | '

            if pol_sample_lists is None:
                if algorithm.dists_to_target:
                    if itr in algorithm.dists_to_target and algorithm.dists_to_target[itr]:
                        itr_data += ' %8.2f' % (algorithm.dists_to_target[itr][m])
            else:
                if 'target_end_effector' in algorithm._hyperparams:
                    from gps.proto.gps_pb2 import END_EFFECTOR_POINTS
                    if type(algorithm._hyperparams['target_end_effector']) is list:
                        target_position = algorithm._hyperparams['target_end_effector'][m][:3]
                    else:
                        target_position = algorithm._hyperparams['target_end_effector'][:3]
                    cur_samples = pol_sample_lists[m].get_samples()
                    sample_end_effectors = [cur_samples[i].get(END_EFFECTOR_POINTS) for i in xrange(len(cur_samples))]
                    dists = [np.amin(np.sqrt(np.sum((sample_end_effectors[i][:, :3] - \
                                target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                                for i in xrange(len(cur_samples))]
                    itr_data += ' %8.2f' % (sum(dists) / len(cur_samples))
            if isinstance(algorithm, AlgorithmBADMM):
                kl_div_i = algorithm.cur[m].pol_info.init_kl.mean()
                kl_div_f = algorithm.cur[m].pol_info.prev_kl.mean()
                itr_data += ' %8.2f %8.2f %8.2f' % (pol_costs[m], kl_div_i, kl_div_f)
            elif isinstance(algorithm, AlgorithmMDGPS) or isinstance(algorithm, BehaviorCloning):
                # TODO: Change for test/train better.
                # if test_idx == algorithm._hyperparams['train_conditions']:
                #     itr_data += ' %8.2f' % (pol_costs[m])
                # else:
                #     itr_data += ' %8s' % ("N/A")
                itr_data += ' %8.2f' % (fk_costs[m])
        self.append_output_text(itr_data)

    def _update_feature_visualization(self, image, feature_points):
        """
        Update feature point visualization
        """
        self._fp_visualizer.cla()
        IMAGE_SIZE = 64
        image = sp.misc.imresize(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self._fp_visualizer.imshow(image)

        print 'Feature Points:', feature_points

        fp_x = []
        fp_y = []
        colors = []
        condition_colors = cm.rainbow(np.linspace(0, 1, len(feature_points)))
        for i, feature_point in enumerate(feature_points):
            fp = (feature_point + 1.) * IMAGE_SIZE / 2
            i_fp_x = fp[0::2]
            i_fp_y = IMAGE_SIZE - fp[1::2]
            N = len(i_fp_y)
            i_colors = np.tile(condition_colors[i], [N, 1])
            fp_x.append(i_fp_x)
            fp_y.append(i_fp_y)
            colors.append(i_colors)
        fp_x = np.concatenate(fp_x)
        fp_y = np.concatenate(fp_y)
        colors = np.concatenate(colors)
        print 'FP_X:', fp_x
        print 'FP_Y:', fp_y
        for i in range(N*len(feature_points)):
            self._fp_visualizer.scatter(fp_x[i], fp_y[i], color=colors[i])

    def _update_trajectory_visualizations(self, algorithm, agent,
                traj_sample_lists, pol_sample_lists):
        """
        Update 3D trajectory visualizations information: the trajectory samples,
        policy samples, and linear Gaussian controller means and covariances.
        """
        xlim, ylim, zlim = self._calculate_3d_axis_limits(traj_sample_lists, pol_sample_lists)
        for m in range(algorithm.M):
            self._traj_visualizer.clear(m)
            self._traj_visualizer.set_lim(i=m, xlim=xlim, ylim=ylim, zlim=zlim)
            self._update_samples_plots(traj_sample_lists, m, 'green', 'Trajectory Samples')
            self._update_linear_gaussian_controller_plots(algorithm, agent, m)
            if pol_sample_lists:
                self._update_samples_plots(pol_sample_lists,  m, 'blue',  'Policy Samples')
        if self.gui_on:
            self._traj_visualizer.draw()    # this must be called explicitly

    def _calculate_3d_axis_limits(self, traj_sample_lists, pol_sample_lists):
        """
        Calculate the 3D axis limits shared between trajectory plots,
        based on the minimum and maximum xyz values across all samples.
        """
        all_eept = np.empty((0, 3))
        sample_lists = traj_sample_lists
        if pol_sample_lists:
            sample_lists += traj_sample_lists
        for sample_list in sample_lists:
            for sample in sample_list.get_samples():
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                for i in range(ee_pt.shape[1]/3):
                    ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                    all_eept = np.r_[all_eept, ee_pt_i]
        min_xyz = np.amin(all_eept, axis=0)
        max_xyz = np.amax(all_eept, axis=0)
        xlim = buffered_axis_limits(min_xyz[0], max_xyz[0], buffer_factor=1.25)
        ylim = buffered_axis_limits(min_xyz[1], max_xyz[1], buffer_factor=1.25)
        zlim = buffered_axis_limits(min_xyz[2], max_xyz[2], buffer_factor=1.25)
        return xlim, ylim, zlim

    def _update_linear_gaussian_controller_plots(self, algorithm, agent, m):
        """
        Update the linear Guassian controller plots with iteration data,
        for the mean and covariances of the end effector points.
        """
        # Calculate mean and covariance for end effector points
        eept_idx = agent.get_idx_x(END_EFFECTOR_POINTS)
        start, end = eept_idx[0], eept_idx[-1]
        mu, sigma = algorithm.traj_opt.forward(algorithm.prev[m].traj_distr, algorithm.prev[m].traj_info)
        mu_eept, sigma_eept = mu[:, start:end+1], sigma[:, start:end+1, start:end+1]

        # Linear Gaussian Controller Distributions (Red)
        for i in range(mu_eept.shape[1]/3):
            mu, sigma = mu_eept[:, 3*i+0:3*i+3], sigma_eept[:, 3*i+0:3*i+3, 3*i+0:3*i+3]
            self._traj_visualizer.plot_3d_gaussian(i=m, mu=mu, sigma=sigma,
                    edges=100, linestyle='-', linewidth=1.0, color='red',
                    alpha=0.15, label='LG Controller Distributions')

        # Linear Gaussian Controller Means (Dark Red)
        for i in range(mu_eept.shape[1]/3):
            mu = mu_eept[:, 3*i+0:3*i+3]
            self._traj_visualizer.plot_3d_points(i=m, points=mu, linestyle='None',
                    marker='x', markersize=5.0, markeredgewidth=1.0,
                    color=(0.5, 0, 0), alpha=1.0, label='LG Controller Means')

    def _update_samples_plots(self, sample_lists, m, color, label):
        """
        Update the samples plots with iteration data, for the trajectory samples
        and the policy samples.
        """
        samples = sample_lists[m].get_samples()
        for sample in samples:
            ee_pt = sample.get(END_EFFECTOR_POINTS)
            for i in range(ee_pt.shape[1]/3):
                ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                self._traj_visualizer.plot_3d_points(m, ee_pt_i, color=color, label=label)

    def _update_ioc_demo_plot(self, demo_losses, sample_losses):
        for i in range(len(demo_losses)):
            self._demo_cost_plotter.set_sequence(i, demo_losses[i])

        for i in range(max(NUM_DEMO_PLOTS, len(sample_losses))):
            self._demo_cost_plotter.set_sequence(len(demo_losses)+i, sample_losses[i], style='--')
            pass

    def _update_scatter_plot(self, demo_data, sample_data):
        self._scatter_cost_plotter.clear()
        self._scatter_cost_plotter.add_data(sample_data[0], sample_data[1])
        self._scatter_cost_plotter.add_data(demo_data[0], demo_data[1], color='red')

    def save_figure(self, filename):
        self._fig.savefig(filename)
