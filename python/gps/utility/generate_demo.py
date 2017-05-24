""" This file generates for a point mass for 4 starting positions and a single goal position. """

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.use('Qt4Agg')


import sys
import os
import os.path
import logging
import copy
import argparse
import time
import threading
import numpy as np
import scipy as sp
import scipy.io
import numpy.matlib
import random
import pickle
from random import shuffle

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.utility.data_logger import DataLogger, open_zip
from gps.utility.general_utils import compute_distance, Timer, mkdir_p
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
                END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
                END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
                CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, END_EFFECTOR_POINTS_NO_TARGET, \
                END_EFFECTOR_POINT_VELOCITIES_NO_TARGET

LOGGER = logging.getLogger(__name__)

class GenDemo(object):
        """ Generator of demos. """
        def __init__(self, config):
            self._hyperparams = config
            self._conditions = config['common']['conditions']

            self.nn_demo = config['common']['nn_demo']
            self._exp_dir = config['common']['demo_exp_dir']
            self._data_files_dir = config['common']['data_files_dir']
            self._algorithm_files_dir = config['common']['demo_controller_file']
            self.data_logger = DataLogger()

        def load_algorithms(self):
            algorithm_files = self._algorithm_files_dir
            if isinstance(algorithm_files, basestring):
                with open_zip(algorithm_files, 'r') as f:
                    algorithms = [pickle.load(f)]
            else:
                algorithms = []
                for filename in algorithm_files:
                    with open_zip(filename, 'r') as f:
                        algorithms.append(pickle.load(f))
            return algorithms

        def generate(self, demo_file, ioc_agent):
            """
             Generate demos and save them in a file/files for experiment.
             Args:
                 demo_file - place to store the demos
                 ioc_agent - ioc agent, for grabbing the observation using the ioc agent's observation data types
             Returns: None.
            """
            # Load the algorithm

            self.algorithms = self.load_algorithms()
            self.algorithm = self.algorithms[0]

            # Keep the initial states of the agent the sames as the demonstrations.
            agent_config = self._hyperparams['demo_agent']
            if type(agent_config) is not list:
                self.agent = agent_config['type'](agent_config)
                M = agent_config['conditions']
            else:
                assert self.nn_demo and type(demo_file) is list
                M = agent_config[0]['conditions']

            # Roll out the demonstrations from controllers
            var_mult = self._hyperparams['algorithm'].get('demo_var_mult', 1.0)
            T = self.algorithms[0].T

            N = self._hyperparams['algorithm']['num_demos']
            demo_M = self._hyperparams['algorithm']['demo_M']
            start_idx = self._hyperparams.get('start_idx', 0)
            batch = self._hyperparams.get('batch', 0)
            if type(agent_config) is list:
                demos = {i:[] for i in xrange(len(agent_config))}
                demo_idx_conditions = {i:[] for i in xrange(len(agent_config))}
            elif demo_M != M or M == 1:
                demos = []
                demo_idx_conditions = []
            else:
                demos = {i:[] for i in xrange(M)}
                demo_idx_conditions = []  # Stores conditions for each demo
            if not self.nn_demo:
                controllers = {}

                # Store each controller under M conditions into controllers.
                for i in xrange(M):
                    controllers[i] = self.algorithm.cur[i].traj_distr
                controllers_var = copy.copy(controllers)
                for i in xrange(M):
                    # Increase controller variance.
                    controllers_var[i].chol_pol_covar *= var_mult
                    # Gather demos.
                    for j in xrange(N):
                        demo = self.agent.sample(
                            controllers_var[i], i,
                            verbose=(j < self._hyperparams['verbose_trials']), noisy=True,
                            save = True
                        )
                        if demo_M != M or M == 1:
                            demos.append(demo)
                            demo_idx_conditions.append(i)
                        else:
                            demos[i].append(demo)
            else:
                all_pos_body_offsets = []
                # Gather demos.
                if type(agent_config) is not list:
                    for a in xrange(len(self.algorithms)):
                        pol = self.algorithms[a].policy_opt.policy
                        for i in xrange(M / len(self.algorithms) * a, M / len(self.algorithms) * (a + 1)):
                            for j in xrange(N):
                                demo = self.agent.sample(
                                    pol, i, # Should be changed back to controller if using linearization
                                    verbose=(j < self._hyperparams['verbose_trials']), noisy=True
                                    )
                                if demo_M != M or M == 1:
                                    demos.append(demo)
                                    demo_idx_conditions.append(i)
                                else:
                                    demos[i].append(demo)
                else:
                    assert len(self.algorithms) == 1
                    pol = self.algorithms[0].policy_opt.policy
                    for i in xrange(len(agent_config)):
                        agent = agent_config[i]['type'](agent_config[i])
                        M = agent_config[i]['conditions']
                        for j in xrange(M):
                            for k in xrange(N):
                                if 'record_gif' in self._hyperparams:
                                    gif_config = self._hyperparams['record_gif']
                                    if k < gif_config.get('gifs_per_condition', float('inf')):
                                        gif_fps = gif_config.get('fps', None)
                                        gif_dir = gif_config.get('demo_gif_dir', 'gps/data/demo_gifs/')
                                        gif_dir = gif_dir + 'color_%d/' % (i+start_idx*batch)
                                        mkdir_p(gif_dir)
                                        gif_name = os.path.join(gif_dir,'cond%d.samp%d.gif' % (j, k))
                                    else:
                                        gif_name=None
                                        gif_fps = None
                                else:
                                    gif_name=None
                                    gif_fps = None
                                demo = agent.sample(
                                    pol, j, # Should be changed back to controller if using linearization
                                    verbose=(k < self._hyperparams['verbose_trials']), noisy=True, record_image=True, generate_demo=True,
                                    include_no_target=True, record_gif=gif_name, record_gif_fps=gif_fps, reset=False #don't reset images
                                    )
                                demos[i].append(demo)
                                demo_idx_conditions[i].append(j)

            if type(agent_config) is not list:
                self.filter(demos, demo_idx_conditions, agent_config, ioc_agent, demo_file, demo_M)
            else:
                for i in xrange(len(agent_config)):
                    with Timer('Saving demo file %d' % (i + start_idx*batch)):
                        if agent_config[i].get('save_images', False):
                            gif_config = self._hyperparams['record_gif']
                            gif_dir = gif_config.get('demo_gif_dir', 'gps/data/demo_gifs/')
                            gif_dir = gif_dir + 'color_%d/' % (i+start_idx*batch)
                        else:
                            gif_dir = None
                        self.filter(demos[i], demo_idx_conditions[i], agent_config[i], ioc_agent, demo_file[i], demo_M, gif_dir=gif_dir)

        def filter(self, demos, demo_idx_conditions, agent_config, ioc_agent, demo_file, demo_M, gif_dir=None):
            """
            Filter out failed demos.
            Args:
                demos: generated demos
                demo_idx_conditions: the conditions of generated demos
                agent_config: config of the demo agent
                ioc_agent: the agent for ioc
                demo_file: the path to save demos
                demo_M: demo conditions
            """
            M = agent_config['conditions']
            N = self._hyperparams['algorithm']['num_demos']
            
            # Filter failed demos
            if 'filter_demos' in agent_config:
                filter_options = agent_config['filter_demos']
                filter_type = filter_options.get('type', 'min')
                targets = filter_options['target']
                pos_idx = filter_options.get('state_idx', range(4, 7))
                end_effector_idx = filter_options.get('end_effector_idx', range(0, 3))
                max_per_condition = filter_options.get('max_demos_per_condition', 999)
                dist_threshold = filter_options.get('success_upper_bound', 0.01)
                cur_samples = SampleList(demos)
                dists = compute_distance(targets, cur_samples, pos_idx, end_effector_idx, filter_type=filter_type)
                failed_idx = []
                for i, distance in enumerate(dists):
                    if (distance > dist_threshold):
                        failed_idx.append(i)

                LOGGER.debug("Removing %d failed demos: %s", len(failed_idx), str(failed_idx))
                demos_filtered = [demo for (i, demo) in enumerate(demos) if i not in failed_idx]
                demo_idx_conditions = [cond for (i, cond) in enumerate(demo_idx_conditions) if i not in failed_idx]
                demos = demos_filtered

                # Filter max demos per condition
                condition_to_demo = {
                    cond: [demo for (i, demo) in enumerate(demos) if demo_idx_conditions[i]==cond][:max_per_condition]
                    for cond in range(M)
                }
                LOGGER.debug('Successes per condition: %s', str([len(demo_list) for demo_list in condition_to_demo.values()]))
                demos = [demo for cond in condition_to_demo for demo in condition_to_demo[cond]]
                shuffle(demos)
                # import pdb; pdb.set_trace()
                for demo in demos: demo.reset_agent(ioc_agent) # save images as observations!
                if not agent_config.get('save_images', False):
                    if demo_M != M or M == 1:
                        demo_list = SampleList(demos)
                        demo_store = {'demoX': demo_list.get_X(),
                                      'demoU': demo_list.get_U(),
                                    #   'demoO': demo_list.get_obs(),
                                      'demoO': demo_list.get_obs()[:, :, 10:].astype(np.uint8), #only saving the integet part (for reacher only)
                                      'demoConditions': demo_idx_conditions}
                    else:
                        for m in xrange(demo_M):
                            shuffle(condition_to_demo[m])
                        demo_list = [SampleList(condition_to_demo[m]) for m in xrange(demo_M)]
                        demo_store = {'demoX': [demo_list[m].get_X() for m in xrange(demo_M)], 'demoU': [demo_list[m].get_U() for m in xrange(demo_M)], \
                                        'demoO': [demo_list[m].get_obs() for m in xrange(demo_M)], 'demoConditions': demo_idx_conditions}
                else:
                    demo_list = SampleList(demos)
                    demo_store = {'demoX': demo_list.get_X(),
                                  'demoU': demo_list.get_U(),
                                  'demoConditions': demo_idx_conditions}
                    # TODO: save observations as images.
                    assert gif_dir is not None
                    for cond in xrange(M):
                        if cond not in demo_idx_conditions:
                            gif_name = os.path.join(gif_dir,'cond%d.samp0.gif' % cond)
                            os.remove(gif_name)
            else:
                shuffle(demos)
                for demo in demos: demo.reset_agent(ioc_agent)
                demo_list = SampleList(demos)
                demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
            # Save the demos.
            self.data_logger.pickle(
                demo_file,
                copy.copy(demo_store)
            )
            # Clear memory for sequential pickles. Has some issue right now.
            # self.data_logger.clear_memo()
