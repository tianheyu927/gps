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
from gps.utility.demo_utils import compute_distance
from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm_utils import gauss_fit_joint_prior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
                END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
                END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
                CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy  # Maybe useful if we unpickle the file as controllers

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
             Generate demos and save them in a file for experiment.
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
            else:
                assert self.nn_demo
                self.agents = [agent_config[i]['type'](agent_config[i]) for i in xrange(len(agent_config))]

            # Roll out the demonstrations from controllers
            var_mult = self._hyperparams['algorithm'].get('demo_var_mult', 1.0)
            T = self.algorithms[0].T

            M = agent_config['conditions']
            N = self._hyperparams['algorithm']['num_demos']
            demo_M = self._hyperparams['algorithm']['demo_M']
            if type(agent_config) is not list:
                demos = {i:[] for i in xrange(len(agent_config))}
            elif demo_M != M or M == 1:
                demos = []
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
                    pol = self.algorithms[a].policy_opt.policy
                    for i in xrange(len(agent_config)):
                        for j in xrange(N):
                            demo = self.agents[i].sample(
                                pol, i, # Should be changed back to controller if using linearization
                                verbose=(j < self._hyperparams['verbose_trials']), noisy=False, record_image=True
                                )
                            demos[i].append(demo)
    
            self.filter(demos, demo_idx_conditions, agent_config, ioc_agent, demo_file)
            # # Filter failed demos
            # if type(agent_config) is list: # USED FOR ONE-SHOT LEARNING
            #     if agent_config[0].get('filter_demos', False):

            # elif agent_config.get('filter_demos', False): # USED FOR PR2
            #     filter_type = agent_config.get('filter_type', 'last')
            #     max_per_condition = agent_config.get('max_demos_per_condition', 999)
            #     target_position = agent_config['target_end_effector']
            #     dist_threshold = agent_config.get('success_upper_bound', 0.01)
            #     dists = compute_distance(target_position, SampleList(demos), agent_config['filter_end_effector_idxs'])
            #     failed_idx = []
            #     for i, distance in enumerate(dists):
            #         if filter_type == 'last':
            #             distance = distance[-1]
            #         elif filter_type == 'min':
            #             distance = min(distance)
            #         else:
            #             raise NotImplementedError()

            #         print distance
            #         if(distance > dist_threshold):
            #             failed_idx.append(i)
            #     LOGGER.debug("Removing %d failed demos: %s", len(failed_idx), str(failed_idx))
            #     demos_filtered = [demo for (i, demo) in enumerate(demos) if i not in failed_idx]
            #     demo_idx_conditions = [cond for (i, cond) in enumerate(demo_idx_conditions) if i not in failed_idx]
            #     demos = demos_filtered

            #     # Filter max demos per condition
            #     condition_to_demo = {
            #         cond: [demo for (i, demo) in enumerate(demos) if demo_idx_conditions[i]==cond][:max_per_condition]
            #         for cond in range(M)
            #     }
            #     LOGGER.debug('Successes per condition: %s', str([len(demo_list) for demo_list in condition_to_demo.values()]))
            #     demos = [demo for cond in condition_to_demo for demo in condition_to_demo[cond]]
            #     shuffle(demos)

            #     for demo in demos: demo.reset_agent(ioc_agent)
            #     demo_list = SampleList(demos)
            #     demo_store = {'demoX': demo_list.get_X(),
            #                   'demoU': demo_list.get_U(),
            #                   'demoO': demo_list.get_obs(),
            #                   'demoConditions': demo_idx_conditions}

            # elif agent_config['type']==AgentMuJoCo and \
            #     ('reacher' in agent_config.get('exp_name', []) or 'pointmass' in agent_config.get('exp_name', []) or\
            #         'peg' in agent_config.get('exp_name', [])):
            #     dists = []; failed_indices = []
            #     if demo_M == M or M == 1:
            #         failed_indices = {i:[] for i in xrange(demo_M)}
            #     success_thresh = agent_config['success_upper_bound'] # for reacher
            #     for m in range(M):
            #         if type(agent_config['target_end_effector']) is list:
            #             target_position = agent_config['target_end_effector'][m][:3]
            #         else:
            #             target_position = agent_config['target_end_effector'][:3]
            #         for i in range(N):
            #           index = m*N + i
            #           if demo_M != M or M == 1:
            #               demo = demos[index]
            #               demo_ee = demo.get(END_EFFECTOR_POINTS)
            #               dists.append(np.min(np.sqrt(np.sum((demo_ee[:, :3] - target_position.reshape(1, -1))**2, axis = 1))))
            #               if dists[index] >= success_thresh: #agent_config['success_upper_bound']:
            #                 failed_indices.append(index)
            #           else:
            #               demo = demos[m][i]
            #               demo_ee = demo.get(END_EFFECTOR_POINTS)
            #               dists.append(np.min(np.sqrt(np.sum((demo_ee[:, :3] - target_position.reshape(1, -1))**2, axis = 1))))
            #               if dists[i] >= success_thresh: #agent_config['success_upper_bound']:
            #                 failed_indices[m].append(i)
            #     if demo_M != M or M == 1:
            #         good_indices = [i for i in xrange(len(demos)) if i not in failed_indices]
            #         self._hyperparams['algorithm']['demo_cond'] = len(good_indices)
            #         filtered_demos = []
            #         filtered_demo_conditions = []
            #         for i in good_indices:
            #             filtered_demos.append(demos[i])
            #             filtered_demo_conditions.append(demo_idx_conditions[i])
    
            #         print 'Num demos:', len(filtered_demos)
            #         shuffle(filtered_demos)
            #         for demo in filtered_demos: demo.reset_agent(ioc_agent)
            #         demo_list =  SampleList(filtered_demos)
            #         demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs(),
            #                       'demoConditions': filtered_demo_conditions} #, \
            #     else:
            #         filtered_demos = {i:[] for i in xrange(demo_M)}
            #         filtered_demo_conditions = []
            #         for m in xrange(M):
            #             for j in xrange(N):
            #                 if j not in failed_indices[m]:
            #                     demos[m][j].reset_agent(ioc_agent)
            #                     filtered_demos[m].append(demos[m][j])
            #             filtered_demo_conditions.append(m)
            #             shuffle(filtered_demos[m])
            #         print "Num demos:", sum(len(filtered_demos[m]) for m in xrange(M))
            #         demo_list = [SampleList(filtered_demos[m]) for m in xrange(demo_M)]
            #         demo_store = {'demoX': [demo_list[m].get_X() for m in xrange(demo_M)], 'demoU': [demo_list[m].get_U() for m in xrange(demo_M)], \
            #                         'demoO': [demo_list[m].get_obs() for m in xrange(demo_M)], 'demoConditions': filtered_demo_conditions}
            # else:
            #     shuffle(demos)
            #     for demo in demos: demo.reset_agent(ioc_agent)
            #     demo_list = SampleList(demos)
            #     demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
            # # Save the demos.
            # self.data_logger.pickle(
            #     demo_file,
            #     copy.copy(demo_store)
            # )

        def filter(self, demos, demo_idx_conditions, agent_config, ioc_agent, demo_file, demo_M):
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

                for demo in demos: demo.reset_agent(ioc_agent)
                if demo_M != M or M == 1:
                    demo_list = SampleList(demos)
                    demo_store = {'demoX': demo_list.get_X(),
                                  'demoU': demo_list.get_U(),
                                  'demoO': demo_list.get_obs(),
                                  'demoConditions': demo_idx_conditions}
                else:
                    for m in xrange(demo_M):
                        shuffle(condition_to_demo[m])
                    demo_list = [SampleList(condition_to_demo[m]) for m in xrange(demo_M)]
                    demo_store = {'demoX': [demo_list[m].get_X() for m in xrange(demo_M)], 'demoU': [demo_list[m].get_U() for m in xrange(demo_M)], \
                                    'demoO': [demo_list[m].get_obs() for m in xrange(demo_M)], 'demoConditions': demo_idx_conditions}
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

        # Maybe useful to linearize the neural net policy before taking demos
        def linearize_policy(self, samples, cond):
            policy_prior = self.algorithms[cond]._hyperparams['policy_prior']
            init_policy_prior = policy_prior['type'](policy_prior)
            init_policy_prior._hyperparams['keep_samples'] = False
            dX, dU, T = self.algorithms[cond].dX, self.algorithms[cond].dU, self.algorithms[cond].T
            N = len(samples)
            X = samples.get_X()
            pol_mu, pol_sig = self.algorithms[cond].policy_opt.prob(samples.get_obs().copy())[:2]
            # Update the policy prior with collected samples
            init_policy_prior.update(SampleList([]), self.algorithms[cond].policy_opt, samples)
            # Collapse policy covariances. This is not really correct, but
            # it works fine so long as the policy covariance doesn't depend
            # on state.
            pol_sig = np.mean(pol_sig, axis=0)
            pol_info_pol_K = np.zeros((T, dU, dX))
            pol_info_pol_k = np.zeros((T, dU))
            pol_info_pol_S = np.zeros((T, dU, dU))
            pol_info_chol_pol_S = np.zeros((T, dU, dU))
            pol_info_inv_pol_S = np.empty_like(pol_info_chol_pol_S)
            # Estimate the policy linearization at each time step.
            for t in range(T):
                # Assemble diagonal weights matrix and data.
                dwts = (1.0 / N) * np.ones(N)
                Ts = X[:, t, :]
                Ps = pol_mu[:, t, :]
                Ys = np.concatenate((Ts, Ps), axis=1)
                # Obtain Normal-inverse-Wishart prior.
                mu0, Phi, mm, n0 = init_policy_prior.eval(Ts, Ps)
                sig_reg = np.zeros((dX+dU, dX+dU))
                # On the first time step, always slightly regularize covariance.
                if t == 0:
                    sig_reg[:dX, :dX] = 1e-8 * np.eye(dX)
                # Perform computation.
                pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0,
                                                                                                        dwts, dX, dU, sig_reg)
                pol_S += pol_sig[t, :, :]
                pol_info_pol_K[t, :, :], pol_info_pol_k[t, :] = pol_K, pol_k
                pol_info_pol_S[t, :, :], pol_info_chol_pol_S[t, :, :] = \
                        pol_S, sp.linalg.cholesky(pol_S)
                pol_info_inv_pol_S[t, :, :] = sp.linalg.solve(
                                                    pol_info_chol_pol_S[t, :, :],
                                                    np.linalg.solve(pol_info_chol_pol_S[t, :, :].T, np.eye(dU))
                                                    )
            return LinearGaussianPolicy(pol_info_pol_K, pol_info_pol_k, pol_info_pol_S, pol_info_chol_pol_S, \
                                                                            pol_info_inv_pol_S)

