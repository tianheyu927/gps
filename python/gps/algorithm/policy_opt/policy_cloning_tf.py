""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.
import tensorflow as tf

from random import shuffle
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

class PolicyCloningTf(PolicyOptTf):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams, dO, dU):
        super(PolicyCloningTf, self).__init__(hyperparams, dO, dU)
        if hyperparams.get('agent', False):
            test_agent = hyperparams['agent']  # Required for sample packing
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file']
        self.update_batch_size = self._hyperparams.get('update_batch_size', 1)
        if hyperparams.get('agent', False):
            trainO, trainU, valO, valU = self.extract_supervised_data(demo_file)
            if valO is not None:
                self.update(trainO, trainU, 1.0, 1.0, test_obs=valO,\
                            test_acts=valU, behavior_clone=True)
            else:
                self.update(trainO, trainU, 1.0, 1.0, test_obs=None,\
                            test_acts=None, behavior_clone=True)
            self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        self.demos = None
        self.train_demos = None
        self.val_demos = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']


    def extract_supervised_data(self, demo_file):
        demos = extract_demo_dict(demo_file)
        n_val = self._hyperparams['n_val'] # number of demos for testing
        if type(demo_file) is list:
            N = np.sum(demo['demoO'].shape[0] for i, demo in demos.iteritems())
            print "Number of demos: %d" % N
            X = demos[0]['demoX']
            U = demos[0]['demoU']
            O = demos[0]['demoO']
            cond = np.array(demos[0]['demoConditions'])
            for i in xrange(1, len(demo_file)):
                n_demo = demos[i]['demoX'].shape[0]
                X = np.concatenate((X, demos[i]['demoX']))
                U = np.concatenate((U, demos[i]['demoU']))
                O = np.concatenate((O, demos[i]['demoO']))
                cond = np.concatenate((cond, np.array(demos[i]['demoConditions'])))
            print "Number of few-shot demos is %d" % (X.shape[0])
        else:
            N = demos['demoO'].shape[0]
            print "Number of demos: %d" % N
            X = demos['demoX']
            U = demos['demoU']
            O = demos['demoO']
            cond = demos['demoConditions']
        if n_val != 0:
            n_folders = len(demos.keys())
            idx = range(n_folders)
            shuffle(idx)
            self.demos = demos
            self.val_demos = {key: demos[key] for key in np.array(demos.keys())[idx[:n_val]]}
            self.train_demos = {key: demos[key] for key in np.array(demos.keys())[idx[n_val:]]}
            self.val_idx = sorted(idx[:n_val])
            self.train_idx = sorted(idx[n_val:])
            valO = O[idx[:n_val]]
            valX = X[idx[:n_val]]
            valU = U[:idx[:n_val]]
            val_cond = cond[idx[:n_val]]
            O = O[idx[n_val:]]
            X = X[idx[n_val:]]
            U = U[idx[n_val:]]
            cond = cond[idx[n_val:]]
        else:
            self.demos = self.train_demos = demos
            self.val_demos = None
            valO = None
            valX = None
            valU = None
            val_cond = None

        return O, U, valO, valU

    def sample(self, agent, idx, conditions, N=1, testing=False):
        samples = []
        for i in xrange(len(conditions)):
            for j in xrange(N):
                if 'record_gif' in self._hyperparams:
                    gif_config = self._hyperparams['record_gif']
                    if j < gif_config.get('gifs_per_condition', float('inf')):
                        gif_fps = gif_config.get('fps', None)
                        if testing:
                            gif_dir = gif_config.get('test_gif_dir', self._hyperparams['plot_dir'] + 'test_gifs/')
                        else:
                            gif_dir = gif_config.get('gif_dir', self._hyperparams['plot_dir'] + 'gifs/')
                        mkdir_p(gif_dir)
                        gif_name = os.path.join(gif_dir,'color%d.cond%d.samp%d.gif' % (idx, i, j))
                    else:
                        gif_name=None
                        gif_fps = None
                else:
                    gif_name=None
                    gif_fps = None
                samples.append(agent.sample(
                    self.policy, conditions[i],
                    verbose=False, save=False, noisy=False,
                    record_gif=gif_name, record_gif_fps=gif_fps))
        return SampleList(samples)
    
    def eval_success_rate(self, test_agent):
        assert type(test_agent) is list
        success_thresh = test_agent[0]['filter_demos'].get('success_upper_bound', 0.05)
        state_idx = np.array(list(test_agent[0]['filter_demos'].get('state_idx', range(4, 7))))
        train_dists = []
        val_dists = []
        for i in xrange(len(test_agent)):
            agent = test_agent[i]['type'](test_agent[i])
            conditions = self.demos[i]['demoConditions']
            target_eepts = np.array(test_agent[i]['target_end_effector'])[conditions]
            if len(target_eepts.shape) == 1:
                target_eepts = np.expand_dims(target_eepts, axis=0)
            target_eepts = target_eepts[:, :3]
            if self.val_demos and i in self.val_idx:
                # Sample on validation conditions.
                val_sample_list = self.sample(agent, i, conditions, N=1, testing=True)
                # Calculate val distances
                X_val = val_sample_list.get_X()
                val_dists.extend([np.nanmin(np.linalg.norm(X_val[j, :, state_idx].T - target_eepts[j], axis=1)) \
                                    for j in xrange(X_val.shape[0])])
            else:
                # Sample on training conditions.
                train_sample_list = self.sample(agent, i, conditions, N=1)
                # Calculate train distances
                X_train = train_sample_list.get_X()
                train_dists.extend([np.nanmin(np.linalg.norm(X_train[j, :, state_idx].T - target_eepts[j], axis=1)) \
                                    for j in xrange(X_train.shape[0])])

        import pdb; pdb.set_trace()
        print "Training success rate is %.5f" % (np.array(train_dists) <= success_thresh).mean()
        if self.val_demos:
            print "Validation success rate is %.5f" % (np.array(val_dists) <= success_thresh).mean()
