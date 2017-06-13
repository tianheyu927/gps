""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile
from datetime import datetime
from collections import OrderedDict

import numpy as np
import random
import matplotlib.pyplot as plt
# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.
import tensorflow as tf

try:
    import imageio
except ImportError:
    print 'imageio not found'
    imageio = None

from natsort import natsorted
from random import shuffle
from gps.algorithm.policy.tf_policy_random import TfPolicyRandom
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_cloning_maml import PolicyCloningMAML
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict#, extract_demo_dict_multi
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

ANNEAL_INTERVAL = 20000 # this used to be 5000

class PolicyCloningRandom(PolicyCloningMAML):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.graph = tf.Graph()
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            if not self._hyperparams.get('use_vision', True):
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
            else:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
                # self._sess = tf.Session(graph=self.graph)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
        else:
            self._sess = tf.Session(graph=self.graph)
        self.scale = None
        self.bias = None
        self.demo_gif_dir = self._hyperparams.get('demo_gif_dir', None)
        
        self.T = self._hyperparams.get('T', 50)
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i+dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i+dim))
            i += dim
        
        # For loading demos
        if hyperparams.get('agent', False):
            # test_agent = hyperparams['agent']
            test_agent = hyperparams['agent'][:300]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-100:])
            # test_agent = hyperparams['agent'][:1500]  # Required for sampling
            test_agent.extend(hyperparams['agent'][-150:])
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file']
        # demo_file = hyperparams['demo_file'][:1200]
        # demo_file.extend(hyperparams['demo_file'][-100:])
        demo_file = hyperparams['demo_file'][:300]
        demo_file.extend(hyperparams['demo_file'][-150:])
        
        if hyperparams.get('agent', False):
            self.restore_iter = hyperparams.get('restore_iter', 0)
            self.extract_supervised_data(demo_file)
        
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.policy = TfPolicyRandom(dU, self.var, use_vision=self._hyperparams.get('use_vision', True),
                                    scale=self.scale, bias=self.bias, x_idx=self.x_idx,
                                    img_idx=self.img_idx, std=self._hyperparams.get('std', 0.1))
        self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        self.policy.demos = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']
