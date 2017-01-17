""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.utility.demo_utils import xu_to_sample_list, extract_demos
from gps.utility.general_utils import BatchSampler
from gps.utility.general_utils import Timer

class PolicyCloningTf(PolicyOptTf):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams, dO, dU):
        super(PolicyCloningTf, self).__init__(hyperparams, dO, dU)

        if hyperparams.get('agent', False):
            demo_agent = hyperparams['agent']  # Required for sample packing
            demo_agent = demo_agent['type'](demo_agent)
        demo_file = hyperparams['demo_file']

        if hyperparams.get('agent', False):
            train_samples, test_samples = self.extract_supervised_data(demo_agent, demo_file) 
            self.update(train_samples.get_X(), train_samples.get_U(), 1.0, 1.0, behavior_clone=True)
        self.sup_samples = train_samples
        self.sup_test_samples = test_samples

        self.demo_agent = None  # don't pickle agent
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']


    def extract_supervised_data(self, demo_agent, demo_file):
        X, U, O, cond = extract_demos(demo_file)
        n_test = 1 # number of demos for testing
        testX = X[-n_test:]
        testU = U[-n_test:]
        X = X[:-n_test]
        U = U[:-n_test]
        train_samples = xu_to_sample_list(demo_agent, X, U)
        test_samples = xu_to_sample_list(demo_agent, testX, testU)
        return train_samples, test_samples
