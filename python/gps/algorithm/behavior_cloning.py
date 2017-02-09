""" This file defines the behavior cloning algorithm class. """

import abc
import copy

import numpy as np
from numpy.linalg import LinAlgError
import tensorflow as tf

from gps.algorithm.config import ALG_BC
from gps.utility import ColorLogger
from gps.utility.general_utils import Timer

LOGGER = ColorLogger(__name__)

class BehaviorCloning(object):
    """ the behavior cloning class. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_BC)
        config.update(hyperparams)
        self._hyperparams = config

        if 'train_conditions' in self._hyperparams:
            self._cond_idx = self._hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = self._hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO
        del self._hyperparams['agent']  # Don't want to pickle this.

        if self._hyperparams['global_cost']:
            if type(hyperparams['cost']) == list:
                self.cost = [
                    hyperparams['cost'][i]['type'](hyperparams['cost'][i])
                    for i in range(hyperparams['conditions'])]
            else:
                self.cost = self._hyperparams['cost']['type'](self._hyperparams['cost'])
        else:
            self.cost = [
                self._hyperparams['cost']['type'](self._hyperparams['cost'])
                for _ in range(self.M)
            ]
        if type(self._hyperparams['cost']) is dict and self._hyperparams['cost'].get('agent', False):
            del self._hyperparams['cost']['agent']

        with Timer('build policy opt'):
            self.policy_opt = self._hyperparams['policy_opt']['type'](
                self._hyperparams['policy_opt'], self.dO, self.dU
            )
        del self._hyperparams['policy_opt']['agent']
        