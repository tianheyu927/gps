import pickle
import os
import uuid

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicyRandom(Policy):
    """
    A neural network policy implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """
    def __init__(self, dU, var, use_vision=True, scale=None, bias=None, x_idx=None, img_idx=None, std=0.1):
        Policy.__init__(self)
        self.dU = dU
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = scale  # must be set from elsewhere based on observations
        self.bias = bias
        self.x_idx = x_idx
        self.img_idx = img_idx
        self.std = std
        self.use_vision = use_vision

    def act(self, x, obs, t, noise, idx):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
            idx: The index of the task. Use this to get the demos.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if self.scale is not None and self.bias is not None:
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        if self.use_vision:
            obs[:, self.img_idx] /= 255.0
            state = obs[:, self.x_idx]
            obs = obs[:, self.img_idx]
        
        action_mean = np.random.normal(scale=self.std, size=(1, self.dU))
        
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return np.squeeze(action_mean), np.squeeze(u)  # the DAG computations are batched by default, but we use batch size 1.
