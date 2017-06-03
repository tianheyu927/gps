import pickle
import os
import uuid

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy


class TfPolicyLSTM(TfPolicy):
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
    def __init__(self, dU, obs_tensor, state_tensor, act_op, feat_op, image_op, var, sess, graph, device_string, copy_param_scope=None):
        super(TfPolicyLSTM, self).__init__(dU, obs_tensor, act_op, feat_op, image_op, var, sess, graph, device_string, copy_param_scope=None)
        self.state_tensor = state_tensor
        
    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """

        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        state = obs[:, self.x_idx]
        assert hasattr(self, 'img_idx')
        assert hasattr(self, 'T')
        assert hasattr(self, 'update_batch_size')
        obs = obs[:, self.img_idx]
        tiled_obs = np.tile(np.expand_dims(obs, axis=0), (1, self.update_batch_size*self.T, 1))
        tiled_state = np.tile(np.expand_dims(state, axis=0), (1, self.update_batch_size*self.T, 1))
        action_mean = self.run(self.act_op, feed_dict={self.obs_tensor: tiled_obs,
                                                       self.state_tensor: tiled_state})[:, 0, :]
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return np.squeeze(action_mean), np.squeeze(u)  # the DAG computations are batched by default, but we use batch size 1.
