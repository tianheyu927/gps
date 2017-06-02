import pickle
import os
import uuid

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicyMAML(Policy):
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
    # def __init__(self, dU, obs_tensor, act_op, reference_op, reference_out, feat_op, image_op, norm_type, var, sess, graph, device_string, copy_param_scope=None):
    def __init__(self, dU, obsa, statea, actiona, obsb, stateb, act_op, reference_op, reference_out, feat_op, image_op, norm_type, var, sess, graph, device_string, copy_param_scope=None):
        Policy.__init__(self)
        self.dU = dU
        # self.obs_tensor = obs_tensor
        self.obsa = obsa
        self.statea = statea
        self.actiona = actiona
        self.obsb = obsb
        self.stateb = stateb
        self.act_op = act_op
        self.feat_op = feat_op
        self.image_op = image_op
        self.norm_type = norm_type
        self._sess = sess
        self.graph = graph
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.img_idx = None
        self.reference_op = reference_op
        self.reference_out = reference_out

        if copy_param_scope:
            with self.graph.as_default():
                self.copy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=copy_param_scope)
                self.copy_params_assign_placeholders = [tf.placeholder(tf.float32, shape=param.get_shape()) for
                                                          param in self.copy_params]

                self.copy_params_assign_ops = [tf.assign(self.copy_params[i],
                                                         self.copy_params_assign_placeholders[i])
                                                 for i in range(len(self.copy_params))]

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
        obs[:, self.img_idx] /= 255.0
        state = obs[:, self.x_idx]
        obs = obs[:, self.img_idx]
        # This following code seems to be buggy
        # TODO: figure out why this doesn't work
        # if t == 0:
        #     assert hasattr(self, 'fast_weights_value')
        #     self.set_copy_params(self.fast_weights_value[idx])
        # if self.batch_norm:
        #     action_mean = self.run(self.act_op, feed_dict={self.inputa: obs, self.phase_op: 0}) # testing
        # else:
        if self.norm_type == 'vbn':
            assert hasattr(self, 'reference_batch')
            if self.reference_out is not None:
                action_mean = self.run([self.reference_out, self.act_op], feed_dict={self.obsa: obs, self.statea: state, self.reference_op: self.reference_batch})[1]
            else:
                action_mean = self.run(self.act_op, feed_dict={self.obsa: obs, self.statea: state, self.reference_op: self.reference_batch})
        # action_mean = self.run(self.act_op, feed_dict={self.inputa: obs}) #need to set act_op to be act_op_b if using set_params
        else:
            assert hasattr(self, 'selected_demoO')
            # import pdb; pdb.set_trace()
            selected_obs = self.selected_demoO[idx].astype(np.float32)
            selected_obs /= 255.0
            action_mean = self.run(self.act_op, feed_dict={self.obsa: selected_obs,
                                                          self.statea: self.selected_demoX[idx],
                                                          self.actiona: self.selected_demoU[idx],
                                                          self.obsb: np.expand_dims(obs, axis=0),
                                                          self.stateb: np.expand_dims(state, axis=0)})
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return np.squeeze(action_mean), np.squeeze(u)  # the DAG computations are batched by default, but we use batch size 1.

    def run(self, op, feed_dict=None):
        with tf.device(self.device_string):
            with self.graph.as_default():
                result = self._sess.run(op, feed_dict=feed_dict)
        return result

    def get_features(self, obs):
        """
        Return the image features for an observation.
        Args:
            obs: Observation vector.
        """
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        # Assume that features don't depend on the robot config, so don't normalize by scale and bias.
        feat = self.run(self.feat_op, feed_dict={self.obs_a: obs[:, self.img_idx], self.statea: obs[:, self.x_idx]})
        return feat  # This used to be feat[0] because we would only ever call it with a batch size of 1. Now this isn't true.

    def get_image_features(self, image):
        """
        Return the image features for an image (not including other obs data).
        Args:
            image: Image vector.
        """
        if len(image.shape) == 1:
            image = np.expand_dims(image, axis=0)
        feat = self.run(self.feat_op, feed_dict={self.image_op: image})
        return feat  # This used to be feat[0] because we would only ever call it with a batch size of 1. Now this isn't true.


    def get_copy_params(self):
        param_values = self.run(self.copy_params)
        return {self.copy_params[i].name:param_values[i] for i in range(len(self.copy_params))}

    def set_copy_params(self, param_values):
        value_list = [param_values[self.copy_params[i].name.split('/')[-1][:3]] for i in range(len(self.copy_params))]
        feeds = {self.copy_params_assign_placeholders[i]:value_list[i] for i in range(len(self.copy_params))}
        self.run(self.copy_params_assign_ops, feed_dict=feeds)


    def pickle_policy(self, deg_obs, deg_action, checkpoint_path, goal_state=None, should_hash=False):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        if should_hash is True:
            hash_str = str(uuid.uuid4())
            checkpoint_path += hash_str
        os.mkdir(checkpoint_path + '/')
        checkpoint_path += '/_pol'
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data', 'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string, 'goal_state': goal_state, 'x_idx': self.x_idx}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._sess, checkpoint_path + '_tf_data')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        #ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"))


        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        with graph.as_default():
            init_op = tf.initialize_all_variables()
            tf_map = tf_generator(dim_input=pol_dict['deg_obs'], dim_output=pol_dict['deg_action'],
                                  batch_size=1, network_config=network_config)
            sess.run(init_op)
            saver = tf.train.Saver()
            check_file = pol_dict['checkpoint_path_tf']
            saver.restore(sess, check_file)

            device_string = pol_dict['device_string']

            cls_init = cls(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op(), np.zeros((1,)),
                           sess, graph, device_string)
            cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
            cls_init.scale = pol_dict['scale']
            cls_init.bias = pol_dict['bias']
            cls_init.x_idx = pol_dict['x_idx']
            return cls_init

