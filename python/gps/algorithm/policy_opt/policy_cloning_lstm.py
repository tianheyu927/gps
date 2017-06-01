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
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict, extract_demo_dict_multi
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

ANNEAL_INTERVAL = 20000 # this used to be 5000

class PolicyCloningLSTM(PolicyOptTf):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.graph = tf.Graph()
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            if not self._hyperparams.get('uses_vision', False):
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.08)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
            else:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
                # self._sess = tf.Session(graph=self.graph)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
        else:
            self._sess = tf.Session(graph=self.graph)
        self.act_op = None  # mu_hat
        self.feat_op = None # features
        self.image_op = None # image
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.feat_vals = None
        self.debug = None
        self.debug_vals = None
        self.bias = None
        self.scale = None
        self.norm_type = self._hyperparams.get('norm_type', False)
        self._hyperparams['network_params'].update({'norm_type': self.norm_type})
        self._hyperparams['network_params'].update({'decay': self._hyperparams.get('decay', 0.99)})
        # MAML hyperparams
        self.update_batch_size = self._hyperparams.get('update_batch_size', 1)
        self.meta_batch_size = self._hyperparams.get('meta_batch_size', 10)
        self.num_updates = self._hyperparams.get('num_updates', 1)
        self.meta_lr = self._hyperparams.get('lr', 1e-3) #1e-3
        self.weight_decay = self._hyperparams.get('weight_decay', 0.005)
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
            test_agent = hyperparams['agent']
            # test_agent = hyperparams['agent'][:1200]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-100:])
            # test_agent = hyperparams['agent'][:1500]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-150:])
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file']
        # demo_file = hyperparams['demo_file'][:1200]
        # demo_file.extend(hyperparams['demo_file'][-100:])
        # demo_file = hyperparams['demo_file'][:1500]
        # demo_file.extend(hyperparams['demo_file'][-150:])
        
        if hyperparams.get('agent', False):
            self.restore_iter = hyperparams.get('restore_iter', 0)
            self.extract_supervised_data(demo_file)

        self.init_network(self.graph)
        self.init_network(self.graph, prefix='Validation_')
        
        with self.graph.as_default():
            self.saver = tf.train.Saver()
        
        with self.graph.as_default():
            init_op = tf.global_variables_initializer()
        self.run(init_op)
        with self.graph.as_default():
            tf.train.start_queue_runners(sess=self._sess)
        
        if self.restore_iter > 0:
            self.restore_model(hyperparams['save_dir'] + '_%d' % self.restore_iter)
            # import pdb; pdb.set_trace()
            self.update()
            # TODO: also implement resuming training from restored model
        else:
            self.update()
            # import pdb; pdb.set_trace()
        # os._exit(1) # debugging

        # Replace input tensors to be placeholders for rolling out learned policy
        self.init_network(self.graph, prefix='Testing')
        # Initialize policy with noise
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        # use test action for policy action
        self.policy = TfPolicy(dU, self.obs_tensor, self.test_act_op, self.feat_op, self.image_op,
                               0.5*np.ones(dU), self._sess, self.graph, self.device_string, copy_param_scope=self._hyperparams['copy_param_scope'])
        self.policy.scale = self.scale
        self.policy.bias = self.bias
        self.policy.x_idx = self.x_idx
        self.policy.img_idx = self.img_idx
        
        self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self, graph, input_tensors=None, prefix='Training_'):
        """ Helper method to initialize the tf networks used """
        with graph.as_default():
            if 'Training' in prefix:
                image_tensors = self.make_batch_tensor(self._hyperparams['network_params'])
            elif 'Validation' in prefix:
                image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], train=False)
            else:
                image_tensors = None
            with Timer('building TF network'):
                result = self.construct_model(input_tensors=image_tensors, prefix=prefix, dim_input=self._dO, dim_output=self._dU,
                                          network_config=self._hyperparams['network_params'])
            # outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test = result
            test_output, loss, flat_img_input= result
            if 'Testing' in prefix:
                self.obs_tensor = self.obs
                self.state_tensor = self.state
                self.action_tensor = self.action
                self.test_act_op = test_output # post-update output
                self.image_op = flat_img_input

            total_loss = tf.reduce_sum(loss) / tf.to_float(self.meta_batch_size)

            if 'Training' in prefix:
                self.total_loss = total_loss
            elif 'Validation' in prefix:
                self.val_total_loss = total_loss

            # Initialize solver
            # mom1, mom2 = 0.9, 0.999 # adam defaults
            # self.global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(self.meta_lr, self.global_step, ANNEAL_INTERVAL, 0.5, staircase=True)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_losses2[self.num_updates - 1], global_step=self.global_step)
            if 'Training' in prefix:
                self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_loss)
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'loss', self.total_loss)] # tf.scalar_summary('Learning rate', learning_rate)
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                self.train_summ_op = tf.summary.merge(summ)
            elif 'Validation' in prefix:
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'loss', self.val_total_loss)] # tf.scalar_summary('Learning rate', learning_rate)
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                self.val_summ_op = tf.summary.merge(summ)

    def construct_image_input(self, nn_input, x_idx, img_idx, network_config=None):
        state_input = nn_input[:, 0:x_idx[-1]+1]
        flat_image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]
    
        # image goes through 3 convnet layers
        num_filters = network_config['num_filters']
    
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        image_input = tf.reshape(flat_image_input, [-1, num_channels, im_width, im_height])
        image_input = tf.transpose(image_input, perm=[0,3,2,1])
        return image_input, flat_image_input, state_input
    
    def construct_weights(self, dim_input=27, dim_output=7, network_config=None):
        n_layers = network_config.get('n_layers', 4) # TODO TODO this used to be 3.
        layer_size = network_config.get('layer_size', 100)  # TODO TODO This used to be 20.
        dim_hidden = (n_layers - 1)*[layer_size]
        dim_hidden.append(dim_output)
        filter_size = 3 # used to be 2
        num_filters = network_config['num_filters']
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        is_dilated = self._hyperparams.get('is_dilated', False)
        lstm_size = self._hyperparams.get('lstm_size', 512)
        weights = {}
        if is_dilated:
            self.conv_out_size = int(im_width*im_height*num_filters[2])
        else:
            self.conv_out_size = int(im_width/(8.0)*im_height/(8.0)*num_filters[2]) # 3 layers each with stride 2
        # self.conv_out_size = int(im_width/(16.0)*im_height/(16.0)*num_filters[3]) # 3 layers each with stride 2

        # conv weights
        # weights['wc1'] = get_he_weights([filter_size, filter_size, num_channels, num_filters[0]], name='wc1') # 5x5 conv, 1 input, 32 outputs
        # weights['wc2'] = get_he_weights([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2') # 5x5 conv, 32 inputs, 64 outputs
        # weights['wc3'] = get_he_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3') # 5x5 conv, 32 inputs, 64 outputs
        weights['wc1'] = init_conv_weights_xavier([filter_size, filter_size, num_channels, num_filters[0]], name='wc1') # 5x5 conv, 1 input, 32 outputs
        weights['wc2'] = init_conv_weights_xavier([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2') # 5x5 conv, 32 inputs, 64 outputs
        weights['wc3'] = init_conv_weights_xavier([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3') # 5x5 conv, 32 inputs, 64 outputs
        # weights['wc4'] = init_conv_weights_xavier([filter_size, filter_size, num_filters[2], num_filters[3]], name='wc4') # 5x5 conv, 32 inputs, 64 outputs

        weights['bc1'] = init_bias([num_filters[0]], name='bc1')
        weights['bc2'] = init_bias([num_filters[1]], name='bc2')
        weights['bc3'] = init_bias([num_filters[2]], name='bc3')
        # weights['bc4'] = init_bias([num_filters[3]], name='bc4')
        
        # LSTM cell
        self.lstm = tf.nn.rnn_cell.BasicRNNCell(lstm_size)
        self.lstm_initial_state = safe_get('lstm_initial_state', initializer=tf.zeros([self.update_batch_size, self.lstm.state_size], dtype=tf.float32))
        
        # fc weights
        # in_shape = 40 # dimension after feature computation
        in_shape = self.lstm.output_size + self.lstm.state_size # hard-coded for last conv layer output
        if self._hyperparams.get('color_hints', False):
            in_shape += 3
        for i in xrange(n_layers):
            weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
            # weights['w_%d' % i] = init_fc_weights_xavier([in_shape, dim_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
            in_shape = dim_hidden[i]
        return weights
    
    def vbn(self, tensor, name, update=False):
        VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor, update=update)

    def forward(self, image_input, state_input, weights, update=False, is_training=True, network_config=None):
        n_layers = network_config.get('n_layers', 4) # 3
        norm_type = self.norm_type
        decay = network_config.get('decay', 0.9)
        use_dropout = self._hyperparams.get('use_dropout', False)
        prob = self._hyperparams.get('keep_prob', 0.5)
        is_dilated = self._hyperparams.get('is_dilated', False)
        # conv_layer_0, _, _ = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, conv_id=0, is_training=is_training)
        # conv_layer_1, _, _ = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2']), norm_type=norm_type, decay=decay, conv_id=1, is_training=is_training)
        # conv_layer_2, moving_mean, moving_variance = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3']), norm_type=norm_type, decay=decay, conv_id=2, is_training=is_training)            
        if norm_type == 'vbn':
            if not use_dropout:
                conv_layer_0 = self.vbn(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_1', update=update)
                conv_layer_1 = self.vbn(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_2', update=update)
                conv_layer_2 = self.vbn(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_3', update=update)       
            else:
                conv_layer_0 = dropout(self.vbn(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_1', update=update), keep_prob=prob, is_training=is_training, name='dropout_1')
                conv_layer_1 = dropout(self.vbn(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_2', update=update), keep_prob=prob, is_training=is_training, name='dropout_2')
                conv_layer_2 = dropout(self.vbn(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3'], strides=[1,2,2,1], is_dilated=is_dilated), name='vbn_3', update=update), keep_prob=prob, is_training=is_training, name='dropout_3')       
        else:
            if True:#not use_dropout:
                conv_layer_0 = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=0, is_training=is_training)
                conv_layer_1 = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=1, is_training=is_training)
                conv_layer_2 = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=2, is_training=is_training)       
                # conv_layer_3 = norm(conv2d(img=conv_layer_2, w=weights['wc4'], b=weights['bc4'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, conv_id=3, is_training=is_training)       
            else:
                conv_layer_0 = dropout(norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=0, is_training=is_training), keep_prob=prob, is_training=is_training, name='dropout_1')
                conv_layer_1 = dropout(norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=1, is_training=is_training), keep_prob=prob, is_training=is_training, name='dropout_2')
                conv_layer_2 = dropout(norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3'], strides=[1,2,2,1], is_dilated=is_dilated), norm_type=norm_type, decay=decay, conv_id=2, is_training=is_training), keep_prob=prob, is_training=is_training, name='dropout_3')       

        conv_out_flat = tf.reshape(conv_layer_2, [-1, self.conv_out_size])
        # conv_out_flat = tf.reshape(conv_layer_3, [-1, self.conv_out_size])
        # if use_dropout:
            # conv_out_flat = dropout(conv_out_flat, keep_prob=0.8, is_training=is_training, name='dropout_input')
        lstm_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])
        
        lstm_input = tf.reshape(lstm_input, [-1, self.T, self.conv_out_size+len(self.x_idx)])
        
        # LSTM forward
        state = self.lstm_initial_state
        lstm_output = []
        with tf.variable_scope('LSTM', reuse=None) as lstm_scope:
            for t in xrange(self.T):
                try:
                    fc_input, next_state = self.lstm(lstm_input[:, t, :], state)
                except ValueError:
                    lstm_scope.reuse_variables()
                    fc_input, next_state = self.lstm(lstm_input[:, t, :], state)
                fc_output = tf.concat(1, [fc_input, next_state])
                for i in xrange(n_layers):
                    fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
                    if i != n_layers - 1:
                        fc_output = tf.nn.relu(fc_output)
                        if use_dropout:
                            fc_output = dropout(fc_output, keep_prob=prob, is_training=is_training)
                fc_output = tf.expand_dims(fc_output, axis=1) # N x 1 x dU
                lstm_output.append(fc_output)
                state = next_state
        lstm_output = tf.reshape(tf.concat(1, lstm_output), [-1, self._dU])
        return lstm_output

    def construct_model(self, input_tensors=None, prefix='Training_', dim_input=27, dim_output=7, batch_size=25, network_config=None):
        """
        An example a network in theano that has both state and image inputs, with the feature
        point architecture (spatial softmax + expectation).
        Args:
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        # List of indices for state (vector) data and image (tensor) data in observation.
        x_idx, img_idx, i = [], [], 0
        for sensor in network_config['obs_include']:
            dim = network_config['sensor_dims'][sensor]
            if sensor in network_config['obs_image_data']:
                img_idx = img_idx + list(range(i, i+dim))
            else:
                x_idx = x_idx + list(range(i, i+dim))
            i += dim
        
        if input_tensors is None:
            self.obs = obs = tf.placeholder(tf.float32, name='obs') # meta_batch_size x update_batch_size x dim_input
        else:
            self.obs = obs = input_tensors # meta_batch_size x update_batch_size x dim_input
        if 'Training' in prefix:
            self.state = state = tf.placeholder(tf.float32, name='state')
            self.action = action = tf.placeholder(tf.float32, name='action')
            self.reference_tensor = reference_tensor = tf.placeholder(tf.float32, [self.T, self._dO], name='reference')
            if self.norm_type:
                self.phase = tf.placeholder(tf.bool, name='phase')
        else:
            state = self.state
            action = self.action
            reference_tensor = self.reference_tensor
        
        network_input = tf.concat(2, [obs, state])

        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            # TODO: since we flip to reuse automatically, this code below is unnecessary
            if 'weights' not in dir(self):
                self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
                self.sorted_weight_keys = natsorted(self.weights.keys())
            else:
                training_scope.reuse_variables()
                weights = self.weights
            if self._hyperparams.get('color_hints', False):
                self.color_hints = tf.maximum(tf.minimum(safe_get('color_hints', initializer=0.5*tf.ones([3], dtype=tf.float32)), 0.0), 1.0)
            
            # TODO: add variable that indicates the color?
            
            def batch_metalearn(inp, update=False):
                network_input, action = inp #image input
                network_input = tf.reshape(network_input, [-1, dim_input])
                action = tf.reshape(action, [-1, dim_output])
                
                # Convert to image dims
                network_input, flat_img_input, state_input = self.construct_image_input(network_input, x_idx, img_idx, network_config=network_config)
                if self._hyperparams.get('color_hints', False):
                    color_hints = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputa)), range(3)))
                    color_hints += self.color_hints
                    state_input_new = tf.concat(1, [state_input, color_hints])
                else:
                    state_input_new = state_input

                if 'Training' in prefix:
                    # local_outputa, fp, moving_mean, moving_variance = self.forward(inputa, state_inputa, weights, network_config=network_config)
                    local_output = self.forward(network_input, state_input_new, weights, network_config=network_config)
                else:
                    # local_outputa, _, moving_mean_test, moving_variance_test = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
                    local_output = self.forward(network_input, state_input_new, weights, update=update, is_training=False, network_config=network_config)
                local_loss = euclidean_loss_layer(local_output, action, None, behavior_clone=True)
                local_fn_output = [local_output, local_loss, flat_img_input]
                return local_fn_output

        if self.norm_type:
            # initialize batch norm vars.
            # TODO: figure out if this line of code is necessary
            if self.norm_type == 'vbn':
                # Initialize VBN
                # Uncomment below to update the mean and mean_sq of the reference batch
                self.reference_out = batch_metalearn((reference_tensor, action[0]), update=True)[0]
                # unused = batch_metalearn((reference_tensor, reference_tensor, actionb[0], actionb[0]), update=False)[3]
            else:
                unused = batch_metalearn((network_input[0], action[0]))
        
        out_dtype = [tf.float32, tf.float32, tf.float32]
        result = tf.map_fn(batch_metalearn, elems=(network_input, action), dtype=out_dtype)
        print 'Done with map.'
        return result
    
    def extract_supervised_data(self, demo_file):
        """
            Load demos into memory.
            Args:
                demo_file: list of demo files where each file contains demos of one task.
            Return:
                total_train_obs: all training observations
                total_train_U: all training actions
        """
        demos = extract_demo_dict(demo_file)
        n_folders = len(demos.keys())
        n_val = self._hyperparams['n_val'] # number of demos for testing
        N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.iteritems())
        print "Number of demos: %d" % N_demos
        idx = np.arange(n_folders)
        # shuffle(idx)
        # self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
        # mask = np.array([(i in self.val_idx) for i in idx])
        # self.train_idx = np.sort(idx[~mask])
        if n_val != 0:
            self.val_idx = idx[-n_val:]
            self.train_idx = idx[:-n_val]
        else:
            self.train_idx = idx
            self.val_idx = []
        # Normalizing observations
        with Timer('Normalizing states'):
            if self.scale is None or self.bias is None:
                states = np.vstack((demos[i]['demoX'] for i in self.train_idx)) # hardcoded here to solve the memory issue
                states = states.reshape(-1, len(self.x_idx))
                # 1e-3 to avoid infs if some state dimensions don't change in the
                # first batch of samples
                self.scale = np.diag(
                    1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                self.bias = - np.mean(
                    states.dot(self.scale), axis=0)
                for key in demos.keys():
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.x_idx))
                    demos[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                    demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.x_idx))
        self.demos = demos
        self.val_demos = {key: demos[key].copy() for key in self.val_idx}
        self.train_demos = {key: demos[key].copy() for key in self.train_idx}
        self.train_img_folders = {i: os.path.join(self.demo_gif_dir, 'color_%d' % i) for i in self.train_idx}
        self.val_img_folders = {i: os.path.join(self.demo_gif_dir, 'color_%d' % i) for i in self.val_idx}
        with Timer('Generating batches for each iteration'):
            self.generate_batches()
    
    def generate_reference_batch(self, obs):
        """
            Generate the reference batch for VBN. The reference batch is generated randomly
            at each time step.
            Args:
                obs: total observations.
        """
        assert self.norm_type == 'vbn'
        self.reference_batch = np.zeros((self.T, self._dO))
        for t in xrange(self.T):
            idx = np.random.choice(np.arange(obs.shape[0]))
            self.reference_batch[t, :] = obs[idx, t, :]
        self.policy.reference_batch = self.reference_batch
        # TODO: make this compatible in new code.
        # self.policy.reference_batch_X = self.reference_batch[:, self.x_idx]
        # self.policy.reference_batch_O = self.reference_batch[:, self.img_idx]
    
    def generate_batches(self):
        # TODO: generate all batches of images, states, and actions before training
        TEST_PRINT_INTERVAL = 500
        TOTAL_ITERS = self._hyperparams['iterations']
        # VAL_ITERS = int(TOTAL_ITERS / 500)
        self.all_training_filenames = []
        self.all_val_filenames = []
        self.training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
        self.val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, TOTAL_ITERS/TEST_PRINT_INTERVAL)}
        for itr in xrange(TOTAL_ITERS):
            sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
            for idx in sampled_train_idx:
                sampled_folder = self.train_img_folders[idx]
                image_paths = natsorted(os.listdir(sampled_folder))
                assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size, replace=True)
                sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                self.all_training_filenames.extend(sampled_images)
                self.training_batch_idx[itr][idx] = sampled_image_idx
            if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                for idx in sampled_val_idx:
                    sampled_folder = self.val_img_folders[idx]
                    image_paths = natsorted(os.listdir(sampled_folder))
                    assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                    sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size, replace=True)
                    sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                    self.all_val_filenames.extend(sampled_images)
                    self.val_batch_idx[itr][idx] = sampled_image_idx
        # import pdb; pdb.set_trace()

    def make_batch_tensor(self, network_config, train=True):
        # TODO: load images using tensorflow fileReader and gif decoder
        if train:
            all_filenames = self.all_training_filenames
        else:
            all_filenames = self.all_val_filenames
        
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print 'Generating image processing ops'
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.T, im_height, im_width, num_channels))
        image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        image = tf.cast(image, tf.float32)
        image /= 255.0
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        batch_image_size = self.update_batch_size * self.meta_batch_size
        print 'Batching images'
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []
        for i in xrange(self.meta_batch_size):
            image = images[i*self.update_batch_size:(i+1)*self.update_batch_size]
            image = tf.reshape(image, [self.update_batch_size*self.T, -1])
            all_images.append(image)
        return tf.pack(all_images)
        
    def generate_data_batch(self, itr, train=True):
        if train:
            demos = self.train_demos
            idxes = self.training_batch_idx[itr]
        else:
            demos = self.val_demos
            idxes = self.val_batch_idx[itr]
        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        U = [demos[k]['demoU'][v].reshape(update_batch_size*self.T, -1) for k, v in idxes.items()]
        X = [demos[k]['demoX'][v].reshape(update_batch_size*self.T, -1) for k, v in idxes.items()]
        U = np.array(U)
        X = np.array(X)
        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.x_idx)
        return X, U
    
    def update(self):
        """
        Update (train) policy.
        """
        # TODO: Do we need to normalize the observations?
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        TOTAL_ITERS = self._hyperparams['iterations']
        prelosses, postlosses = [], []
        log_dir = self._hyperparams['log_dir']# + '_%s' % datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        # log_dir = self._hyperparams['log_dir'] # for debugging
        save_dir = self._hyperparams['save_dir'] #'_model' #'_model_ln'
        train_writer = tf.train.SummaryWriter(log_dir, self.graph)
        losses = []
        # actual training.
        with Timer('Training'):
            if self.restore_iter == 0:
                training_range = range(TOTAL_ITERS)
            else:
                training_range = range(self.restore_iter+1, self.restore_iter+1+TOTAL_ITERS)
            for itr in training_range:
                # TODO: need to make state and obs compatible
                state, tgt_mu = self.generate_data_batch(itr)
                feed_dict = {self.state: state,
                            self.action: tgt_mu}
                input_tensors = [self.train_op]
                # if self.use_batchnorm:
                #     feed_dict[self.phase] = 1
                if self.norm_type == 'vbn':
                    feed_dict[self.reference_tensor] = self.reference_batch
                    input_tensors.append(self.reference_out)
                if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
                    input_tensors.extend([self.train_summ_op, self.total_loss])
                result = self.run(input_tensors, feed_dict=feed_dict)
    
                if itr != 0 and itr % SUMMARY_INTERVAL == 0:
                    losses.append(result[-1])
                    train_writer.add_summary(result[-2], itr)

                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print 'Iteration %d: average loss is %.2f' % (itr, np.mean(losses))
                    losses = []
    
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    if len(self.val_idx) > 0:
                        input_tensors = [self.val_summ_op, self.val_total_loss]
                        val_state, val_act = self.generate_data_batch(itr, train=False)
                        feed_dict = {self.state: val_state,
                                    self.action: val_act}
                        # if self.use_batchnorm:
                        #     feed_dict[self.phase] = 0
                        if self.norm_type == 'vbn':
                            feed_dict[self.reference_tensor] = self.reference_batch
                            input_tensors.append(self.reference_out)
                        results = self.run(input_tensors, feed_dict=feed_dict)
                        train_writer.add_summary(results[0], itr)
                        print 'Test results: average loss is %.2f' % (np.mean(results[-1]))
                
                if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
                    self.save_model(save_dir + '_%d' % itr)

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

    def sample(self, agent, idx, conditions, N=1, testing=False):
        samples = []
        for i in xrange(len(conditions)):
            for j in xrange(N):
                if 'record_gif' in self._hyperparams:
                    gif_config = self._hyperparams['record_gif']
                    if j < gif_config.get('gifs_per_condition', float('inf')):
                        gif_fps = gif_config.get('fps', None)
                        if testing:
                            gif_dir = gif_config.get('test_gif_dir', self._hyperparams['plot_dir'])
                        else:
                            gif_dir = gif_config.get('gif_dir', self._hyperparams['plot_dir'])
                        gif_dir = gif_dir + 'color_%d/' % idx
                        mkdir_p(gif_dir)
                        gif_name = os.path.join(gif_dir,'cond%d.samp%d.gif' % (conditions[i], j))
                    else:
                        gif_name=None
                        gif_fps = None
                else:
                    gif_name=None
                    gif_fps = None
                samples.append(agent.sample(
                    self.policy, conditions[i],
                    verbose=False, save=False, noisy=False,
                    record_gif=gif_name, record_gif_fps=gif_fps, task_idx=idx))
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
            if i in self.val_idx:
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
        print "Validation success rate is %.5f" % (np.array(val_dists) <= success_thresh).mean()
