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
from gps.algorithm.policy.tf_policy_lstm import TfPolicyLSTM
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_cloning_maml import PolicyCloningMAML
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict, extract_demo_dict_multi
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

ANNEAL_INTERVAL = 20000 # this used to be 5000

class PolicyCloningLSTM(PolicyCloningMAML):
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
            if not self._hyperparams.get('use_vision', False):
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
            else:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
                # self._sess = tf.Session(graph=self.graph)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
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
        self.reference_out = None
        self.norm_type = self._hyperparams.get('norm_type', False)
        self._hyperparams['network_params'].update({'norm_type': self.norm_type})
        self._hyperparams['network_params'].update({'decay': self._hyperparams.get('decay', 0.99)})
        # MAML hyperparams
        self.update_batch_size = self._hyperparams.get('update_batch_size', 1)
        self.test_batch_size = self._hyperparams.get('test_batch_size', 1)
        self.meta_batch_size = self._hyperparams.get('meta_batch_size', 10)
        self.num_updates = self._hyperparams.get('num_updates', 1)
        self.meta_lr = self._hyperparams.get('lr', 1e-3) #1e-3
        self.weight_decay = self._hyperparams.get('weight_decay', 0.005)
        self.demo_gif_dir = self._hyperparams.get('demo_gif_dir', None)
        self.gif_prefix = self._hyperparams.get('gif_prefix', 'color')
        
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
            # test_agent = hyperparams['agent'][:300]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-150:])
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file']
        # demo_file = hyperparams['demo_file'][:100]
        # demo_file.extend(hyperparams['demo_file'][-100:])
        # demo_file = hyperparams['demo_file'][:300]
        # demo_file.extend(hyperparams['demo_file'][-150:])
        
        if hyperparams.get('agent', False):
            self.restore_iter = hyperparams.get('restore_iter', 0)
            self.extract_supervised_data(demo_file)
            if not hyperparams.get('test', False):
                self.generate_batches()

        if not hyperparams.get('test', False):
            self.init_network(self.graph, restore_iter=self.restore_iter)
            self.init_network(self.graph, restore_iter=self.restore_iter, prefix='Validation_')
        else:
            self.init_network(self.graph, prefix='Testing')

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
            if not hyperparams.get('test', False):
                self.update()
            # TODO: also implement resuming training from restored model
        else:
            self.update()
            # import pdb; pdb.set_trace()
        if not hyperparams.get('test', False):
            os._exit(1) # debugging

        # Initialize policy with noise
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        # use test action for policy action
        self.policy = TfPolicyLSTM(dU, self.obsa, self.statea, self.actiona, self.obsb, self.stateb,
                               self.test_act_op, self.reference_tensor, self.reference_out,
                               self.feat_op, self.image_op, self.norm_type,
                               0.5*np.ones(dU), self._sess, self.graph, self.device_string,
                               use_vision=self._hyperparams.get('use_vision', True),
                            #   np.zeros(dU), self._sess, self.graph, self.device_string, 
                               copy_param_scope=self._hyperparams['copy_param_scope'])
        self.policy.scale = self.scale
        self.policy.bias = self.bias
        self.policy.init_stored_obs(self.T, self.update_batch_size, self.x_idx, self.img_idx)
        # Generate selected demos for preupdate pass during testing
        self.generate_testing_demos()
        self.eval_success_rate(test_agent)
        os._exit(1)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self, graph, input_tensors=None, restore_iter=0, prefix='Training_'):
        """ Helper method to initialize the tf networks used """
        with graph.as_default():
            if 'Training' in prefix:
                image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter)
            elif 'Validation' in prefix:
                image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter, train=False)
            else:
                image_tensors = None
            if image_tensors is not None:
                # image_tensors = tf.reshape(image_tensors, [self.meta_batch_size, (self.update_batch_size+1)*self.T, -1])
                # inputa = tf.slice(image_tensors, [0, 0, 0], [-1, self.update_batch_size*self.T, -1])
                # inputb = tf.slice(image_tensors, [0, self.update_batch_size*self.T, 0], [-1, -1, -1])
                inputa = image_tensors[:, :self.update_batch_size*self.T, :]
                inputb = image_tensors[:, self.update_batch_size*self.T:, :]
                input_tensors = {'inputa': inputa, 'inputb': inputb}
            else:
                input_tensors = None
            with Timer('building TF network'):
                result = self.construct_model(input_tensors=input_tensors, prefix=prefix, dim_input=self._dO, dim_output=self._dU,
                                          network_config=self._hyperparams['network_params'])
            # outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test = result
            test_output, loss, final_eept_loss, flat_img_inputb = result
            if 'Testing' in prefix:
                self.obs_tensor = self.obsa
                self.state_tensor = self.statea
                self.action_tensor = self.actiona
                self.test_act_op = test_output # post-update output
                toy_output_variable = tf.add(test_output, 0, name='output_action')
                self.image_op = flat_img_inputb

            total_loss = tf.reduce_sum(loss) / tf.to_float(self.meta_batch_size)
            total_final_eept_loss = tf.reduce_sum(final_eept_loss) / tf.to_float(self.meta_batch_size)

            if 'Training' in prefix:
                self.total_loss = total_loss
                self.total_final_eept_loss = total_final_eept_loss
                self.output = test_output
            elif 'Validation' in prefix:
                self.val_total_loss = total_loss
                self.val_total_final_eept_loss = total_final_eept_loss
            # self.val_total_loss1 = tf.contrib.copy_graph.get_copied_op(total_loss1, self.graph)
            # self.val_total_losses2 = [tf.contrib.copy_graph.get_copied_op(total_losses2[i], self.graph) for i in xrange(len(total_losses2))]
 
            # Initialize solver
            # mom1, mom2 = 0.9, 0.999 # adam defaults
            # self.global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(self.meta_lr, self.global_step, ANNEAL_INTERVAL, 0.5, staircase=True)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_losses2[self.num_updates - 1], global_step=self.global_step)
            # flat_img_inputb = tf.reshape(flat_img_inputb, [self.meta_batch_size, self.T, 3, 125, 125])
            # flat_img_inputb = tf.transpose(flat_img_inputb, perm=[0,1,4,3,2])
            if 'Training' in prefix:
                self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_loss)
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'loss', self.total_loss)] # tf.scalar_summary('Learning rate', learning_rate)
                summ.extend([tf.summary.scalar(prefix + 'final_eept_loss', self.total_final_eept_loss)])
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                # for i in xrange(self.meta_batch_size):
                #     summ.append(tf.summary.image('Training_image_%d' % i, flat_img_inputb[i]*255.0, max_outputs=50))
                self.train_summ_op = tf.summary.merge(summ)
            elif 'Validation' in prefix:
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'loss', self.val_total_loss)] # tf.scalar_summary('Learning rate', learning_rate)
                summ.extend([tf.summary.scalar(prefix + 'final_eept_loss', self.val_total_final_eept_loss)])
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                # for i in xrange(self.meta_batch_size):
                #     summ.append(tf.summary.image('Validation_image_%d' % i, flat_img_inputb[i]*255.0, max_outputs=50))
                self.val_summ_op = tf.summary.merge(summ)
    
    def construct_weights(self, dim_input=27, dim_output=7, network_config=None):
        lstm_size = self._hyperparams.get('lstm_size', 512)
        dropout_for_lstm = self._hyperparams.get('dropout_for_lstm', False)
        keep_prob = self._hyperparams.get('keep_prob', 0.9)
        if not dropout_for_lstm:
            keep_prob = 1.0
        ln_for_lstm = self._hyperparams.get('ln_for_lstm', False)
        lstm_activation_fn = self._hyperparams.get('lstm_activation_fn', tf.nn.tanh)
        weights = {}
        if self._hyperparams.get('use_vision', True):
            filter_size = network_config.get('filter_size', 3) # used to be 2
            num_filters = network_config['num_filters']
            strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
            im_height = network_config['image_height']
            im_width = network_config['image_width']
            num_channels = network_config['image_channels']
            is_dilated = self._hyperparams.get('is_dilated', False)
            use_fp = self._hyperparams.get('use_fp', False)
            pretrain = self._hyperparams.get('pretrain', False)
            train_conv1 = self._hyperparams.get('train_conv1', False)
            initialization = self._hyperparams.get('initialization', 'xavier')
            if pretrain:
                num_filters[0] = 64
                # strides[0] = [1, 1, 1, 1]
                # num_filters[1] = 64
                # strides[1] = [1, 1, 1, 1]
            pretrain_weight_path = self._hyperparams.get('pretrain_weight_path', '/home/kevin/gps/data/vgg19.pkl')
            n_conv_layers = len(num_filters)
            downsample_factor = 1
            for stride in strides:
                downsample_factor *= stride[1]
            if use_fp:
                self.conv_out_size = int(num_filters[-1]*2)
            elif is_dilated:
                self.conv_out_size = int(im_width*im_height*num_filters[-1])
            else:
                self.conv_out_size = int(np.ceil(im_width/(downsample_factor)))*int(np.ceil(im_height/(downsample_factor)))*num_filters[-1] # 3 layers each with stride 2            # self.conv_out_size = int(im_width/(16.0)*im_height/(16.0)*num_filters[3]) # 3 layers each with stride 2
    
            # conv weights
            fan_in = num_channels
            if self._hyperparams.get('use_img_context', False) or self._hyperparams.get('use_conv_context', False):
                fan_in += num_channels
            if self._hyperparams.get('use_conv_context', False):
                weights['img_context'] = safe_get('img_context', initializer=tf.zeros([im_height, im_width, num_channels], dtype=tf.float32))
            for i in xrange(n_conv_layers):
                # if not pretrain or (i != 0 and i != 1):
                if not pretrain or i != 0:
                    if self.norm_type == 'selu':
                        weights['wc%d' % (i+1)] = init_conv_weights_snn([filter_size, filter_size, fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    elif initialization == 'xavier':                
                        weights['wc%d' % (i+1)] = init_conv_weights_xavier([filter_size, filter_size, fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    elif initialization == 'random':
                        weights['wc%d' % (i+1)] = init_weights([filter_size, filter_size, fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    else:
                        raise NotImplementedError
                    weights['bc%d' % (i+1)] = init_bias([num_filters[i]], name='bc%d' % (i+1))
                    fan_in = num_filters[i]
                else:
                    import h5py
                    
                    assert num_filters[i] == 64
                    weights['wc%d' % (i+1)] = safe_get('wc%d' % (i+1), [filter_size, filter_size, fan_in, num_filters[i]], dtype=tf.float32, trainable=train_conv1)
                    weights['bc%d' % (i+1)] = safe_get('bc%d' % (i+1), [num_filters[i]], dtype=tf.float32, trainable=train_conv1)
                    pretrain_weight = h5py.File(pretrain_weight_path, 'r')
                    conv_weight = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_W_1:0' % (i+1)][...]
                    conv_bias = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_b_1:0' % (i+1)][...]
                    weights['wc%d' % (i+1)].assign(conv_weight)
                    weights['bc%d' % (i+1)].assign(conv_bias)
                    fan_in = conv_weight.shape[-1]

            # fc weights
            # in_shape = 40 # dimension after feature computation
            in_shape = self.conv_out_size
            if not self._hyperparams.get('no_state'):
                in_shape += len(self.x_idx) # hard-coded for last conv layer output
        else:
            in_shape = dim_input
        self.conv_out_size_final = in_shape
        
        # LSTM cell
        assert self.update_batch_size == self.test_batch_size
        self.lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_size, activation=lstm_activation_fn, layer_norm=ln_for_lstm, dropout_keep_prob=keep_prob)
        # import pdb; pdb.set_trace()
        lstm_c = safe_get('lstm_c', initializer=tf.zeros([self.update_batch_size, self.lstm.state_size[0]], dtype=tf.float32))
        lstm_h = safe_get('lstm_h', initializer=tf.zeros([self.update_batch_size, self.lstm.state_size[1]], dtype=tf.float32))
        self.lstm_initial_state = (lstm_c, lstm_h)
        
        # fc weights
        # in_shape = 40 # dimension after feature computation
        in_shape += self.lstm.output_size
        if self._hyperparams.get('learn_final_eept', False):
            final_eept_range = self._hyperparams['final_eept_range']
            final_eept_in_shape = in_shape
            if self._hyperparams.get('use_state_context', False):
                final_eept_in_shape += self._hyperparams.get('context_dim', 10)
            n_layers_ee = network_config.get('n_layers_ee', 1)
            layer_size_ee = [self._hyperparams.get('layer_size_ee', 40)]*(n_layers_ee-1)
            layer_size_ee.append(len(final_eept_range))
            for i in xrange(n_layers_ee):
                weights['w_ee_%d' % i] = init_weights([final_eept_in_shape, layer_size_ee[i]], name='w_ee_%d' % i)
                weights['b_ee_%d' % i] = init_bias([layer_size_ee[i]], name='b_ee_%d' % i)
                final_eept_in_shape = layer_size_ee[i]
            # if self._hyperparams.get('two_heads', False):# and self._hyperparams.get('no_final_eept', False):
            #     weights['w_ee_two_heads'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee_two_heads')
            #     weights['b_ee_two_heads'] = init_bias([len(final_eept_range)], name='b_ee_two_heads')
            in_shape += (len(final_eept_range))
        fc_weights = self.construct_fc_weights(in_shape, dim_output, network_config=network_config)
        weights.update(fc_weights)
        return weights

    def conv_forward(self, image_input, state_input, weights, update=False, testing=False, is_training=True, network_config=None):
        if self._hyperparams.get('use_vision', True):
            norm_type = self.norm_type
            decay = network_config.get('decay', 0.9)
            strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
            n_conv_layers = len(strides)
            use_dropout = self._hyperparams.get('use_dropout', False)
            prob = self._hyperparams.get('keep_prob', 0.5)
            is_dilated = self._hyperparams.get('is_dilated', False)
            # conv_layer_0, _, _ = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, id=0, is_training=is_training)
            # conv_layer_1, _, _ = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2']), norm_type=norm_type, decay=decay, id=1, is_training=is_training)
            # conv_layer_2, moving_mean, moving_variance = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3']), norm_type=norm_type, decay=decay, id=2, is_training=is_training)            
            conv_layer = image_input
            for i in xrange(n_conv_layers):
                if norm_type == 'vbn':
                    if not use_dropout:
                        conv_layer = self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), name='vbn_%d' % (i+1), update=update)
                    else:
                        conv_layer = dropout(self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), name='vbn_%d' % (i+1), update=update), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
                else:
                    if not use_dropout:
                        conv_layer = norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), norm_type=norm_type, decay=decay, id=i, is_training=is_training)
                    else:
                        conv_layer = dropout(norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), norm_type=norm_type, decay=decay, id=i, is_training=is_training), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
            if self._hyperparams.get('use_fp', False):
                _, num_rows, num_cols, num_fp = conv_layer.get_shape()
                num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
                x_map = np.empty([num_rows, num_cols], np.float32)
                y_map = np.empty([num_rows, num_cols], np.float32)
        
                for i in range(num_rows):
                    for j in range(num_cols):
                        x_map[i, j] = (i - num_rows / 2.0) / num_rows
                        y_map[i, j] = (j - num_cols / 2.0) / num_cols
        
                x_map = tf.convert_to_tensor(x_map)
                y_map = tf.convert_to_tensor(y_map)
        
                x_map = tf.reshape(x_map, [num_rows * num_cols])
                y_map = tf.reshape(y_map, [num_rows * num_cols])
        
                # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
                features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                                      [-1, num_rows*num_cols])
                softmax = tf.nn.softmax(features)
        
                fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
                fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
        
                conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
            else:
                conv_out_flat = tf.reshape(conv_layer, [-1, self.conv_out_size])
            # conv_out_flat = tf.reshape(conv_layer_3, [-1, self.conv_out_size])
            # if use_dropout:
                # conv_out_flat = dropout(conv_out_flat, keep_prob=0.8, is_training=is_training, name='dropout_input')
            if self._hyperparams.get('no_state'):
                fc_input = tf.add(conv_out_flat, 0)
            else:
                fc_input = tf.concat(axis=1, values=[conv_out_flat, state_input])
        else:
            fc_input = image_input
        return fc_input
        
    def lstm_forward(self, lstm_input, actions, is_training=True, network_config=None):
        use_dropout = self._hyperparams.get('use_dropout', False)
        prob = self._hyperparams.get('keep_prob', 0.5)
        # test_update_batch_size = self._hyperparams.get('test_update_batch_size', 1)
        if self._hyperparams.get('no_action', False):
            # dU = 0
            actions = tf.zeros_like(actions)
        # else:
        lstm_input = tf.concat(axis=1, values=[lstm_input, actions])
        dU = self._dU
        if self._hyperparams.get('learn_final_eept', False):
            final_eept_range = self._hyperparams['final_eept_range']
            dU = self._dU - len(final_eept_range)
        lstm_input = tf.reshape(lstm_input, [-1, self.T, self.conv_out_size_final + dU])
        
        # LSTM forward
        state = self.lstm_initial_state
        lstm_outputs = []
        with tf.variable_scope('LSTM', reuse=None) as lstm_scope:
            # for i in xrange(test_update_batch_size):
            #     lstm_outputs = []
            for t in xrange(self.T):
                try:
                    lstm_output, state = self.lstm(lstm_input[:, t, :], state)
                except ValueError:
                    lstm_scope.reuse_variables()
                    lstm_output, state = self.lstm(lstm_input[:, t, :], state)
                lstm_output = tf.nn.relu(lstm_output)
                if use_dropout:
                    lstm_output = dropout(lstm_output, keep_prob=prob, is_training=is_training)
                lstm_output = tf.expand_dims(lstm_output, axis=1)
                lstm_outputs.append(lstm_output)
                # lstm_outputs_total.append(tf.concat(1, lstm_outputs))
            # lstm_output = tf.reduce_mean(tf.concat(0, lstm_outputs_total), axis=0, keep_dims=True)
            lstm_output = tf.concat(axis=1, values=lstm_outputs)
        return lstm_output
        
    def fc_forward(self, inputa, inputb, weights, is_training=True, testing=False, network_config=None):
        n_layers = network_config.get('n_layers', 4) # 3
        use_dropout = self._hyperparams.get('use_dropout', False)
        prob = self._hyperparams.get('keep_prob', 0.5)
        # fc_output = tf.add(fc_input, 0)
        if self._hyperparams.get('learn_final_eept', False):
            final_eept_range = self._hyperparams['final_eept_range']
            # fc_output_flat = tf.reshape(fc_output, [-1, T, fc_output.get_shape().dims[-1].value])
            n_layers_ee = network_config.get('n_layers_ee', 1)
            final_eept_pred = tf.tile(tf.expand_dims(inputb[:, 0, :], axis=1), [1, self.T, 1])
            final_eept_pred = tf.reshape(tf.concat(axis=2, values=[inputa, final_eept_pred]), [-1, self.conv_out_size_final+self.lstm.output_size])
            for i in xrange(n_layers_ee):
                final_eept_pred = tf.matmul(final_eept_pred, weights['w_ee_%d' % i]) + weights['b_ee_%d' % i]
                if i != n_layers_ee - 1:
                    final_eept_pred = tf.nn.relu(final_eept_pred)
            final_eept_pred = tf.tile(tf.expand_dims(tf.reshape(final_eept_pred, [-1, self.T, len(final_eept_range)])[:, 0, :], axis=1), [1, self.T, 1])
            fc_output = tf.reshape(tf.concat(axis=2, values=[inputa, inputb, final_eept_pred]), [-1, self.conv_out_size_final+self.lstm.output_size+len(final_eept_range)])
        else:
            fc_output = tf.reshape(tf.concat(axis=2, values=[inputa, inputb]), [-1, self.conv_out_size_final+self.lstm.output_size])
            final_eept_pred = None
        for i in xrange(n_layers):
            fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != n_layers - 1:
                fc_output = tf.nn.relu(fc_output)
                if use_dropout:
                    fc_output = dropout(fc_output, keep_prob=prob, is_training=is_training)
            if i == n_layers - 1 and self._hyperparams.get('mixture_density', False):
                    stop_signal, gripper_command = None, None
                    if self._hyperparams.get('stop_signal', False):
                        stop_signal = tf.expand_dims(fc_output[:, -1], axis=1)
                        fc_output = fc_output[:, :-1]
                    if self._hyperparams.get('gripper_command_signal', False):
                        gripper_command = tf.expand_dims(fc_output[:, -1], axis=1)
                        fc_output = fc_output[:, :-1]
                    ds = tf.contrib.distributions
                    num_mixtures = self._hyperparams.get('num_mixtures', 20)
                    mixture_params = tf.reshape(fc_output, [-1, self.mixture_dim, num_mixtures])
                    mu = mixture_params[:, :-2, :]
                    sigma = tf.exp(mixture_params[:, -2, :])
                    alpha = mixture_params[:, -1, :]
                    mix_gauss = ds.Mixture(
                          cat=ds.Categorical(probs=tf.nn.softmax(alpha, dim=1)),
                          components=[ds.MultivariateNormalDiag(loc=mu[:, :, m], scale_identity_multiplier=sigma[:, m])
                          for m in xrange(num_mixtures)])
                    fc_output = (mix_gauss, gripper_command, stop_signal)
        return fc_output, final_eept_pred

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
        
        if self._hyperparams.get('use_vision', True):
            if input_tensors is None:
                self.obsa = obsa = tf.placeholder(tf.float32, name='obsa') # meta_batch_size x update_batch_size x dim_input
                self.obsb = obsb = tf.placeholder(tf.float32, name='obsb')
            else:
                self.obsa = obsa = input_tensors['inputa'] # meta_batch_size x update_batch_size x dim_input
                self.obsb = obsb = input_tensors['inputb']
        else:
            self.obsa, self.obsb = None, None
        # Temporary in order to make testing work
        if not hasattr(self, 'statea'):
            self.stateb = stateb = tf.placeholder(tf.float32, name='stateb')
            self.actionb = actionb = tf.placeholder(tf.float32, name='actionb')
            if not self._hyperparams.get('human_demo', False):
                self.statea = statea = tf.placeholder(tf.float32, name='statea')
                self.actiona = actiona = tf.placeholder(tf.float32, name='actiona')
            else:
                self.statea = statea = tf.zeros_like(obsa)[:, :, :len(self.x_idx)]
                self.actiona = actiona = tf.zeros_like(obsa)[:, :, :dim_output]
            self.reference_tensor = reference_tensor = tf.placeholder(tf.float32, [self.T, self._dO], name='reference')
            if self.norm_type:
                self.phase = tf.placeholder(tf.bool, name='phase')
        else:
            statea = self.statea
            stateb = self.stateb
            # self.inputa = inputa = tf.placeholder(tf.float32)
            # self.inputb = inputb = tf.placeholder(tf.float32)
            actiona = self.actiona
            actionb = self.actionb
            reference_tensor = self.reference_tensor
        
        if self._hyperparams.get('use_vision', True):
            inputa = tf.concat(axis=2, values=[statea, obsa])
            inputb = tf.concat(axis=2, values=[stateb, obsb])
        else:
            inputa = statea
            inputb = stateb
        
        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            if 'weights' not in dir(self):
                if self._hyperparams.get('learn_final_eept', False):
                    final_eept_range = self._hyperparams['final_eept_range']
                    self.weights = weights = self.construct_weights(dim_input, dim_output-len(final_eept_range), network_config=network_config)
                else:
                    self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
                self.sorted_weight_keys = natsorted(self.weights.keys())
            else:
                training_scope.reuse_variables()
                weights = self.weights
            
            loss_multiplier = self._hyperparams.get('loss_multiplier', 100.0)
            final_eept_loss_eps = self._hyperparams.get('final_eept_loss_eps', 0.01)
            act_loss_eps = self._hyperparams.get('act_loss_eps', 1.0)
            is_training = 'Training' in  prefix
            testing = 'Testing' in prefix
            test_update_batch_size = self._hyperparams.get('test_update_batch_size', 1)
            stop_signal_eps = self._hyperparams.get('stop_signal_eps', 1.0)
            gripper_command_signal_eps = self._hyperparams.get('gripper_command_signal_eps', 1.0)

            def batch_metalearn(inp, update=False):
                inputa, inputb, actiona, actionb = inp #image input
                inputa = tf.reshape(inputa, [-1, dim_input])
                inputb = tf.reshape(inputb, [-1, dim_input])
                actiona = tf.reshape(actiona, [-1, dim_output])
                actionb = tf.reshape(actionb, [-1, dim_output])

                if self._hyperparams.get('learn_final_eept', False):
                    final_eept_range = self._hyperparams['final_eept_range']
                    # assumes update_batch_size == 1
                    # final_eepta = tf.reshape(tf.tile(actiona[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
                    # final_eeptb = tf.reshape(tf.tile(actionb[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
                    final_eepta = actiona[:, final_eept_range[0]:final_eept_range[-1]+1]
                    final_eeptb = actionb[:, final_eept_range[0]:final_eept_range[-1]+1]
                    actiona = actiona[:, :final_eept_range[0]]
                    actionb = actionb[:, :final_eept_range[0]]
                else:
                    final_eepta, final_eeptb = None, None
                
                # Convert to image dims
                if self._hyperparams.get('use_vision', True):
                    if test_update_batch_size > 1:
                        inputas = []
                        state_inputas = []
                        inputa_ = tf.reshape(inputa, [test_update_batch_size, self.T, -1])
                        for i in xrange(test_update_batch_size):
                            inputa_tmp, _, state_inputa = self.construct_image_input(inputa_[i], x_idx, img_idx, network_config=network_config)
                            inputas.append(inputa_tmp)
                            state_inputas.append(state_inputa)
                    else:
                        inputa, flat_img_inputa, state_inputa = self.construct_image_input(inputa, x_idx, img_idx, network_config=network_config)
                    inputb, flat_img_inputb, state_inputb = self.construct_image_input(inputb, x_idx, img_idx, network_config=network_config)
                else:
                    flat_img_inputb = tf.add(inputb, 0)
                    
                if self._hyperparams.get('zero_state', False):
                    if testing and test_update_batch_size > 1:
                        state_inputas = [tf.zeros_like(s) for s in state_inputas]
                    else:
                        state_inputa = tf.zeros_like(state_inputa)
                    
                if 'Training' in prefix:
                    # local_outputa, fp, moving_mean, moving_variance = self.forward(inputa, state_inputa, weights, network_config=network_config)
                    if self._hyperparams.get('use_vision', True):
                        inputa = self.conv_forward(inputa, state_inputa, weights, network_config=network_config)
                        inputb = self.conv_forward(inputb, state_inputb, weights, network_config=network_config)
                    local_lstm_outputa = self.lstm_forward(inputa, actiona, network_config=network_config)
                    inputb = tf.reshape(inputb, [-1, self.T, self.conv_out_size_final])
                    # local_outputb = tf.reshape(tf.concat(axis=2, values=[local_lstm_outputa, inputb]), [-1, self.conv_out_size_final+self.lstm.output_size])
                    local_output, final_eept_predb = self.fc_forward(local_lstm_outputa, inputb, weights, network_config=network_config)
                else:
                    # local_outputa, _, moving_mean_test, moving_variance_test = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
                    local_outputs = []
                    final_eeptbs = []
                    if self._hyperparams.get('use_vision', True):
                        inputb = self.conv_forward(inputb, state_inputb, weights, update=update, testing=testing, is_training=False, network_config=network_config)
                    if test_update_batch_size > 1:
                        actionas = tf.reshape(actiona, [test_update_batch_size, self.T, -1])
                        for i in xrange(test_update_batch_size):
                            if self._hyperparams.get('use_vision', True):
                                inputa, _ = self.conv_forward(inputas[i], state_inputas[i], weights, update=update, testing=testing, is_training=False, network_config=network_config)
                            local_lstm_outputa = self.lstm_forward(inputa, actionas[i], is_training=False, network_config=network_config)
                            inputb = tf.reshape(inputb, [-1, self.T, self.conv_out_size_final])
                            # local_outputb = tf.reshape(tf.concat(axis=2, values=[local_lstm_outputa, inputb]), [-1, self.conv_out_size_final+self.lstm.output_size])
                            local_output, final_eept_predb = self.fc_forward(local_outputa, inputb, weights, is_training=False, testing=testing, network_config=network_config)
                            local_outputs.append(local_output)
                            final_eeptbs.append(final_eept_predb)
                        local_output = tf.reduce_mean(local_outputs, axis=0, keep_dims=True)
                        final_eept_predb = tf.reduce_mean(final_eeptbs, axis=0, keep_dims=True)
                    else:
                        if self._hyperparams.get('use_vision', True):
                            inputa = self.conv_forward(inputa, state_inputa, weights, update=update, testing=testing, is_training=False, network_config=network_config)
                        local_lstm_outputa = self.lstm_forward(inputa, actiona, is_training=False, network_config=network_config)
                        inputb = tf.reshape(inputb, [-1, self.T, self.conv_out_size_final])
                        # local_outputb = tf.reshape(tf.concat(axis=2, values=[local_lstm_outputa, inputb]), [-1, self.conv_out_size_final+self.lstm.output_size])
                        local_output, final_eept_predb = self.fc_forward(local_lstm_outputa, inputb, weights, is_training=False, testing=testing, network_config=network_config)
                if self._hyperparams.get('mixture_density', False):
                    local_output, gripper_command, stop_signal = local_output
                    if testing:
                        samples = local_output.sample(100)
                        sample_probs = local_output.prob(tf.reshape(samples, [-1, samples.get_shape().dims[-1].value]))
                        # this assumes T=1 at test time
                        output_sample = samples[tf.argmax(sample_probs, axis=0)]
                    else:
                        output_sample = local_output.sample()
                    if gripper_command is not None:
                        gripper_command_out = tf.cast(tf.sigmoid(gripper_command)>0.5, tf.float32)
                        output_sample = tf.concat([output_sample, gripper_command_out], axis=1)
                    if stop_signal is not None:
                        stop_signal_out = tf.cast(tf.sigmoid(stop_signal)>0.5, tf.float32)
                        output_sample = tf.concat([output_sample, stop_signal_out], axis=1)
                    outputb = output_sample
                else:
                    outputb = local_output
                discrete_loss = 0.0
                if not self._hyperparams.get('mixture_density', False):
                    local_output_ = tf.identity(local_output)
                actionb_ = tf.identity(actionb)
                if self._hyperparams.get('stop_signal', False):
                    if self._hyperparams.get('mixture_density', False):
                        stop_signal_logitb = stop_signal
                    else:
                        stop_signal_logitb = tf.expand_dims(local_output[:, -1], axis=1)
                    stop_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -1], axis=1), logits=stop_signal_logitb)
                    discrete_loss += stop_signal_eps * tf.reduce_mean(stop_lossb)
                    if not self._hyperparams.get('mixture_density', False):
                        local_output_ = local_output_[:, :-1]
                    actionb_ = actionb_[:, :-1]
                if self._hyperparams.get('gripper_command_signal', False):
                    if self._hyperparams.get('mixture_density', False):
                        gripper_command_logitb = gripper_command
                    else:
                        gripper_command_logitb = tf.expand_dims(local_output[:, -2], axis=1)
                    gripper_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -2], axis=1), logits=gripper_command_logitb)
                    discrete_loss += gripper_command_signal_eps * tf.reduce_mean(gripper_lossb)
                    if not self._hyperparams.get('mixture_density', False):
                        local_output_ = local_output_[:, :-1]
                    actionb_ = actionb_[:, :-1]
                if self._hyperparams.get('mixture_density', False):
                    local_loss = act_loss_eps * tf.reduce_mean(-local_output.log_prob(actionb_)) + discrete_loss
                else:
                    local_loss = act_loss_eps * euclidean_loss_layer(local_output_, actionb_, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False)) + discrete_loss
                
                if self._hyperparams.get('learn_final_eept', False):
                    final_eept_loss = euclidean_loss_layer(final_eept_predb, final_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                else:
                    final_eept_loss = tf.constant(0.0)
                if self._hyperparams.get('learn_final_eept', False):
                    local_loss += final_eept_loss_eps * final_eept_loss
                
                local_fn_output = [outputb, local_loss, final_eept_loss, flat_img_inputb]
                return local_fn_output
                
        if self.norm_type:
            # initialize batch norm vars.
            if self.norm_type == 'vbn':
                # Initialize VBN
                # Uncomment below to update the mean and mean_sq of the reference batch
                self.reference_out = batch_metalearn((reference_tensor, reference_tensor, actionb[0], actionb[0]), update=True)[2]
                # unused = batch_metalearn((reference_tensor, reference_tensor, actionb[0], actionb[0]), update=False)[3]
            else:
                unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))
        
        # out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        out_dtype = [tf.float32, tf.float32, tf.float32, tf.float32]
        result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb), dtype=out_dtype)
        print 'Done with map.'
        return result
    
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
        losses= []
        log_dir = self._hyperparams['log_dir']# + '_%s' % datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        # log_dir = self._hyperparams['log_dir'] # for debugging
        save_dir = self._hyperparams['save_dir'] #'_model' #'_model_ln'
        train_writer = tf.summary.FileWriter(log_dir, self.graph)
        # actual training.
        with Timer('Training'):
            if self.restore_iter == 0:
                training_range = range(TOTAL_ITERS)
            else:
                training_range = range(self.restore_iter+1, TOTAL_ITERS)
            for itr in training_range:
                # TODO: need to make state and obs compatible
                state, tgt_mu = self.generate_data_batch(itr)
                statea = state[:, :self.update_batch_size*self.T, :]
                stateb = state[:, self.update_batch_size*self.T:, :]
                actiona = tgt_mu[:, :self.update_batch_size*self.T, :]
                actionb = tgt_mu[:, self.update_batch_size*self.T:, :]
                if self._hyperparams.get('human_demo', False):
                    feed_dict = {self.stateb: state,
                            self.actionb: tgt_mu}
                else:
                    feed_dict = {self.statea: statea,
                                self.stateb: stateb,
                                self.actiona: actiona,
                                self.actionb: actionb}
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
                    train_writer.add_summary(result[-2], itr)
                    losses.append(result[-1])
    
                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print 'Iteration %d: average loss is %.2f' % (itr, np.mean(losses))
                    losses = []

                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    if len(self.val_idx) > 0:
                        input_tensors = [self.val_summ_op, self.val_total_loss]
                        val_state, val_act = self.generate_data_batch(itr, train=False)
                        statea = val_state[:, :self.update_batch_size*self.T, :]
                        stateb = val_state[:, self.update_batch_size*self.T:, :]
                        actiona = val_act[:, :self.update_batch_size*self.T, :]
                        actionb = val_act[:, self.update_batch_size*self.T:, :]
                        if self._hyperparams.get('human_demo', False):
                            feed_dict = {self.stateb: val_state,
                                    self.actionb: val_act}
                        else:
                            feed_dict = {self.statea: statea,
                                        self.stateb: stateb,
                                        self.actiona: actiona,
                                        self.actionb: actionb}
                        # if self.use_batchnorm:
                        #     feed_dict[self.phase] = 0
                        if self.norm_type == 'vbn':
                            feed_dict[self.reference_tensor] = self.reference_batch
                            input_tensors.append(self.reference_out)
                        results = self.run(input_tensors, feed_dict=feed_dict)
                        train_writer.add_summary(results[0], itr)
                        print 'Test results: average loss is %.2f' % (np.mean(results[1]))
                
                if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
                    self.save_model(save_dir + '_%d' % itr)

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']
