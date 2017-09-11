""" This file defines policy optimization for a tensorflow policy. """
from __future__ import division
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
from gps.algorithm.policy.tf_policy_maml import TfPolicyMAML
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.algorithm.policy_opt.learn_state import LearnState
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict#, extract_demo_dict_multi
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

ANNEAL_INTERVAL = 20000 # this used to be 5000

class PolicyCloningMAML(PolicyOptTf):
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
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
            else:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
                # self._sess = tf.Session(graph=self.graph)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)#1.0
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf.Session(graph=self.graph, config=tf_config)
        else:
            self._sess = tf.Session(graph=self.graph)
        self.act_op = None  # mu_hat
        self.test_act_op = None
        self.feat_op = None # features
        self.image_op = None # image
        self.total_loss1 = None
        self.total_losses2 = None
        self.obs_tensor = None
        self.state_tensor = None
        self.action_tensor = None  # mu true
        self.train_op = None
        self.phase = None
        self.reference_tensor = None
        self.reference_out = None
        self.scale = None
        self.bias = None
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
        if self._hyperparams.get('activation_fn', 'lrelu'):
            self.activation_fn = lrelu
        else:
            self.activation_fn = tf.nn.relu

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
        
        demo_file = hyperparams['demo_file']
        # demo_file = hyperparams['demo_file'][:600]
        # demo_file.extend(hyperparams['demo_file'][-76:])
        # demo_file = hyperparams['demo_file']#[:186]
        # demo_file.extend(hyperparams['demo_file'][-61:])
        if self._hyperparams.get('use_noisy_demos', False):
            noisy_demo_file = hyperparams['noisy_demo_file']
        
        self.restore_iter = hyperparams.get('restore_iter', 0)
        if self._hyperparams.get('use_noisy_demos', False):
            self.extract_supervised_data(noisy_demo_file, noisy=True)
        self.extract_supervised_data(demo_file)
        if self._hyperparams.get('learn_state', False):
            self.state_learner = LearnState(hyperparams, dO, len(self.x_idx), self.demos, self.train_idx, self.val_idx)
        if hyperparams.get('test', False):
            import pdb; pdb.set_trace()
            import pickle
            # with open('/home/kevin/gps/scale_and_bias_place_vr.pkl', 'wb') as f:
            # with open('/home/kevin/gps/scale_and_bias_place_vr_rep_last.pkl', 'wb') as f:
            with open('/home/kevin/gps/scale_and_bias_push_consistent_12.pkl', 'wb') as f:
            # with open('/home/kevin/gps/scale_and_bias_reach_vr_3_obj_final_img.pkl', 'wb') as f:
            # with open('/home/kevin/gps/scale_and_bias_reach_vr_3_obj_interleave.pkl', 'wb') as f:
                pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
        if not hyperparams.get('test', False):
            self.generate_batches(noisy=self._hyperparams.get('use_noisy_demos', False))

        if not hyperparams.get('test', False):
            self.init_network(self.graph, restore_iter=self.restore_iter)
            self.init_network(self.graph, restore_iter=self.restore_iter, prefix='Validation_')
        else:
            self.init_network(self.graph, prefix='Testing')
        
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=10)
        
        with self.graph.as_default():
            init_op = tf.global_variables_initializer()
        self.run(init_op)
        with self.graph.as_default():
            tf.train.start_queue_runners(sess=self._sess)
        
        if self.restore_iter > 0:
            self.restore_model(hyperparams['save_dir'] + '_%d' % self.restore_iter)
            import pdb; pdb.set_trace()
            self.update()
            # TODO: also implement resuming training from restored model
        else:
            self.update()
            # import pdb; pdb.set_trace()
        if not hyperparams.get('test', False):
            os._exit(1) # debugging

        # Replace input tensors to be placeholders for rolling out learned policy
        # self.init_network(self.graph, prefix='Testing')
        # Initialize policy with noise
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.policy = TfPolicyMAML(dU, self.obsa, self.statea, self.actiona, self.obsb, self.stateb,
                               self.test_act_op, self.reference_tensor, self.reference_out,
                               self.feat_op, self.image_op, self.norm_type,
                               0.5*np.ones(dU), self._sess, self.graph, self.device_string,
                               use_vision=self._hyperparams.get('use_vision', True),
                            #   np.zeros(dU), self._sess, self.graph, self.device_string, 
                               copy_param_scope=self._hyperparams['copy_param_scope'])
        self.policy.scale = self.scale
        self.policy.bias = self.bias
        self.policy.x_idx = self.x_idx
        self.policy.img_idx = self.img_idx
        # Generate selected demos for preupdate pass during testing
        self.generate_testing_demos()
        # TODO: This has a bug. Fix it at some point.
        # self.eval_fast_weights()
        self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        self.policy.demos = None
        self.policy.selected_demoO = None
        self.policy.selected_demoU = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self, graph, input_tensors=None, restore_iter=0, prefix='Training_'):
        """ Helper method to initialize the tf networks used """
        with graph.as_default():
            image_tensors = None
            if self._hyperparams.get('use_vision', True):
                if 'Training' in prefix:
                    image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter)
                elif 'Validation' in prefix:
                    image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter, train=False)
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
            outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, flat_img_inputb, fast_weights_values, gradients = result
            if 'Testing' in prefix:
                self.obs_tensor = self.obsa
                self.state_tensor = self.statea
                self.action_tensor = self.actiona
                self.act_op = outputas
                self.outputbs = outputbs
                self.test_act_op = test_output # post-update output
                toy_output_variable = tf.add(test_output, 0, name='output_action')
                self.image_op = flat_img_inputb
                self.fast_weights = {key: fast_weights_values[i] for i, key in enumerate(self.sorted_weight_keys)}
            if 'Training' in prefix:
                self.train_act_op = test_output # post-update output

            trainable_vars = tf.trainable_variables()
            total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            total_final_eept_losses2 = [tf.reduce_sum(final_eept_lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            
            if 'Training' in prefix:
                self.total_loss1 = total_loss1
                self.total_losses2 = total_losses2
                self.total_final_eept_losses2 = total_final_eept_losses2
                self.lossesa = lossesa # for testing
                self.lossesb = lossesb[-1] # for testing
            elif 'Validation' in prefix:
                self.val_total_loss1 = total_loss1
                self.val_total_losses2 = total_losses2
                self.val_total_final_eept_losses2 = total_final_eept_losses2
            # self.val_total_loss1 = tf.contrib.copy_graph.get_copied_op(total_loss1, self.graph)
            # self.val_total_losses2 = [tf.contrib.copy_graph.get_copied_op(total_losses2[i], self.graph) for i in xrange(len(total_losses2))]
 
            # Initialize solver
            # mom1, mom2 = 0.9, 0.999 # adam defaults
            # self.global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(self.meta_lr, self.global_step, ANNEAL_INTERVAL, 0.5, staircase=True)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_losses2[self.num_updates - 1], global_step=self.global_step)
            # if self._hyperparams.get('use_vision', True):
            #     flat_img_inputb = tf.reshape(flat_img_inputb, [self.meta_batch_size, self.T, 3, 80, 64])
            #     flat_img_inputb = tf.transpose(flat_img_inputb, perm=[0,1,4,3,2])
            if 'Training' in prefix:
                optimizer = self._hyperparams.get('optimizer', 'adam')
                if optimizer == 'adam':
                    self.train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_losses2[self.num_updates - 1])
                elif optimizer == 'momentum':
                    momentum = self._hyperparams.get('momentum', 0.9)
                    self.train_op = tf.train.MomentumOptimizer(self.meta_lr, momentum).minimize(self.total_losses2[self.num_updates - 1])
                else:
                    raise NotImplementedError
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.total_loss1)] # tf.scalar_summary('Learning rate', learning_rate)
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                # if self._hyperparams.get('use_vision', True):
                #     for i in xrange(self.meta_batch_size):
                #         summ.append(tf.summary.image('Training_image_%d' % i, flat_img_inputb[i]*255.0, max_outputs=50))
                for j in xrange(self.num_updates):
                    summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.total_losses2[j]))
                    summ.append(tf.summary.scalar(prefix + 'Post-update_final_eept_loss_step_%d' % j, self.total_final_eept_losses2[j]))
                    for k in xrange(len(self.sorted_weight_keys)):
                        summ.append(tf.summary.histogram('Gradient_of_%s_step_%d' % (self.sorted_weight_keys[k], j), gradients[j][k]))
                # summ.append(tf.summary.scalar('Step size', self.step_size))
                self.train_summ_op = tf.summary.merge(summ)
            elif 'Validation' in prefix:
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.val_total_loss1)] # tf.scalar_summary('Learning rate', learning_rate)
                # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
                # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
                # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
                # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
                # if self._hyperparams.get('use_vision', True):
                #     for i in xrange(self.meta_batch_size):
                #         summ.append(tf.summary.image('Validation_image_%d' % i, flat_img_inputb[i]*255.0, max_outputs=50))
                for j in xrange(self.num_updates):
                    summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.val_total_losses2[j]))
                    summ.append(tf.summary.scalar(prefix + 'Post-update_final_eept_loss_step_%d' % j, self.val_total_final_eept_losses2[j]))
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
        if self._hyperparams.get('pretrain', False):
            image_input = image_input * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
            # 'RGB'->'BGR'
            image_input = image_input[:, :, :, ::-1]
        return image_input, flat_image_input, state_input
    
    def construct_weights(self, dim_input=27, dim_output=7, network_config=None):
        weights = {}
        if self._hyperparams.get('use_vision', True):
            num_filters = network_config['num_filters']
            strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
            filter_sizes = network_config.get('filter_size', [3]*len(strides)) # used to be 2
            if type(filter_sizes) is not list:
                filter_sizes = len(strides)*[filter_sizes]
            im_height = network_config['image_height']
            im_width = network_config['image_width']
            num_channels = network_config['image_channels']
            is_dilated = network_config.get('is_dilated', False)
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
            # elif is_dilated:
            #     self.conv_out_size = int(im_width*im_height*num_filters[-1])
            else:
                self.conv_out_size = int(np.ceil(im_width/(downsample_factor)))*int(np.ceil(im_height/(downsample_factor)))*num_filters[-1] # 3 layers each with stride 2            # self.conv_out_size = int(im_width/(16.0)*im_height/(16.0)*num_filters[3]) # 3 layers each with stride 2
    
            # conv weights
            fan_in = num_channels
            if self._hyperparams.get('use_img_context', False) or self._hyperparams.get('use_conv_context', False):
                fan_in += num_channels
            if self._hyperparams.get('use_conv_context', False):
                weights['img_context'] = safe_get('img_context', initializer=tf.zeros([im_height, im_width, num_channels], dtype=tf.float32))
                if self._hyperparams.get('normalize_img_context', False):
                    weights['img_context'] = tf.clip_by_value(weights['img_context'], 0., 1.)
            for i in xrange(n_conv_layers):
                # if not pretrain or (i != 0 and i != 1):
                if not pretrain or i != 0:
                    if self.norm_type == 'selu':
                        weights['wc%d' % (i+1)] = init_conv_weights_snn([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    elif initialization == 'xavier':                
                        weights['wc%d' % (i+1)] = init_conv_weights_xavier([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    elif initialization == 'random':
                        weights['wc%d' % (i+1)] = init_weights([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                    else:
                        raise NotImplementedError
                    weights['bc%d' % (i+1)] = init_bias([num_filters[i]], name='bc%d' % (i+1))
                    fan_in = num_filters[i]
                else:
                    import h5py
                    
                    assert num_filters[i] == 64
                    vgg_filter_size = 3
                    weights['wc%d' % (i+1)] = safe_get('wc%d' % (i+1), [vgg_filter_size, vgg_filter_size, fan_in, num_filters[i]], dtype=tf.float32, trainable=train_conv1)
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
            if self._hyperparams.get('learn_final_eept', False):
                final_eept_range = self._hyperparams['final_eept_range']
                final_eept_in_shape = self.conv_out_size
                if self._hyperparams.get('use_state_context', False):
                    final_eept_in_shape += self._hyperparams.get('context_dim', 10)
                weights['w_ee'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee')
                weights['b_ee'] = init_bias([len(final_eept_range)], name='b_ee')
                if self._hyperparams.get('two_heads', False) and self._hyperparams.get('no_final_eept', False):
                    weights['w_ee_two_heads'] = init_weights([final_eept_in_shape, len(final_eept_range)], name='w_ee_two_heads')
                    weights['b_ee_two_heads'] = init_bias([len(final_eept_range)], name='b_ee_two_heads')
                in_shape += (len(final_eept_range))
        else:
            in_shape = dim_input
        if self._hyperparams.get('free_state', False):
            weights['state'] = safe_get('state', initializer=tf.zeros([self.T*self.update_batch_size, len(self.x_idx)], dtype=tf.float32))
        if self._hyperparams.get('use_context', False) or self._hyperparams.get('use_state_context', False):
            in_shape += self._hyperparams.get('context_dim', 10)
        if self._hyperparams.get('use_state_context', False):
            weights['context'] = safe_get('context', initializer=tf.zeros([self._hyperparams.get('context_dim', 10)], dtype=tf.float32))
        if self._hyperparams.get('use_rnn', False):
            fc_weights = self.construct_rnn_weights(in_shape, dim_output, network_config=network_config)
        elif self._hyperparams.get('use_lstm', False):
            fc_weights = self.construct_lstm_weights(in_shape, dim_output, network_config=network_config)
        else:
            fc_weights = self.construct_fc_weights(in_shape, dim_output, network_config=network_config)
        self.conv_out_size_final = in_shape
        weights.update(fc_weights)
        return weights
    
    def construct_fc_weights(self, dim_input=27, dim_output=7, network_config=None):
        n_layers = network_config.get('n_layers', 4) # TODO TODO this used to be 3.
        dim_hidden = network_config.get('layer_size', [100]*(n_layers-1))  # TODO TODO This used to be 20.
        if type(dim_hidden) is not list:
            dim_hidden = (n_layers - 1)*[dim_hidden]
        dim_hidden.append(dim_output)
        weights = {}
        in_shape = dim_input
        for i in xrange(n_layers):
            if self._hyperparams.get('sep_state', False) and i == 0:
                if self.norm_type == 'selu':
                    weights['w_%d_img' % i] = init_fc_weights_snn([in_shape-len(self.x_idx), dim_hidden[i]], name='w_%d_img' % i)
                    weights['w_%d_state' % i] = init_fc_weights_snn([len(self.x_idx), dim_hidden[i]], name='w_%d_state' % i)
                else:
                    weights['w_%d_img' % i] = init_weights([in_shape-len(self.x_idx), dim_hidden[i]], name='w_%d_img' % i)
                    weights['w_%d_state' % i] = init_weights([len(self.x_idx), dim_hidden[i]], name='w_%d_state' % i)
                    if self._hyperparams.get('two_arms', False):
                        weights['b_%d_state_two_arms' % i] = init_bias([dim_hidden[i]], name='b_%d_state_two_arms' % i)
                        if self._hyperparams.get('free_state', False):
                            weights['w_%d_state_two_arms' % i] = init_weights([len(self.x_idx), dim_hidden[i]], name='w_%d_state_two_arms' % i)
                weights['b_%d_img' % i] = init_bias([dim_hidden[i]], name='b_%d_img' % i)
                weights['b_%d_state' % i] = init_bias([dim_hidden[i]], name='b_%d_state' % i)
                in_shape = dim_hidden[i]
                continue
            if i > 0 and self._hyperparams.get('bt_all_fc', False):
                in_shape += self._hyperparams.get('context_dim', 10)
                weights['context_%d' % i] = init_bias([self._hyperparams.get('context_dim', 10)], name='context_%d' % i)
            if self.norm_type == 'selu':
                weights['w_%d' % i] = init_fc_weights_snn([in_shape, dim_hidden[i]], name='w_%d' % i)
            else:
                weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
                # weights['w_%d' % i] = init_fc_weights_xavier([in_shape, dim_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
            if (i == n_layers - 1 or (i == 0 and self._hyperparams.get('zero_state', False) and not self._hyperparams.get('sep_state', False))) and \
                self._hyperparams.get('two_heads', False):
                weights['w_%d_two_heads' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d_two_heads' % i)
                weights['b_%d_two_heads' % i] = init_bias([dim_hidden[i]], name='b_%d_two_heads' % i)
            in_shape = dim_hidden[i]
            if i == n_layers - 1 and self._hyperparams.get('learn_loss', False):
                n_loss_layers = network_config.get('n_loss_layers', 2)
                loss_layer_size = network_config.get('loss_layer_size', 40)
                dim_hidden_loss = (n_loss_layers-1)*[loss_layer_size]
                dim_hidden_loss.append(1) # single output for loss
                in_shape *= 2 # concat act_hat and act_tgt
                for j in xrange(n_loss_layers):
                    weights['w_loss_%d' % j] = init_weights([in_shape, dim_hidden_loss[j]], name='w_loss_%d' % j)
                    weights['b_loss_%d' % j] = init_bias([dim_hidden_loss[j]], name='b_loss_%d' % j)
                    in_shape = dim_hidden_loss[j]
        return weights
    
    def construct_rnn_weights(self, dim_input=27, dim_output=7, network_config=None):
        num_units = self._hyperparams.get('num_units', 200)
        weights = {}
        weights['rnn_state'] = safe_get('rnn_state', initializer=tf.zeros([None, num_units], dtype=tf.float32))
        weights['rnn_weight'] = safe_get('rnn_weight', [dim_input+num_units, num_units], dtype=tf.float32)
        weights['rnn_bias'] = init_bias([num_units], name='rnn_bias')
        fc_weights = self.construct_fc_weights(num_units, dim_output, network_config=network_config)
        weights.update(fc_weights)
        return weights
        
    def construct_lstm_weights(self, dim_input=27, dim_output=7, network_config=None):
        # TODO: implement this
        num_units = self._hyperparams.get('num_units', 200)
        self.lstm_state_size = 2 * num_units
        assert self.update_batch_size == self.test_batch_size
        weights = {}
        weights['lstm_state'] = tf.get_variable('lstm_state', shape=[self.update_batch_size, self.lstm_state_size], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        weights['lstm_weight'] = safe_get('lstm_weight', [dim_input+num_units, 4*num_units], dtype=tf.float32)
        weights['lstm_bias'] = init_bias([4*num_units], name='lstm_bias')
        fc_weights = self.construct_fc_weights(num_units, dim_output, network_config=network_config)
        weights.update(fc_weights)
        return weights
        
    def vbn(self, tensor, name, update=False):
        VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor, update=update)

    def forward(self, image_input, state_input, weights, meta_testing=False, update=False, is_training=True, testing=False, network_config=None):
        # tile up context variable
        if self._hyperparams.get('use_state_context', False):
            # if not testing:
            #     if not meta_testing:
            #         batch_size = self.update_batch_size*self.T
            #     else:
            #         batch_size = self.test_batch_size*self.T
            # else:
            #     batch_size = 1
            # context = tf.reshape(tf.tile(weights['context'], [batch_size]), [batch_size, -1])
            if self._hyperparams.get('use_vision', True):
                im_height = network_config['image_height']
                im_width = network_config['image_width']
                num_channels = network_config['image_channels']
                flatten_image = tf.reshape(image_input, [-1, im_height*im_width*num_channels])
            else:
                flatten_image = image_input
            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(self._hyperparams.get('context_dim', 10))))
            context += weights['context']
        if self._hyperparams.get('use_vision', True):
            norm_type = self.norm_type
            decay = network_config.get('decay', 0.9)
            strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
            downsample_factor = strides[0][1]
            n_strides = len(strides)
            n_conv_layers = len(strides)
            use_dropout = self._hyperparams.get('use_dropout', False)
            prob = self._hyperparams.get('keep_prob', 0.5)
            is_dilated = network_config.get('is_dilated', False)
            im_height = network_config['image_height']
            im_width = network_config['image_width']
            num_channels = network_config['image_channels']
            # conv_layer_0, _, _ = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, id=0, is_training=is_training)
            # conv_layer_1, _, _ = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2']), norm_type=norm_type, decay=decay, id=1, is_training=is_training)
            # conv_layer_2, moving_mean, moving_variance = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3']), norm_type=norm_type, decay=decay, id=2, is_training=is_training)            
            conv_layer = image_input
            if self._hyperparams.get('use_conv_context', False):
                if not testing:
                    if not meta_testing:
                        batch_size = self.update_batch_size*self.T
                    else:
                        batch_size = self.test_batch_size*self.T
                else:
                    batch_size = 1
                img_context = tf.reshape(tf.tile(tf.reshape(weights['img_context'], [-1]), [batch_size]), [-1, im_height, im_width, num_channels])
                conv_layer = tf.concat(3, [conv_layer, img_context])
            for i in xrange(n_conv_layers):
                if norm_type == 'vbn':
                    if not use_dropout:
                        conv_layer = self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                        name='vbn_%d' % (i+1), update=update)
                    else:
                        conv_layer = dropout(self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                        name='vbn_%d' % (i+1), update=update), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
                else:
                    if not use_dropout:
                        conv_layer = norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                        norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn)
                    else:
                        conv_layer = dropout(norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                        norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
            if self._hyperparams.get('use_fp', False):
                _, num_rows, num_cols, num_fp = conv_layer.get_shape()
                if is_dilated:
                    num_rows = int(np.ceil(im_width/(downsample_factor**n_strides)))
                    num_cols = int(np.ceil(im_height/(downsample_factor**n_strides)))
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
        
                fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
                fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)
        
                conv_out_flat = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])
            else:
                conv_out_flat = tf.reshape(conv_layer, [-1, self.conv_out_size])
            # conv_out_flat = tf.reshape(conv_layer_3, [-1, self.conv_out_size])
            # if use_dropout:
                # conv_out_flat = dropout(conv_out_flat, keep_prob=0.8, is_training=is_training, name='dropout_input')
            fc_input = tf.add(conv_out_flat, 0)
            if self._hyperparams.get('learn_final_eept', False):
                final_eept_range = self._hyperparams['final_eept_range']
                if testing:
                    T = 1
                else:
                    T = self.T
                conv_out_flat = tf.reshape(conv_out_flat, [-1, T, self.conv_out_size])
                conv_size = self.conv_out_size
                if self._hyperparams.get('use_state_context', False):
                    context_dim = self._hyperparams.get('context_dim', 10)
                    conv_out_flat = tf.concat(concat_dim=2, values=[conv_out_flat, tf.reshape(context, [-1, T, context_dim])])
                    conv_size += context_dim
                # only predict the final eept using the initial image
                final_ee_inp = tf.reshape(conv_out_flat, [-1, conv_size])
                # use video for preupdate only if no_final_eept
                if (not self._hyperparams.get('learn_final_eept_whole_traj', False)) or meta_testing:
                    final_ee_inp = conv_out_flat[:, 0, :]
                if self._hyperparams.get('two_heads', False) and not meta_testing and self._hyperparams.get('no_final_eept', False):
                    final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee_two_heads']) + weights['b_ee_two_heads']
                else:
                    final_eept_pred = tf.matmul(final_ee_inp, weights['w_ee']) + weights['b_ee']
                if (not self._hyperparams.get('learn_final_eept_whole_traj', False)) or meta_testing:
                    final_eept_pred = tf.reshape(tf.tile(tf.reshape(final_eept_pred, [-1]), [T]), [-1, len(final_eept_range)])
                    final_eept_concat = tf.identity(final_eept_pred)
                else:
                    # Assume tbs == 1
                    # Only provide the FC layers with final_eept_pred at first time step
                    final_eept_concat = final_eept_pred[0]
                    final_eept_concat = tf.reshape(tf.tile(tf.reshape(final_eept_concat, [-1]), [T]), [-1, len(final_eept_range)])
                fc_input = tf.concat(concat_dim=1, values=[fc_input, final_eept_concat])
                # fc_input = tf.concat(concat_dim=1, values=[fc_input, final_eept_pred])
            else:
                final_eept_pred = None
        else:
            fc_input = image_input
            final_eept_pred = None
        if self._hyperparams.get('use_state_context', False):
            fc_input = tf.concat(concat_dim=1, values=[fc_input, context])
        if self._hyperparams.get('use_rnn', False):
            if state_input is not None and self._hyperparams.get('use_vision', False):
                fc_input = tf.concat(1, [fc_input, state_input])
            return self.rnn_forward(fc_input, weights, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), final_eept_pred
        if self._hyperparams.get('use_lstm', False):
            if state_input is not None and self._hyperparams.get('use_vision', False):
                fc_input = tf.concat(1, [fc_input, state_input])
            return self.lstm_forward(fc_input, weights, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), final_eept_pred
        return self.fc_forward(fc_input, weights, state_input=state_input, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), final_eept_pred

    def fc_forward(self, fc_input, weights, state_input=None, meta_testing=False, is_training=True, testing=False, network_config=None):
        n_layers = network_config.get('n_layers', 4) # 3
        use_dropout = self._hyperparams.get('use_dropout', False)
        prob = self._hyperparams.get('keep_prob', 0.5)
        fc_output = tf.add(fc_input, 0)
        use_selu = self.norm_type == 'selu'
        use_ln = self._hyperparams.get('ln_for_fc', False)
        norm_type = self.norm_type
        if self._hyperparams.get('use_vision', False) and state_input is not None:
            if not self._hyperparams.get('sep_state', False):
                fc_output = tf.concat(1, [fc_output, state_input])
            elif self._hyperparams.get('use_context', False):
                context_dim = self._hyperparams.get('context_dim', 10)
                context = state_input[:, :context_dim]
                state_input = state_input[:, context_dim:]
                fc_output = tf.concat(1, [fc_output, context])
        for i in xrange(n_layers):
            if i > 0 and self._hyperparams.get('bt_all_fc', False):
                context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(fc_output)), range(self._hyperparams.get('context_dim', 10))))
                context += weights['context_%d' % i]
                fc_output = tf.concat(1, [fc_output, context])
            if (i == n_layers - 1 or (i == 0 and self._hyperparams.get('zero_state', False) and not self._hyperparams.get('sep_state', False))) and \
                self._hyperparams.get('two_heads', False) and not meta_testing:
                fc_output = tf.matmul(fc_output, weights['w_%d_two_heads' % i]) + weights['b_%d_two_heads' % i]
            elif i == 0 and self._hyperparams.get('sep_state', False):
                assert state_input is not None
                if self._hyperparams.get('two_arms', False):
                    if self._hyperparams.get('free_state', False):
                        state_part = tf.matmul(weights['state'], weights['w_%d_state_two_arms' % i]) + weights['b_%d_state_two_arms' % i]
                    else:
                        state_part = weights['b_%d_state_two_arms' % i]
                elif self._hyperparams.get('free_state', False):
                    state_part = tf.matmul(weights['state'], weights['w_%d_state' % i]) + weights['b_%d_state' % i]
                else:
                    state_part = tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
                if not meta_testing:
                    fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + state_part
                else:
                    fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i] + \
                                tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
            else:
                fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != n_layers - 1:
                if use_selu:
                    fc_output = selu(fc_output)
                elif use_ln:
                    fc_output = norm(fc_output, norm_type='layer_norm', id=i, is_training=is_training, prefix='fc_')
                else:
                    fc_output = self.activation_fn(fc_output)
                # only use dropout for post-update
                if use_dropout:# and meta_testing:
                    fc_output = dropout(fc_output, keep_prob=prob, is_training=is_training, name='dropout_fc_%d' % i, selu=use_selu)
        # return fc_output, fp, moving_mean, moving_variance
        return fc_output
        
    def learn_loss(self, act_hat, act_tgt, weights, network_config=False):
        n_loss_layers = network_config.get('n_loss_layers', 2)
        loss_multiplier = self._hyperparams.get('loss_multiplier', 100.0)
        loss = tf.concat(1, [act_hat, act_tgt])
        for j in xrange(n_loss_layers):
            loss = tf.matmul(loss, weights['w_loss_%d' % j]) + weights['b_loss_%d' % j]
            if j != n_loss_layers - 1:
                loss = self.activation_fn(loss)
        # return (loss_multiplier*loss)**2
        return loss**2
        
    def rnn_forward(self, rnn_input, weights, meta_testing=False, is_training=True, testing=False, network_config=None):
        # LSTM forward
        state = weights['rnn_state']
        rnn_input = tf.reshape(rnn_input, [-1, self.T, self.conv_out_size_final])
        rnn_outputs = []
        for t in xrange(self.T):
            inp = tf.concat(1, [rnn_input[:, t, :], state])
            rnn_output = tf.matmul(inp, weights['rnn_weight']) + weights['rnn_bias']
            if self._hyperparams.get('ln_for_rnn', False):
                rnn_output = norm(rnn_output, norm_type='layer_norm', id=t, is_training=is_training, prefix='rnn_')
            else:
                rnn_output = self.activation_fn(rnn_output)
            rnn_outputs.append(rnn_output)
            state = tf.identity(rnn_output)
        rnn_output = tf.concat(0, rnn_outputs)
        return self.fc_forward(rnn_output, weights, is_training=is_training, testing=testing, network_config=network_config)
        
    def lstm_forward(self, lstm_input, weights, meta_testing=False, is_training=True, testing=False, network_config=None):
        # LSTM forward
        state = weights['lstm_state']
        num_units = self._hyperparams.get('num_units', 200)
        use_norm = self._hyperparams.get('ln_for_rnn', False)
        forget_bias = 1.0
        activation = tf.nn.tanh
        lstm_outputs = []
        lstm_input = tf.reshape(lstm_input, [-1, self.T, self.conv_out_size_final])
        for t in xrange(self.T):
            inp = lstm_input[:, t, :]
            c_prev = state[:, :num_units]
            m_prev = state[:, num_units:]
            cell_inputs = tf.concat(1, [inp, m_prev])
            lstm_matrix = tf.matmul(cell_inputs, weights['lstm_weight']) + weights['lstm_bias']
            i, j, f, o = tf.split(1, 4, lstm_matrix)
            if use_norm:
                i = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_i_')
                j = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_j_')
                f = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_f_')
                o = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_o_')
            c = (tf.sigmoid(f + forget_bias) * c_prev + tf.sigmoid(i) * activation(j))
            if use_norm:
                c = norm(c, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_c_')
            m = tf.sigmoid(o) * activation(c)
            state = tf.concat(1, [c, m])
            lstm_outputs.append(m)
        lstm_output = tf.concat(0, lstm_outputs)
        return self.fc_forward(lstm_output, weights, is_training=is_training, testing=testing, network_config=network_config)

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
            self.statea = statea = tf.placeholder(tf.float32, name='statea')
            self.stateb = stateb = tf.placeholder(tf.float32, name='stateb')
            # self.inputa = inputa = tf.placeholder(tf.float32)
            # self.inputb = inputb = tf.placeholder(tf.float32)
            self.actiona = actiona = tf.placeholder(tf.float32, name='actiona')
            self.actionb = actionb = tf.placeholder(tf.float32, name='actionb')
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
        
        if self._hyperparams.get('remove_first_img', False):
            assert self.test_batch_size == 1
            obsa = obsa[:, 1:, :]
            statea = statea[:, 1:, :]
            actiona = actiona[:, 1:, :]
            if 'Testing' not in prefix:
                obsb = obsb[:, 1:, :]
                stateb = stateb[:, 1:, :]
                actionb = actionb[:, 1:, :]
        
        if self._hyperparams.get('initial_image', False):
            assert self.test_batch_size == 1
            obsa = tf.expand_dims(obsa[:, 0, :], axis=1)
            statea = tf.expand_dims(statea[:, 0, :], axis=1)
            actiona = tf.expand_dims(actiona[:, 0, :], axis=1)
            obsb = tf.expand_dims(obsb[:, 0, :], axis=1)
            stateb = tf.expand_dims(stateb[:, 0, :], axis=1)
            actionb = tf.expand_dims(actionb[:, 0, :], axis=1)

        if self._hyperparams.get('use_vision', True):
            inputa = tf.concat(2, [statea, obsa])
            inputb = tf.concat(2, [stateb, obsb])
        else:
            inputa = statea
            inputb = stateb
        
        with tf.variable_scope('model', reuse=None) as training_scope:
            # Construct layers weight & bias
            # TODO: since we flip to reuse automatically, this code below is unnecessary
            if 'weights' not in dir(self):
                if 'final_eept_range' in self._hyperparams:
                    final_eept_range = self._hyperparams['final_eept_range']
                    self.weights = weights = self.construct_weights(dim_input, dim_output-len(final_eept_range), network_config=network_config)
                else:
                    self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
                    # self.weights = weights = self.construct_weights(dim_input, 2, network_config=network_config)
                self.sorted_weight_keys = natsorted(self.weights.keys())
            else:
                training_scope.reuse_variables()
                weights = self.weights
            # self.step_size = tf.abs(tf.Variable(self._hyperparams.get('step_size', 1e-3), trainable=False))
            if self._hyperparams.get('learn_step_size', False):
                self.step_size = tf.abs(safe_get('step_size', initializer=self._hyperparams.get('step_size', 1e-3), dtype=tf.float32))
            else:
                self.step_size = self._hyperparams.get('step_size', 1e-3)
            self.grad_reg = self._hyperparams.get('grad_reg', 0.005)
            act_noise_std = self._hyperparams.get('act_noise_std', 0.5)
            loss_multiplier = self._hyperparams.get('loss_multiplier', 100.0)
            final_eept_loss_eps = self._hyperparams.get('final_eept_loss_eps', 0.01)
            act_loss_eps = self._hyperparams.get('act_loss_eps', 1.0)
            use_whole_traj = self._hyperparams.get('learn_final_eept_whole_traj', False)
            test_update_batch_size = self._hyperparams.get('test_update_batch_size', 1)
            if self._hyperparams.get('use_context', False):
                # self.color_hints = tf.maximum(tf.minimum(safe_get('color_hints', initializer=0.5*tf.ones([3], dtype=tf.float32)), 0.0), 1.0)
                self.context_var = safe_get('context_variable', initializer=tf.zeros([self._hyperparams.get('context_dim', 10)], dtype=tf.float32))
            if self._hyperparams.get('use_img_context', False):
                H = network_config['image_height']
                W = network_config['image_width']
                C = network_config['image_channels']
                self.img_context_var = safe_get('img_context_variable', initializer=tf.zeros([H, W, C], dtype=tf.float32))
                # restrict img context var
                if self._hyperparams.get('normalize_img_context', False):
                    self.img_context_var = tf.clip_by_value(self.img_context_var, 0., 1.)
                
            num_updates = self.num_updates
            lossesa, outputsa = [], []
            lossesb = [[] for _ in xrange(num_updates)]
            outputsb = [[] for _ in xrange(num_updates)]
            
            # TODO: add variable that indicates the color?
            
            def batch_metalearn(inp, update=False):
                inputa, inputb, actiona, actionb = inp #image input
                inputa = tf.reshape(inputa, [-1, dim_input])
                inputb = tf.reshape(inputb, [-1, dim_input])
                actiona = tf.reshape(actiona, [-1, dim_output])
                actionb = tf.reshape(actionb, [-1, dim_output])
                gradients_summ = []
                testing = 'Testing' in prefix
                update_rule = self._hyperparams.get('update_rule', 'sgd')
                
                if update_rule == 'adam':
                    assert num_updates > 1
                    beta1 = 0.9
                    beta2 = 0.999
                    eps = 1e-8
                    m = {key: tf.zeros_like(weights[key]) for key in weights.keys()}
                    v = {key: tf.zeros_like(weights[key]) for key in weights.keys()}
                elif update_rule == 'momentum':
                    mu = self._hyperparams.get('mu', 0.9)
                    v = {key: tf.zeros_like(weights[key]) for key in weights.keys()}
                
                final_eepta, final_eeptb = None, None
                if 'final_eept_range' in self._hyperparams:
                    final_eept_range = self._hyperparams['final_eept_range']
                    # assumes update_batch_size == 1
                    # final_eepta = tf.reshape(tf.tile(actiona[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
                    # final_eeptb = tf.reshape(tf.tile(actionb[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
                    if self._hyperparams.get('learn_final_eept', False):
                        final_eepta = actiona[:, final_eept_range[0]:final_eept_range[-1]]
                        final_eeptb = actionb[:, final_eept_range[0]:final_eept_range[-1]]
                    actiona = actiona[:, :final_eept_range[0]]
                    actionb = actionb[:, :final_eept_range[0]]
                    if self._hyperparams.get('no_final_eept', False):
                        final_eepta = tf.zeros_like(final_eepta)
                # else:
                    # actiona = actiona[:, 6:]
                    # actionb = actionb[:, 6:]
                    
                if self._hyperparams.get('no_action', False):
                    actiona = tf.zeros_like(actiona)
                
                if self._hyperparams.get('add_noise', False):
                    actiona += tf.random_normal([self.T*self.update_batch_size, dim_output], stddev=act_noise_std)
                
                local_outputbs, local_lossesb, final_eept_lossesb = [], [], []
                # Assume fixed data for each update
                actionas = [actiona]*num_updates
                
                # Convert to image dims
                if self._hyperparams.get('use_vision', True):
                    if testing:
                        test_inputas, test_state_inputas = [], []
                        inputa_temp = tf.reshape(inputa, [test_update_batch_size, self.T, -1])
                        for i in xrange(test_update_batch_size):
                            inputa_, _, state_inputa_ = self.construct_image_input(inputa_temp[i], x_idx, img_idx, network_config=network_config)
                            test_inputas.append(inputa_)
                            test_state_inputas.append(state_inputa_)
                    else:
                        inputa, _, state_inputa = self.construct_image_input(inputa, x_idx, img_idx, network_config=network_config)
                    inputb, flat_img_inputb, state_inputb = self.construct_image_input(inputb, x_idx, img_idx, network_config=network_config)
                    inputas = [inputa]*num_updates
                    inputbs = [inputb]*num_updates
                    if self._hyperparams.get('zero_state', False):
                        if testing:
                            test_state_inputas = [tf.zeros_like(s) for s in test_state_inputas]
                            state_inputas = [test_state_inputas]*num_updates
                        else:
                            state_inputa = tf.zeros_like(state_inputa)
                            state_inputas = [state_inputa]*num_updates
                    if self._hyperparams.get('use_img_context', False):
                        if testing:
                            inputa_temp = test_inputas[0]
                        else:
                            inputa_temp = inputa
                        img_context = tf.zeros_like(inputa_temp)
                        img_context += self.img_context_var
                        if testing:
                            test_inputas = [tf.concat(3, [_, img_context]) for _ in test_inputas]
                        else:
                            inputa = tf.concat(3, [inputa, img_context])
                    if testing:
                        state_inputa_news = test_state_inputas
                        state_inputa_temp = test_state_inputas[0]
                    else:
                        state_inputa_new = state_inputa
                        state_inputa_temp = state_inputa
                    if self._hyperparams.get('use_context', False):
                        context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputa_temp)), range(self._hyperparams.get('context_dim', 10))))
                        context += self.context_var
                        if self._hyperparams.get('no_state'):
                            if testing:
                                state_inputa_news = [context]*test_update_batch_size
                            else:
                                state_inputa_new = context
                        else:
                            if testing:
                                state_inputa_news = [tf.concat(1, [context, _]) for _ in state_inputa_news]
                    elif self._hyperparams.get('no_state'):
                        if testing:
                            state_inputa_news = [None]*test_update_batch_size
                        else:
                            state_inputa_new = None
                else:
                    inputas = [inputa]*num_updates
                    if self._hyperparams.get('use_context', False):
                        context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputa)), range(self._hyperparams.get('context_dim', 10))))
                        context += self.context_var
                        inputa = tf.concat(1, [context, inputa])
                    state_inputb = None
                    state_inputa_new = None
                    flat_img_inputb = tf.add(inputb, 0) # pseudo-tensor
                
                if self._hyperparams.get('learn_final_eept', False):
                    final_eeptas = [final_eepta]*num_updates
                outputbs = []
                test_actiona = tf.reshape(actiona, [test_update_batch_size, self.T, -1])
                for i in xrange(test_update_batch_size):
                    if 'Training' in prefix:
                        # local_outputa, fp, moving_mean, moving_variance = self.forward(inputa, state_inputa, weights, network_config=network_config)
                        local_outputa, final_eept_preda = self.forward(test_inputas[i], state_inputa_news[i], weights, network_config=network_config)
                    else:
                        # local_outputa, _, moving_mean_test, moving_variance_test = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
                        local_outputa, final_eept_preda = self.forward(test_inputas[i], state_inputa_news[i], weights, update=update, is_training=False, network_config=network_config)
                    if self._hyperparams.get('learn_final_eept', False):
                        final_eept_lossa = euclidean_loss_layer(final_eept_preda, final_eepta, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    else:
                        final_eept_lossa = tf.constant(0.0)
                    if self._hyperparams.get('learn_loss', False):
                        local_lossa = act_loss_eps * self.learn_loss(local_outputa, test_actiona[i], weights, network_config=network_config)
                    else:
                        local_lossa = act_loss_eps * euclidean_loss_layer(local_outputa, test_actiona[i], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    if self._hyperparams.get('learn_final_eept', False):
                        local_lossa += final_eept_loss_eps * final_eept_lossa
                    
                    grads = tf.gradients(local_lossa, weights.values())
                    gradients = dict(zip(weights.keys(), grads))
                    # make fast gradient zero for weights with gradient None
                    for key in gradients.keys():
                        if gradients[key] is None:
                            gradients[key] = tf.zeros_like(weights[key])
                    if self._hyperparams.get('stop_grad', False):
                        gradients = {key:tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                    if self._hyperparams.get('use_clip', False):
                        clip_min = self._hyperparams['clip_min']
                        clip_max = self._hyperparams['clip_max']
                        for key in gradients.keys():
                            # if 'context' not in key:
                            gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                    if self._hyperparams.get('pretrain', False) and not self._hyperparams.get('train_conv1', False):
                        gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                        gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                        # gradients['wc2'] = tf.zeros_like(gradients['wc2'])
                        # gradients['bc2'] = tf.zeros_like(gradients['bc2'])
                    # if self._hyperparams.get('zero_state', False) and self._hyperparams.get('two_heads', False):
                    #     gradients['w_0_state_two_heads'] = tf.zeros_like(gradients['w_0_state_two_heads'])
                    #     gradients['b_0_state_two_heads'] = tf.zeros_like(gradients['b_0_state_two_heads'])
                    gradients_summ.append([gradients[key] for key in self.sorted_weight_keys])
                    if self._hyperparams.get('use_context', False):
                        context_grad = tf.gradients(local_lossa, self.context_var)[0]
                        if self._hyperparams.get('stop_grad', False):
                            context_grad = tf.stop_gradient(context_grad)
                        if self._hyperparams.get('clip_context', False):
                            clip_min = self._hyperparams['clip_min']
                            clip_max = self._hyperparams['clip_max']
                            context_grad = tf.clip_by_value(context_grad, clip_min, clip_max)
                        fast_context = self.context_var - self.step_size*context_grad
                    if self._hyperparams.get('use_img_context', False):
                        img_context_grad = tf.gradients(local_lossa, self.img_context_var)[0]
                        if self._hyperparams.get('stop_grad', False):
                            img_context_grad = tf.stop_gradient(context_grad)
                        if self._hyperparams.get('clip_context', False):
                            clip_min = self._hyperparams['clip_min']
                            clip_max = self._hyperparams['clip_max']
                            img_context_grad = tf.clip_by_value(img_context_grad, clip_min, clip_max)
                        fast_img_context = self.img_context_var - self.step_size*img_context_grad
                    if self._hyperparams.get('use_vision', True):
                        if self._hyperparams.get('use_img_context', False):
                            img_contextb = tf.zeros_like(inputb)
                            img_contextb += fast_img_context
                            inputb_ = tf.concat(3, [inputb, img_contextb])
                        state_inputb_new_ = state_inputb
                        if self._hyperparams.get('use_context', False):
                            contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputb)), range(self._hyperparams.get('context_dim', 10))))
                            contextb += fast_context
                            if self._hyperparams.get('no_state'):
                                state_inputb_new_ = contextb
                            else:
                                state_inputb_new_ = tf.concat(1, [contextb, state_inputb_new_])
                        elif self._hyperparams.get('no_state'):
                            state_inputb_new_ = None
                    else:
                        if self._hyperparams.get('use_context', False):
                            contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputb)), range(self._hyperparams.get('context_dim', 10))))
                            contextb += fast_context
                            inputb_ = tf.concat(1, [contextb, inputb])
                        state_inputb_new_ = None
                    # Is mask used here?
                    if update_rule == 'adam':
                        m = dict(zip(m.keys(), [m[key]*beta1 + (1.-beta1)*gradients[key] for key in m.keys()]))
                        v = dict(zip(v.keys(), [v[key]*beta2 + (1.-beta2)*(gradients[key]**2) for key in v.keys()]))
                        fast_weights = dict(zip(weights.keys(), [weights[key] - self.step_size*m[key]/(tf.sqrt(v[key])+eps) for key in weights.keys()]))
                    elif update_rule == 'momentum':
                        v_prev = {key: tf.identity(v[key]) for key in v.keys()}
                        v = dict(zip(v.keys(), [mu*v[key] - self.step_size*gradients[key] for key in v.keys()]))
                        fast_weights = dict(zip(weights.keys(), [weights[key] - mu * v_prev[key] + (1 + mu) * v[key] for key in weights.keys()]))
                    else:
                        fast_weights = dict(zip(weights.keys(), [weights[key] - self.step_size*gradients[key] for key in weights.keys()]))
                    if 'Training' in prefix:
                        outputb, final_eept_predb = self.forward(inputb_, state_inputb_new_, fast_weights, meta_testing=True, network_config=network_config)
                    else:
                        outputb, final_eept_predb = self.forward(inputb_, state_inputb_new_, fast_weights, meta_testing=True, update=update, is_training=False, testing=testing, network_config=network_config)
                    outputbs.append(outputb)
                # fast_weights_reg = tf.reduce_sum([self.weight_decay*tf.nn.l2_loss(var) for var in fast_weights.values()]) / tf.to_float(self.T)
                outputb = tf.reduce_mean(tf.concat(0, outputbs), axis=0, keep_dims=True)
                local_outputbs.append(outputb)
                if self._hyperparams.get('learn_final_eept', False):
                    final_eept_lossb = euclidean_loss_layer(final_eept_predb, final_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                else:
                    final_eept_lossb = tf.constant(0.0)
                local_lossb = act_loss_eps * euclidean_loss_layer(outputb, actionb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                if self._hyperparams.get('learn_final_eept', False):
                    local_lossb += final_eept_loss_eps * final_eept_lossb
                if use_whole_traj:
                    # assume tbs == 1
                    final_eept_lossb = euclidean_loss_layer(final_eept_predb[0], final_eeptb[0], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                final_eept_lossesb.append(final_eept_lossb)
                local_lossesb.append(local_lossb)
                
                # TODO: add img context var to cases where num_updates > 1
                for j in range(num_updates - 1):
                    if self._hyperparams.get('use_vision', True):
                        state_inputa_new = state_inputas[j+1]
                        if self._hyperparams.get('use_img_context', False):
                            img_context = tf.zeros_like(img_context)
                            img_context += fast_img_context
                            inputas[j+1] = tf.concat(3, [inputas[j+1], img_context])
                        if self._hyperparams.get('use_context', False):
                            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputas[j+1])), range(self._hyperparams.get('context_dim', 10))))
                            context += fast_context
                            if self._hyperparams.get('no_state'):
                                state_inputa_new = context
                            else:
                                state_inputa_new = tf.concat(1, [context, state_inputa_new])
                        elif self._hyperparams.get('no_state'):
                            state_inputa_new = None
                    else:
                        if self._hyperparams.get('use_context', False):
                            # context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputas[j+1])), range(self._hyperparams.get('context_dim', 10))))
                            # context += self.context_var
                            inputas[j+1] = tf.concat(1, [fast_context, inputas[j+1]])
                        state_inputa_new = None
                        
                    outputa, final_eept_preda = self.forward(inputas[j+1], state_inputa_new, fast_weights, network_config=network_config)
                    if self._hyperparams.get('learn_final_eept', False):
                        final_eept_lossa = euclidean_loss_layer(final_eept_preda, final_eeptas[j+1], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    else:
                        final_eept_lossa = tf.constant(0.0)
                    if self._hyperparams.get('learn_loss', False):
                        loss = act_loss_eps * self.learn_loss(outputa, actionas[j+1], fast_weights, network_config=network_config)
                    else:
                        loss = act_loss_eps * euclidean_loss_layer(outputa, actionas[j+1], None, multiplier=loss_multiplier, behavior_clone=True,
                                                    use_l1=self._hyperparams.get('use_l1', False))# + fast_weights_reg / tf.to_float(self.update_batch_size)
                    if self._hyperparams.get('learn_final_eept', False):
                        loss += final_eept_loss_eps * final_eept_lossa
                    grads = tf.gradients(loss, fast_weights.values())
                    if self._hyperparams.get('use_img_context', False):
                        img_context_grad = tf.gradients(loss, fast_img_context)[0]
                        if self._hyperparams.get('stop_grad', False):
                            img_context_grad = tf.stop_gradient(context_grad)
                        if self._hyperparams.get('clip_context', False):
                            clip_min = self._hyperparams['clip_min']
                            clip_max = self._hyperparams['clip_max']
                            img_context_grad = tf.clip_by_value(img_context_grad, clip_min, clip_max)
                        fast_img_context = fast_img_context - self.step_size*img_context_grad
                    if self._hyperparams.get('use_context', False):
                        context_grad = tf.gradients(loss, fast_context)[0]
                        if self._hyperparams.get('stop_grad', False):
                            context_grad = tf.stop_gradient(context_grad)
                        if self._hyperparams.get('clip_context', False):
                            clip_min = self._hyperparams['clip_min']
                            clip_max = self._hyperparams['clip_max']
                            context_grad = tf.clip_by_value(context_grad, clip_min, clip_max)
                        fast_context = fast_context - self.step_size*context_grad
                    if self._hyperparams.get('use_vision', True):
                        state_inputb_new = state_inputb
                        if self._hyperparams.get('use_img_context', False):
                            img_contextb = tf.zeros_like(img_contextb)
                            img_contextb += fast_img_context
                            inputbs[j+1] = tf.concat(3, [inputbs[j+1], img_contextb])
                        if self._hyperparams.get('use_context', False):
                            contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputb)), range(self._hyperparams.get('context_dim', 10))))
                            contextb += fast_context
                            if self._hyperparams.get('no_state'):
                                state_inputb_new = contextb
                            else:
                                state_inputb_new = tf.concat(1, [contextb, state_inputb_new])
                        elif self._hyperparams.get('no_state'):
                            state_inputb_new = None
                    else:
                        if self._hyperparams.get('use_context', False):
                            contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputbs[j+1])), range(self._hyperparams.get('context_dim', 10))))
                            contextb += fast_context
                            inputb[j+1] = tf.concat(1, [contextb, inputb[j+1]])
                        state_inputb_new = None
                    gradients = dict(zip(fast_weights.keys(), grads))
                    # make fast gradient zero for weights with gradient None
                    for key in gradients.keys():
                        if gradients[key] is None:
                            gradients[key] = tf.zeros_like(fast_weights[key])
                    if self._hyperparams.get('stop_grad', False):
                        gradients = {key:tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                    if self._hyperparams.get('use_clip', False):
                        clip_min = self._hyperparams['clip_min']
                        clip_max = self._hyperparams['clip_max']
                        for key in gradients.keys():
                            # if 'context' not in key:
                            gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                    if self._hyperparams.get('pretrain', False) and not self._hyperparams.get('train_conv1', False):
                        gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                        gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                        # gradients['wc2'] = tf.zeros_like(gradients['wc2'])
                        # gradients['bc2'] = tf.zeros_like(gradients['bc2'])
                    gradients_summ.append([gradients[key] for key in self.sorted_weight_keys])
                    if update_rule == 'adam':
                        m = dict(zip(m.keys(), [m[key]*beta1 + (1.-beta1)*gradients[key] for key in m.keys()]))
                        v = dict(zip(v.keys(), [v[key]*beta2 + (1.-beta2)*(gradients[key]**2) for key in v.keys()]))
                        fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.step_size*m[key]/(tf.sqrt(v[key])+eps) for key in fast_weights.keys()]))
                    elif update_rule == 'momentum':
                        v_prev = {key: tf.identity(v[key]) for key in v.keys()}
                        v = dict(zip(v.keys(), [mu*v[key] - self.step_size*gradients[key] for key in v.keys()]))
                        fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - mu * v_prev[key] + (1 + mu) * v[key] for key in fast_weights.keys()]))
                    else:
                        fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.step_size*gradients[key] for key in fast_weights.keys()]))
                    if 'Training' in prefix:
                        output, final_eept_predb = self.forward(inputbs[j+1], state_inputb_new, fast_weights, meta_testing=True, network_config=network_config)
                        # output = self.forward(inputb, state_inputb, fast_weights, update=update, is_training=False, network_config=network_config)
                    else:
                        output, final_eept_predb = self.forward(inputbs[j+1], state_inputb_new, fast_weights, meta_testing=True, update=update, is_training=False, testing=testing, network_config=network_config)
                    local_outputbs.append(output)
                    if self._hyperparams.get('learn_final_eept', False):
                        final_eept_lossb = euclidean_loss_layer(final_eept_predb, final_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    else:
                        final_eept_lossb = tf.constant(0.0)
                    # fast_weights_reg = tf.reduce_sum([self.weight_decay*tf.nn.l2_loss(var) for var in fast_weights.values()]) / tf.to_float(self.T)
                    lossb = act_loss_eps * euclidean_loss_layer(output, actionb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    if self._hyperparams.get('learn_final_eept', False):
                        lossb += final_eept_loss_eps * final_eept_lossb
                    if use_whole_traj:
                        # assume tbs == 1
                        final_eept_lossb = euclidean_loss_layer(final_eept_predb[0], final_eeptb[0], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
                    final_eept_lossesb.append(final_eept_lossb)
                    local_lossesb.append(lossb)
                if self._hyperparams.get('use_grad_reg'):
                    fast_gradient_reg = 0.0
                    for key in gradients.keys():
                        fast_gradient_reg += self.grad_reg*tf.reduce_sum(tf.abs(gradients[key]))
                    local_lossesb[-1] += self._hyperparams['grad_reg'] *fast_gradient_reg / self.update_batch_size
                # local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test]
                # local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, conv_layer_2, outputs, test_outputs, mean, variance, moving_mean, moving_variance, moving_mean_new, moving_variance_new]
                fast_weights_values = [fast_weights[key] for key in self.sorted_weight_keys]
                # use post update output
                local_fn_output = [local_outputa, local_outputbs, local_outputbs[-1], local_lossa, local_lossesb, final_eept_lossesb, flat_img_inputb, fast_weights_values, gradients_summ]
                import pdb; pdb.s
                return local_fn_output

        if self.norm_type:
            # initialize batch norm vars.
            # TODO: figure out if this line of code is necessary
            if self.norm_type == 'vbn':
                # Initialize VBN
                # Uncomment below to update the mean and mean_sq of the reference batch
                self.reference_out = batch_metalearn((reference_tensor, reference_tensor, actionb[0], actionb[0]), update=False)[2] # used to be True
                # unused = batch_metalearn((reference_tensor, reference_tensor, actionb[0], actionb[0]), update=False)[3]
            else:
                unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))
        
        # out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32, [tf.float32]*len(self.weights.keys()), [[tf.float32]*len(self.weights.keys())]*num_updates]
        result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb), dtype=out_dtype)
        print 'Done with map.'
        return result
    
    def extract_supervised_data(self, demo_file, noisy=False):
        """
            Load demos into memory.
            Args:
                demo_file: list of demo files where each file contains demos of one task.
            Return:
                total_train_obs: all training observations
                total_train_U: all training actions
        """    
        demos = extract_demo_dict(demo_file)
        if not self._hyperparams.get('use_vision', True) and demos[0]['demoX'].shape[2] > self._dO:
            for key in demos.keys():
                demos[key]['demoX'] = demos[key]['demoX'][:, :, :-9].copy()
        if demos[0]['demoX'].shape[2] > len(self.x_idx):
            for key in demos.keys():
                demos[key]['demoX'] = demos[key]['demoX'][:, :, -9:].copy()
        for key in demos.keys():
            demos[key]['demoX'] = demos[key]['demoX'][6:-6, :, :].copy()
            demos[key]['demoU'] = demos[key]['demoU'][6:-6, :, :].copy()
        n_folders = len(demos.keys())
        n_val = self._hyperparams['n_val'] # number of demos for testing
        N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.iteritems())
        print "Number of demos: %d" % N_demos
        idx = np.arange(n_folders)
        # shuffle(idx)
        if not hasattr(self, 'train_idx'):
            if n_val != 0:
                if not self._hyperparams.get('shuffle_val', False):
                    self.val_idx = idx[-n_val:]
                    self.train_idx = idx[:-n_val]
                else:
                    self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
                    mask = np.array([(i in self.val_idx) for i in idx])
                    self.train_idx = np.sort(idx[~mask])
            else:
                self.train_idx = idx
                self.val_idx = []
        # self.gif_prefix = self._hyperparams.get('gif_prefix', 'color')
        # train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.train_idx}
        # # self.val_img_folders = {i: os.path.join(self.demo_gif_dir, 'color_%d' % i) for i in self.val_idx}
        # val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.val_idx}
        # self.train_obs = {}
        # self.val_obs = {}
        # for i in idx:
        #     demoO = []
        #     for j in xrange(8):
        #         print 'Loading gifs for object %d cond %d' % (i, j)
        #         if i in self.train_idx:
        #             demoO.append(np.array(imageio.mimread(os.path.join(train_img_folders[i], 'cond%d.samp0.gif' % j)))[:, :, :, :3])
        #         else:
        #             demoO.append(np.array(imageio.mimread(os.path.join(val_img_folders[i], 'cond%d.samp0.gif' % j)))[:, :, :, :3])
        #     demoO = np.array(demoO).transpose(0, 1, 4, 3, 2)
        #     N, T, _, _, _ = demoO.shape
        #     demoO = demoO.reshape(N, T, -1)
        #     if i in self.train_idx:
        #         self.train_obs[i] = demoO.copy()
        #     else:
        #         self.val_obs[i] = demoO.copy()
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
        if not noisy:
            self.demos = demos
        else:
            self.noisy_demos = demos
        if self.norm_type == 'vbn':
            self.generate_reference_batch()
            
    def generate_reference_batch(self):
        """
            Generate the reference batch for VBN. The reference batch is generated randomly
            at each time step.
        """
        self.reference_batch = np.zeros((self.T, self._dO))
        n_demo = self.demos[0]['demoX'].shape[0]
        for t in xrange(self.T):
            idx = np.random.choice(self.train_idx)
            selected_cond = np.random.choice(np.arange(n_demo))
            self.reference_batch[t, self.x_idx] = self.demos[idx]['demoX'][selected_cond, t, :]
            if self._hyperparams.get('use_vision', True):
                O = np.array(imageio.mimread(self.demo_gif_dir + self.gif_prefix + '_%d/cond%d.samp0.gif' % (idx, selected_cond)))[t, :, :, :3]
                O = np.transpose(O, [2, 1, 0]).flatten()
                O = O / 255.0
                self.reference_batch[t, self.img_idx] = O
        print 'Done generating reference batch.'
        # TODO: make this compatible in new code.
        # self.policy.reference_batch_X = self.reference_batch[:, self.x_idx]
        # self.policy.reference_batch_O = self.reference_batch[:, self.img_idx]
    
    def generate_testing_demos(self):
        if not self._hyperparams.get('use_noisy_demos', False):
            n_folders = len(self.demos.keys())
            demos = self.demos
        else:
            n_folders = len(self.noisy_demos.keys())
            demos = self.noisy_demos
        policy_demo_idx = [np.random.choice(n_demo, replace=False, size=self.update_batch_size) for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]
        self.policy.selected_demoO = []
        self.policy.selected_demoX = []
        self.policy.selected_demoU = []
        for i in xrange(n_folders):
            # TODO: load observations from images instead
            selected_cond = demos[i]['demoConditions'][policy_demo_idx[i][0]] # TODO: make this work for update_batch_size > 1
            if self._hyperparams.get('use_vision', True):
                # For half of the dataset
                if i in self.val_idx and not self._hyperparams.get('use_noisy_demos', False):
                    idx = i + 1000
                else:
                    idx = i
                if self._hyperparams.get('use_noisy_demos', False):
                    O = np.array(imageio.mimread(self.demo_gif_dir + self.gif_prefix + '_%d_noisy/cond%d.samp0.gif' % (idx, selected_cond)))[:, :, :, :3]
                else:
                    O = np.array(imageio.mimread(self.demo_gif_dir + self.gif_prefix + '_%d/cond%d.samp0.gif' % (idx, selected_cond)))[:, :, :, :3]
                if len(O.shape) == 4:
                    O = np.expand_dims(O, axis=0)
                O = np.transpose(O, [0, 1, 4, 3, 2]) # transpose to mujoco setting for images
                O = O.reshape(O.shape[0], self.T, -1) # keep uint8 to save RAM
                self.policy.selected_demoO.append(O)
            X = demos[i]['demoX'][policy_demo_idx[i]].copy()
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=0)
            self.policy.selected_demoX.append(X)
            self.policy.selected_demoU.append(demos[i]['demoU'][policy_demo_idx[i]].copy())
        print "Selected demo is %d" % self.policy.selected_demoX[0].shape[0]
        # self.policy.demos = self.demos #debug
    
    def generate_batches(self, noisy=False):
        with Timer('Generating batches for each iteration'):
            train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.train_idx}
            # self.val_img_folders = {i: os.path.join(self.demo_gif_dir, 'color_%d' % i) for i in self.val_idx}
            val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % i) for i in self.val_idx}
            # val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % (i+93)) for i in self.val_idx}
            if noisy:
                noisy_train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d_noisy' % i) for i in self.train_idx}
                noisy_val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d_noisy' % i) for i in self.val_idx}
            TEST_PRINT_INTERVAL = 500
            TOTAL_ITERS = self._hyperparams['iterations']
            # VAL_ITERS = int(TOTAL_ITERS / 500)
            self.all_training_filenames = []
            self.all_val_filenames = []
            self.training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
            self.val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, int(TOTAL_ITERS/TEST_PRINT_INTERVAL))}
            if noisy:
                self.noisy_training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
                self.noisy_val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, TOTAL_ITERS/TEST_PRINT_INTERVAL)}
            for itr in xrange(TOTAL_ITERS):
                sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
                for idx in sampled_train_idx:
                    if self._hyperparams.get('use_vision', True):
                        sampled_folder = train_img_folders[idx]
                        image_paths = natsorted(os.listdir(sampled_folder))[6:-6]
                        try:
                            assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                        except AssertionError:
                            import pdb; pdb.set_trace()
                        if noisy:
                            noisy_sampled_folder = noisy_train_img_folders[idx]
                            noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                            assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                        if not noisy:
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                            sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                        else:
                            noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) #True
                            sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) #True
                            sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                            sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                        self.all_training_filenames.extend(sampled_images)
                        self.training_batch_idx[itr][idx] = sampled_image_idx
                        if noisy:
                            self.noisy_training_batch_idx[itr][idx] = noisy_sampled_image_idx
                    else:
                        if noisy:
                            self.training_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.test_batch_size, replace=False) # True
                            self.noisy_training_batch_idx[itr][idx] = np.random.choice(range(self.noisy_demos[idx]['demoX'].shape[0]), size=self.update_batch_size, replace=False) # True
                        else:
                            self.training_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                    for idx in sampled_val_idx:
                        if self._hyperparams.get('use_vision', True):
                            sampled_folder = val_img_folders[idx]
                            image_paths = natsorted(os.listdir(sampled_folder))[6:-6]
                            assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                            if noisy:
                                noisy_sampled_folder = noisy_val_img_folders[idx]
                                noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
                                assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
                            if not noisy:
                                sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
                                sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                            else:
                                noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) # True
                                sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) # True
                                sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
                                sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
                            self.all_val_filenames.extend(sampled_images)
                            self.val_batch_idx[itr][idx] = sampled_image_idx
                            if noisy:
                                self.noisy_val_batch_idx[itr][idx] = noisy_sampled_image_idx
                        else:
                            if noisy:
                                self.val_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.test_batch_size, replace=False) # True
                                self.noisy_val_batch_idx[itr][idx] = np.random.choice(range(self.noisy_demos[idx]['demoX'].shape[0]), size=self.update_batch_size, replace=False) # True
                            else:
                                self.val_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.update_batch_size+self.test_batch_size, replace=False) # True

    def make_batch_tensor(self, network_config, restore_iter=0, train=True):
        # TODO: load images using tensorflow fileReader and gif decoder
        TEST_INTERVAL = 500
        batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size
        if train:
            all_filenames = self.all_training_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
        else:
            all_filenames = self.all_val_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size*(int(restore_iter/TEST_INTERVAL)+1):]
        
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
        image = tf.cast(image, tf.float32)
        image /= 255.0
        if self._hyperparams.get('use_hsv', False):
            eps_min, eps_max = self._hyperparams.get('random_V', (0.5, 1.5))
            assert eps_max >= eps_min >= 0
            # convert to HSV only fine if input images in [0, 1]
            img_hsv = tf.image.rgb_to_hsv(image)
            img_h = img_hsv[..., 0]
            img_s = img_hsv[..., 1]
            img_v = img_hsv[..., 2]
            eps = tf.random_uniform([self.T, 1, 1], eps_min, eps_max)
            img_v = tf.clip_by_value(eps * img_v, 0., 1.)
            img_hsv = tf.stack([img_h, img_s, img_v], 3)
            image_rgb = tf.image.hsv_to_rgb(img_hsv)
            image = image_rgb
        image = tf.transpose(image, perm=[0, 3, 2, 1]) # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 64 #128 #256
        print 'Batching images'
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_images = []
        for i in xrange(self.meta_batch_size):
            image = images[i*(self.update_batch_size+self.test_batch_size):(i+1)*(self.update_batch_size+self.test_batch_size)]
            image = tf.reshape(image, [(self.update_batch_size+self.test_batch_size)*self.T, -1])
            all_images.append(image)
        return tf.pack(all_images)
        
    def generate_data_batch(self, itr, train=True):
        if train:
            demos = {key: self.demos[key].copy() for key in self.train_idx}
            idxes = self.training_batch_idx[itr]
            if self._hyperparams.get('use_noisy_demos', False):
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.train_idx}
                noisy_idxes = self.noisy_training_batch_idx[itr]
        else:
            demos = {key: self.demos[key].copy() for key in self.val_idx}
            idxes = self.val_batch_idx[itr]
            if self._hyperparams.get('use_noisy_demos', False):
                noisy_demos = {key: self.noisy_demos[key].copy() for key in self.val_idx}
                noisy_idxes = self.noisy_val_batch_idx[itr]
        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        test_batch_size = self.test_batch_size
        if not self._hyperparams.get('use_noisy_demos', False):
            U = [demos[k]['demoU'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
            U = np.array(U)
            if not self._hyperparams.get('learn_state', False):
                X = [demos[k]['demoX'][v].reshape((test_batch_size+update_batch_size)*self.T, -1) for k, v in idxes.items()]
                X = np.array(X)
            else:
                X = self.state_learner.get_states(U)
        else:
            noisy_U = [noisy_demos[k]['demoU'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            noisy_X = [noisy_demos[k]['demoX'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
            U = [demos[k]['demoU'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            U = np.concatenate((np.array(noisy_U), np.array(U)), axis=1)
            X = [demos[k]['demoX'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
            X = np.concatenate((np.array(noisy_X), np.array(X)), axis=1)
        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.x_idx)
        return X, U
    
    # def generate_data_batch(self, itr, train=True):
    #     if train:
    #         demos = {key: self.demos[key].copy() for key in self.train_idx}
    #         obs = self.train_obs
    #         folder_idx = self.train_idx.copy()
    #         if self._hyperparams.get('use_noisy_demos', False):
    #             noisy_demos = {key: self.noisy_demos[key].copy() for key in self.train_idx}
    #             noisy_idxes = self.noisy_training_batch_idx[itr]
    #     else:
    #         demos = {key: self.demos[key].copy() for key in self.val_idx}
    #         obs = self.val_obs
    #         folder_idx = self.val_idx.copy()
    #         if self._hyperparams.get('use_noisy_demos', False):
    #             noisy_demos = {key: self.noisy_demos[key].copy() for key in self.val_idx}
    #             noisy_idxes = self.noisy_val_batch_idx[itr]
    #     batch_size = self.meta_batch_size
    #     update_batch_size = self.update_batch_size
    #     # shuffle(folder_idx)
    #     # batch_idx = folder_idx[:batch_size]
    #     batch_idx = np.random.choice(folder_idx, size=batch_size, replace=False)
    #     batch_demos = {key: demos[key] for key in batch_idx}
    #     batch_obs = {key: obs[key] for key in batch_idx}
    #     U, X, O = [], [], []
    #     for i in xrange(batch_size):
    #         n_demo = batch_demos[batch_idx[i]]['demoX'].shape[0]
    #         idx_i = np.random.choice(np.arange(n_demo), replace=True, size=update_batch_size+1) # Used to set replace=False
    #         U.append(batch_demos[batch_idx[i]]['demoU'][idx_i].reshape((update_batch_size+1)*self.T, -1))
    #         X.append(batch_demos[batch_idx[i]]['demoX'][idx_i].reshape((update_batch_size+1)*self.T, -1))
    #         O.append(batch_obs[batch_idx[i]][idx_i].reshape((update_batch_size+1)*self.T, -1))
    #     U = np.array(U)
    #     X = np.array(X)
    #     O = np.float32(np.array(O)) / 255.0
    #     return O, X, U
    
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
                    input_tensors.append(self.train_act_op)
                    input_tensors.extend([self.train_summ_op, self.total_loss1, self.total_losses2[self.num_updates-1]])
                result = self.run(input_tensors, feed_dict=feed_dict)
    
                if itr != 0 and itr % SUMMARY_INTERVAL == 0:
                    prelosses.append(result[-2])
                    train_writer.add_summary(result[-3], itr)
                    postlosses.append(result[-1])
    
                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print 'Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(prelosses), np.mean(postlosses))
                    # print 'predict eept is ', result[-4][0, 0, :]
                    # print 'true eept is ', actionb[0, 0, 6:8]
                    prelosses, postlosses = [], []

                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    if len(self.val_idx) > 0:
                        input_tensors = [self.val_summ_op, self.val_total_loss1, self.val_total_losses2[self.num_updates-1]]
                        val_state, val_act = self.generate_data_batch(itr, train=False)
                        statea = val_state[:, :self.update_batch_size*self.T, :]
                        stateb = val_state[:, self.update_batch_size*self.T:, :]
                        actiona = val_act[:, :self.update_batch_size*self.T, :]
                        actionb = val_act[:, self.update_batch_size*self.T:, :]
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
                        print 'Test results: average preloss is %.2f, average postloss is %.2f' % (np.mean(results[1]), np.mean(results[2]))
                
                if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
                    self.save_model(save_dir + '_%d' % itr)

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

    def eval_fast_weights(self):
        fast_weights = {}
        for i in xrange(len(self.policy.selected_demoO)):
            with Timer('Evaluate fast weights %d' % i):
                fast_weights[i] = dict(zip(self.fast_weights.keys(), [np.squeeze(self.run(self.fast_weights[k], feed_dict={self.obs_tensor:np.expand_dims(self.policy.selected_demoO[i], axis=0),
                                    self.state_tensor: np.expand_dims(self.policy.selected_demoX[i], axis=0),
                                    self.action_tensor:np.expand_dims(self.policy.selected_demoU[i], axis=0)}), axis=0) for k in self.fast_weights.keys()]))
        self.policy.fast_weights_value = fast_weights

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
                        gif_dir = gif_dir + self.gif_prefix + '_%d/' % idx
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
                    save=False, noisy=False,
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
                # Sample on validation conditions and get the states.
                X_val = self.sample(agent, i, conditions, N=1, testing=True).get_X()
                val_dists.extend([np.nanmin(np.linalg.norm(X_val[j, :, state_idx].T - target_eepts[j], axis=1)) \
                                    for j in xrange(X_val.shape[0])])
            else:
                # Sample on training conditions.
                X_train = self.sample(agent, i, conditions, N=1).get_X()
                train_dists.extend([np.nanmin(np.linalg.norm(X_train[j, :, state_idx].T - target_eepts[j], axis=1)) \
                                    for j in xrange(X_train.shape[0])])

        import pdb; pdb.set_trace()
        print "Training success rate is %.5f" % (np.array(train_dists) <= success_thresh).mean()
        print "Validation success rate is %.5f" % (np.array(val_dists) <= success_thresh).mean()
