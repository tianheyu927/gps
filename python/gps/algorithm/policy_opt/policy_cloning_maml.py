""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.
import tensorflow as tf

from random import shuffle
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import xu_to_sample_list, extract_demo_dict
from gps.utility.general_utils import BatchSampler, compute_distance, mkdir_p, Timer

ANNEAL_INTERVAL = 2000

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
            if not self._hyperparams.get('uses_vision', False):
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.08)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf_Session(graph=graph, config=tf_config)
            else:
                self.gpu_device = self._hyperparams['gpu_id']
                self.device_string = "/gpu:" + str(self.gpu_device)
                self._sess = tf.Session(graph=self.graph)
                # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
                # tf_config = tf.ConfigProto(gpu_options=gpu_options)
                # self._sess = tf_Session(graph=graph, config=tf_config)
        else:
            self._sess = tf.Session(graph=self.graph)
        self.act_op = None  # mu_hat
        self.test_act_op = None
        self.feat_op = None # features
        self.image_op = None # image
        self.total_loss1 = None
        self.total_losses2 = None
        self.obs_tensor = None
        self.action_tensor = None  # mu true
        self.train_op = None
        self.phase = None
        self.norm_type = self._hyperparams.get('norm_type', False)
        self._hyperparams['network_params'].update({'norm_type': self.norm_type})
        self._hyperparams['network_params'].update({'decay': self._hyperparams.get('decay', 0.99)})
        # MAML hyperparams
        self.update_batch_size = self._hyperparams.get('update_batch_size', 1)
        self.meta_batch_size = self._hyperparams.get('meta_batch_size', 10)
        self.num_updates = self._hyperparams.get('num_updates', 1)
        self.meta_lr = self._hyperparams.get('lr', 1e-3)
        self.weight_decay = self._hyperparams.get('weight_decay', 0.005)
        
        self.init_network()
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        self.var = self._hyperparams['init_var'] * np.ones(dU)
        # use test action for policy action
        self.policy = TfPolicy(dU, self.obs_tensor, self.test_act_op, self.feat_op, self.image_op, self.phase,
                               np.zeros(dU), self._sess, self.graph, self.device_string, 
                               copy_param_scope=self._hyperparams['copy_param_scope'])
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

        with self.graph.as_default():
            init_op = tf.initialize_all_variables()
        self.run(init_op)

        # For loading demos
        if hyperparams.get('agent', False):
            test_agent = hyperparams['agent'][:2]  # Required for sampling
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file'][:2]
        
        if hyperparams.get('agent', False):
            restore_iter = hyperparams.get('restore_iter', 0)
            self.extract_supervised_data(demo_file)
            if restore_iter > 0:
                self.restore_model(hyperparams['log_dir'] + '_model_%d' % restore_iter)
                import pdb; pdb.set_trace()
                # TODO: also implement resuming training from restored model
                self.eval_success_rate(test_agent)
            else:
                self.update()
                # import pdb; pdb.set_trace()
                self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        with self.graph.as_default():
            with Timer('building TF network'):
                result = self.construct_model(dim_input=self._dO, dim_output=self._dU,
                                          network_config=self._hyperparams['network_params'])
            # outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test = result
            # outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp = result
            outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp, conv_layer_2, fc_input, fc_outputs = result
            self.obs_tensor = self.inputa
            self.action_tensor = self.actiona
            self.act_op = outputas
            self.test_act_op = test_outputa
            self.feat_op = fp
            self.image_op = flat_img_inputa
            self.conv_layer_2 = conv_layer_2
            self.fc_input = fc_input
            self.fc_outputs = fc_outputs
            # self.outputs = outputs
            # self.test_outputs = test_outputs
            # self.mean = mean
            # self.variance = variance
            # self.moving_mean = moving_mean
            # self.moving_variance = moving_variance
            # self.moving_mean_new = moving_mean_new
            # self.moving_variance_new = moving_variance_new
            # self.moving_mean = tf.reduce_mean(moving_mean)
            # self.moving_variance = tf.reduce_mean(moving_variance)
            # self.moving_mean_test = tf.reduce_mean(moving_mean_test)
            # self.moving_variance_test = tf.reduce_mean(moving_variance_test)
            trainable_vars = tf.trainable_variables()
            total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
            total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            # Adding regularization term
            for var in trainable_vars:
                total_loss1 += self.weight_decay*tf.nn.l2_loss(var)
                total_losses2 = [total_loss2 + self.weight_decay*tf.nn.l2_loss(var) for total_loss2 in total_losses2]
            self.total_loss1 = total_loss1
            self.total_losses2 = total_losses2
            self.lossesa = lossesa # for testing
            self.lossesb = lossesb[-1] # for testing
            self.val_total_loss1 = tf.contrib.copy_graph.get_copied_op(total_loss1, self.graph)
            self.val_total_losses2 = [tf.contrib.copy_graph.get_copied_op(total_losses2[i], self.graph) for i in xrange(len(total_losses2))]
            # after the map_fn
            self.test_act_op = outputas
            self.outputbs = outputbs
            self.image_op = flat_img_inputa
            self.feat_op = fp

            # Initialize solver
            # mom1, mom2 = 0.9, 0.999 # adam defaults
            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.meta_lr, self.global_step, ANNEAL_INTERVAL, 0.5, staircase=True)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_losses2[self.num_updates - 1], global_step=self.global_step)
            # Add summaries
            train_summ = [tf.scalar_summary('Training Pre-update loss', self.total_loss1)] # tf.scalar_summary('Learning rate', learning_rate)
            # train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
            # train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
            # train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
            # train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
            val_summ = [tf.scalar_summary('Validation Pre-update loss', self.val_total_loss1)]
            for j in xrange(self.num_updates):
                train_summ.append(tf.scalar_summary('Training Post-update loss, step %d' % j, self.total_losses2[j]))
                val_summ.append(tf.scalar_summary('Validation Post-update loss, step %d' % j, self.val_total_losses2[j]))
            self.train_summ_op = tf.merge_summary(train_summ)
            self.val_summ_op = tf.merge_summary(val_summ)

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
        n_layers = 4 # TODO TODO this used to be 3.
        layer_size = 40  # TODO TODO This used to be 20.
        dim_hidden = (n_layers - 1)*[layer_size]
        dim_hidden.append(dim_output)
        filter_size = 5
        num_filters = network_config['num_filters']
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        weights = {}
        
        # conv weights
        weights['wc1'] = get_he_weights([filter_size, filter_size, num_channels, num_filters[0]], name='wc1') # 5x5 conv, 1 input, 32 outputs
        weights['wc2'] = get_he_weights([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2') # 5x5 conv, 32 inputs, 64 outputs
        weights['wc3'] = get_he_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3') # 5x5 conv, 32 inputs, 64 outputs

        weights['bc1'] = init_bias([num_filters[0]], name='bc1')
        weights['bc2'] = init_bias([num_filters[1]], name='bc2')
        weights['bc3'] = init_bias([num_filters[2]], name='bc3')
        
        # fc weights
        in_shape = 40 # dimension after feature computation
        for i in xrange(n_layers):
            weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
            in_shape = dim_hidden[i]
        return weights
        
    def forward(self, image_input, state_input, weights, is_training=True, network_config=None):
        n_layers = 4 # 3
        norm_type = self.norm_type
        decay = network_config.get('decay', 0.9)

        # conv_layer_0, _, _ = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, conv_id=0, is_training=is_training)
        # conv_layer_1, _, _ = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2']), norm_type=norm_type, decay=decay, conv_id=1, is_training=is_training)
        # conv_layer_2, moving_mean, moving_variance = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3']), norm_type=norm_type, decay=decay, conv_id=2, is_training=is_training)            
        conv_layer_0 = norm(conv2d(img=image_input, w=weights['wc1'], b=weights['bc1'], strides=[1,2,2,1]), norm_type=norm_type, decay=decay, conv_id=0, is_training=is_training)
        conv_layer_1 = norm(conv2d(img=conv_layer_0, w=weights['wc2'], b=weights['bc2']), norm_type=norm_type, decay=decay, conv_id=1, is_training=is_training)
        conv_layer_2 = norm(conv2d(img=conv_layer_1, w=weights['wc3'], b=weights['bc3']), norm_type=norm_type, decay=decay, conv_id=2, is_training=is_training)[0]        
        _, num_rows, num_cols, num_fp = conv_layer_2.get_shape()
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
        features = tf.reshape(tf.transpose(conv_layer_2, [0,3,1,2]),
                              [-1, num_rows*num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
        fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)

        fp = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp*2])

        fc_input = tf.concat(concat_dim=1, values=[fp, state_input]) # TODO - switch these two?

        fc_output = tf.add(fc_input, 0)
        outputs = []
        for i in xrange(n_layers):
            fc_output = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != n_layers - 1:
                fc_output = tf.nn.relu(fc_output)
            outputs.append(fc_output)
        # return fc_output, fp, moving_mean, moving_variance
        # return fc_output, fp
        return fc_output, fp, conv_layer_2, fc_input, outputs
        
    def construct_model(self, dim_input=27, dim_output=7, batch_size=25, network_config=None):
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
        
        self.inputa = inputa = tf.placeholder(tf.float32) # meta_batch_size x update_batch_size x dim_input
        self.inputb = inputb = tf.placeholder(tf.float32)
        self.actiona = actiona = tf.placeholder(tf.float32)
        self.actionb = actionb = tf.placeholder(tf.float32)
        if self.norm_type:
            self.phase = tf.placeholder(tf.bool, name='phase')
        
        # Construct layers weight & bias
        self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
        self.step_size = tf.abs(tf.Variable(self._hyperparams.get('step_size', 1e-3)))
        # self.step_size = self._hyperparams.get('step_size', 1e-3)
        
        num_updates = self.num_updates
        lossesa, outputsa = [], []
        lossesb = [[] for _ in xrange(num_updates)]
        outputsb = [[] for _ in xrange(num_updates)]
        
        def batch_metalearn(inp):
            inputa, inputb, actiona, actionb = inp #image input
            inputa = tf.reshape(inputa, [-1, dim_input])
            inputb = tf.reshape(inputb, [-1, dim_input])
            actiona = tf.reshape(actiona, [-1, dim_output])
            actionb = tf.reshape(actionb, [-1, dim_output])
            
            # Convert to image dims
            inputa, flat_img_inputa, state_inputa = self.construct_image_input(inputa, x_idx, img_idx, network_config=network_config)
            inputb, _, state_inputb = self.construct_image_input(inputb, x_idx, img_idx, network_config=network_config)
            
            local_outputbs, local_lossesb = [], []
            # Assume fixed data for each update
            inputas = [inputa]*num_updates
            state_inputas = [state_inputa]*num_updates
            actionas = [actiona]*num_updates
            
            # local_outputa, fp, moving_mean, moving_variance = self.forward(inputa, state_inputa, weights, network_config=network_config)
            local_outputa = self.forward(inputa, state_inputa, weights, network_config=network_config)[0]
            # test_outputa, _, moving_mean_test, moving_variance_test = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
            test_outputa, fp, conv_layer_2, fc_input, fc_outputs = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
            local_lossa = euclidean_loss_layer(local_outputa, actiona, None, behavior_clone=True)
            
            gradients = dict(zip(weights.keys(), tf.gradients(local_lossa, weights.values())))
            # Is mask used here?
            fast_weights = dict(zip(weights.keys(), [weights[key] - self.step_size*gradients[key] for key in weights.keys()]))
            output = self.forward(inputb, state_inputb, fast_weights, network_config=network_config)[0]
            local_outputbs.append(output)
            local_lossesb.append(euclidean_loss_layer(output, actionb, None, behavior_clone=True))

            for j in range(num_updates - 1):
                loss = euclidean_loss_layer(self.forward(inputas[j+1], state_inputas[j+1], fast_weights, network_config=network_config)[0], actionas[j+1])
                gradients = dict(zip(fast_weights.keys(), tf.gradients(loss, fast_weights.values())))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.step_size*gradients[key] for key in fast_weights.keys()]))
                output = self.forward(inputb, state_inputb, fast_weights, network_config=network_config)[0]
                local_outputbs.append(output)
                local_lossesb.append(euclidean_loss_layer(output, actionb, None, behavior_clone=True))
            # local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test]
            # local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, conv_layer_2, outputs, test_outputs, mean, variance, moving_mean, moving_variance, moving_mean_new, moving_variance_new]
            local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, conv_layer_2, fc_input, fc_outputs]
            return local_fn_output

        if self.norm_type:
            # initialize batch norm vars.
            # TODO: figure out if this line of code is necessary
            unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))
        
        # out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32]
        out_dtype += 2*[tf.float32] + [[tf.float32]*4]
        result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb), dtype=out_dtype)
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
        N_demos = np.sum(demo['demoO'].shape[0] for i, demo in demos.iteritems())
        print "Number of demos: %d" % N_demos
        # Normalizing observations
        if self.policy.scale is None or self.policy.bias is None:
            obs = np.vstack((demo['demoO'] for demo in demos.values()))
            self.T = obs.shape[1]
            obs = obs.reshape(-1, self._dO)
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
            for key in demos.keys():
                demos[key]['demoO'] = demos[key]['demoO'].reshape(-1, self._dO)
                demos[key]['demoO'][:, self.x_idx] = demos[key]['demoO'][:, self.x_idx].dot(self.policy.scale) + self.policy.bias
                demos[key]['demoO'] = demos[key]['demoO'].reshape(-1, self.T, self._dO)
        idx = range(n_folders)
        shuffle(idx)
        self.demos = demos
        self.val_demos = {key: demos[key] for key in np.array(demos.keys())[idx[:n_val]]}
        self.train_demos = {key: demos[key] for key in np.array(demos.keys())[idx[n_val:]]}
        self.val_idx = sorted(idx[:n_val])
        self.train_idx = sorted(idx[n_val:])

    def generate_data_batch(self, train=True):
        if train:
            demos = self.train_demos
            folder_idx = list(np.array(self.train_idx).copy())
        else:
            demos = self.val_demos
            folder_idx = list(np.array(self.val_idx).copy())
        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        shuffle(folder_idx)
        batch_idx = folder_idx[:batch_size]
        batch_demos = {key: demos[key] for key in batch_idx}
        n_demo = batch_demos[batch_idx[0]]['demoX'].shape[0]
        idx_i = np.random.choice(np.arange(n_demo), replace=False, size=update_batch_size*2)
        U = batch_demos[batch_idx[0]]['demoU'][idx_i]
        O = batch_demos[batch_idx[0]]['demoO'][idx_i]
        for i in xrange(1, batch_size):
            n_demo = batch_demos[batch_idx[i]]['demoX'].shape[0]
            idx_i = np.random.choice(np.arange(n_demo), replace=False, size=update_batch_size*2)
            U = np.concatenate((U, batch_demos[batch_idx[i]]['demoU'][idx_i]))
            O = np.concatenate((O, batch_demos[batch_idx[i]]['demoO'][idx_i]))
        U = U.reshape(batch_size, 2*update_batch_size*self.T, -1)
        O = O.reshape(batch_size, 2*update_batch_size*self.T, -1)
        return O, U
    
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
        log_dir = self._hyperparams['log_dir'] + '_%s' % datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        save_dir = self._hyperparams['log_dir'] + '_model' #'_model_ln'
        train_writer = tf.train.SummaryWriter(log_dir, self.graph)
        # actual training.
        with Timer('Training'):
            for itr in range(TOTAL_ITERS):
                obs, tgt_mu = self.generate_data_batch()
                inputa = obs[:, :self.update_batch_size*self.T, :]
                inputb = obs[:, self.update_batch_size*self.T:, :]
                actiona = tgt_mu[:, :self.update_batch_size*self.T, :]
                actionb = tgt_mu[:, self.update_batch_size*self.T:, :]
                feed_dict = {self.inputa: inputa,
                            self.inputb: inputb,
                            self.actiona: actiona,
                            self.actionb: actionb}
                input_tensors = [self.train_op]
                # if self.use_batchnorm:
                #     feed_dict[self.phase] = 1
                if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
                    input_tensors.extend([self.train_summ_op, self.total_loss1, self.total_losses2[self.num_updates-1]])
                result = self.run(input_tensors, feed_dict=feed_dict)
    
                if itr % SUMMARY_INTERVAL == 0:
                    prelosses.append(result[-2])
                    train_writer.add_summary(result[1], itr)
                    postlosses.append(result[-1])
    
                if itr != 0 and itr % PRINT_INTERVAL == 0:
                    print 'Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(prelosses), np.mean(postlosses))
                    prelosses, postlosses = [], []
    
                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    input_tensors = [self.val_summ_op, self.val_total_loss1, self.val_total_losses2[self.num_updates-1]]
                    val_obs, val_act = self.generate_data_batch(train=False)
                    inputa = val_obs[:, :self.update_batch_size*self.T, :]
                    inputb = val_obs[:, self.update_batch_size*self.T:, :]
                    actiona = val_act[:, :self.update_batch_size*self.T, :]
                    actionb = val_act[:, self.update_batch_size*self.T:, :]
                    feed_dict = {self.inputa: inputa,
                                self.inputb: inputb,
                                self.actiona: actiona,
                                self.actionb: actionb}
                    # if self.use_batchnorm:
                    #     feed_dict[self.phase] = 0
                    results = self.run(input_tensors, feed_dict=feed_dict)
                    train_writer.add_summary(results[0], itr)
                    print 'Test results: average preloss is %.2f, average postloss is %.2f' % (np.mean(results[1]), np.mean(results[2]))
                
                if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == TOTAL_ITERS - 1):
                    # TODO: implement restore
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
                        mkdir_p(gif_dir + 'color_%d/' % idx)
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
