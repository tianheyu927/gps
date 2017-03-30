""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging
import os
import tempfile

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
                # self.gpu_device = self._hyperparams['gpu_id']
                # self.device_string = "/gpu:" + str(self.gpu_device)
                # self._sess = tf.Session(graph=self.graph)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
                tf_config = tf.ConfigProto(gpu_options=gpu_options)
                self._sess = tf_Session(graph=graph, config=tf_config)
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
        self._hyperparams['network_params'].update({'batch_norm': self._hyperparams['batch_norm']})
        self._hyperparams['network_params'].update({'decay': self._hyperparams['decay']})

        self.init_network()

        self.var = self._hyperparams['init_var'] * np.ones(dU)
        # use test action for policy action
        self.policy = TfPolicy(dU, self.obs_tensor, self.test_act_op, self.feat_op, self.image_op,
                               np.zeros(dU), self._sess, self.graph, self.device_string, copy_param_scope=self._hyperparams['copy_param_scope'])
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

        if hyperparams.get('agent', False):
            test_agent = hyperparams['agent']  # Required for sample packing
            if type(test_agent) is not list:
                test_agent = [test_agent]
        demo_file = hyperparams['demo_file']
        
        self.update_batch_size = self._hyperparams.get('update_batch_size', 1)
        self.meta_batch_size = self._hyperparams.get('meta_batch_size', 10)
        self.num_updates = self._hyperparams.get('num_updates', 1)
        self.meta_lr = self._hyperparms.get('lr', 1e-3)
        
        if hyperparams.get('agent', False):
            trainO, trainU, valO, valU = self.extract_supervised_data(demo_file)
            if valO is not None:
                self.update(trainO, trainU, 1.0, 1.0, test_obs=valO,\
                            test_acts=valU, behavior_clone=True)
            else:
                self.update(trainO, trainU, 1.0, 1.0, test_obs=None,\
                            test_acts=None, behavior_clone=True)
            self.eval_success_rate(test_agent)

        self.test_agent = None  # don't pickle agent
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_map_generator = self._hyperparams['network_model']
        with self.graph.as_default():
            with Timer('building TF network'):
                tf_map, fc_vars, last_conv_vars = tf_map_generator(dim_input=self._dO, dim_output=self._dU, batch_size=self.batch_size,
                                          network_config=self._hyperparams['network_params'])
            self.obs_tensor = tf_map.get_input_tensor()
            if not self._hyperparams['network_params'].get('bc', False):
                self.precision_tensor = tf_map.get_precision_tensor()
            self.action_tensor = tf_map.get_target_output_tensor()
            self.act_op = tf_map.get_output_op()
            self.test_act_op = tf_map.get_test_output_op() # used for policy action
            self.feat_op = tf_map.get_feature_op()
            self.image_op = tf_map.get_image_op()  # TODO - make this.
            self.weights = tf_map.get_weights() # get weights from model, used for metalearn
            self.loss_scalar = tf_map.get_loss_op()
            self.val_loss_scalar = tf_map.get_val_loss_op()
            self.debug = tf_map.debug
            if self.uses_vision:
                self.fc_vars = fc_vars
                self.last_conv_vars = last_conv_vars
            else:
                self.fc_vars = None
                self.last_conv_vars = None

            # Setup the gradients. Use test act op
            self.grads = [tf.gradients(self.test_act_op[:,u], self.obs_tensor)[0]
                    for u in range(self._dU)]

    def construct_image_input(self, nn_input, x_idx, img_idx, network_config=None)
        state_input = nn_input[:, 0:x_idx[-1]+1]
        flat_image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]
    
        # image goes through 3 convnet layers
        num_filters = network_config['num_filters']
    
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        image_input = tf.reshape(flat_image_input, [-1, num_channels, im_width, im_height])
        image_input = tf.transpose(image_input, perm=[0,3,2,1])
        return image_input
    
    def construct_weights(self, dim_input=27, dim_output=7, batch_size=25, network_config=None):
        n_layers = 3 # TODO TODO this used to be 3.
        layer_size = 20  # TODO TODO This used to be 20.
        dim_hidden = (n_layers - 1)*[layer_size]
        dim_hidden.append(dim_output)
        pool_size = 2
        filter_size = 5
        weights = {}
        
        # conv weights
        weights['wc1'] = get_he_weights([filter_size, filter_size, num_channels, num_filters[0]], name='wc1') # 5x5 conv, 1 input, 32 outputs
        weights['wc2'] = get_he_weights([filter_size, filter_size, num_filters[0], num_filters[1]], name='wc2') # 5x5 conv, 32 inputs, 64 outputs
        weights['wc3'] = get_he_weights([filter_size, filter_size, num_filters[1], num_filters[2]], name='wc3') # 5x5 conv, 32 inputs, 64 outputs

        weights['bc1'] = init_bias([num_filters[0]], name='bc1')
        weights['bc2'] = init_bias([num_filters[1]], name='bc2')
        weights['bc3'] = init_bias([num_filters[2]], name='bc3')
        
        # fc weights
        in_shape = dim_input
        for i in xrange(n_layers):
            weights['w_%d' % i] = init_weights([in_shape, dimension_hidden[i]], name='w_%d' % i)
            weights['b_%d' % i] = init_bias([dimension_hidden[i]], name='b_%d' % i)
            in_shape = dimension_hidden[i]
        return weights
        
    def forward(self, image_input, weights, network_config=None):
        n_layers = 3
        behavior_clone = network_config.get('bc', False)
        batch_norm = network_config.get('batch_norm', False)
        decay = network_config.get('decay', 0.9)

        conv_layer_0 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'], strides=[1,2,2,1], batch_norm=batch_norm, decay=decay, conv_id=0)
        conv_layer_1 = conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'], batch_norm=batch_norm, decay=decay, conv_id=1)
        conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc3'], b=biases['bc3'], batch_norm=batch_norm, decay=decay, conv_id=2)
        
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

        cur = fc_output
        for i in xrange(n_layers):
            cur = tf.matmul(cur, weights['w_%d' % i]) + weights['b_%d' % i]
            if i != n_layers - 1:
                cur = tf.nn.relu(cur)
        return cur
        
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
    
        # Construct layers weight & bias
        self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
        self.step_size = tf.abs(tf.Variable(self._hyperparams.get('step_size', 1e-3)))
        
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
            inputa = self.construct_image_input(inputa, x_idx, img_idx, network_config=network_config)
            inputb = self.construct_image_input(inputb, x_idx, img_idx, network_config=network_config)
            
            local_outputbs, local_lossesb = [], []
            # Assume fixed data for each update
            inputas = [inputa]*num_updates
            actionas = [actiona]*num_updates
            
            local_outputa = self.forward(inputa, weights, network_config=network_config)
            local_lossa = euclidean_loss_layer(local_outputa, actiona, None, behavior_clone=True)
            
            gradients = dict(zip(weights.keys(), tf.gradients(local_lossa, weights.values())))
            # Is mask used here?
            fast_weights = dict(zip(weights.keys(), [weights[key] - self.step_size*gradients[key] for key in weights.keys()]))
            output = self.forward(inputb, fast_weights, network_config=network_config)
            local_outputbs.append(output)
            local_lossesb.append(euclidean_loss_layer(output, actionb, None, behavior_clone=True))

            for j in range(num_updates - 1):
                loss = euclidean_loss_layer(self.forward(inputas[j+1], fast_weights, network_config=network_config), actionas[j+1])
                gradients = dict(zip(fast_weights.keys(), tf.gradients(loss, fast_weights.values())))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.step_size*gradients[key] for key in fast_weights.keys()]))
                output = self.forward(inputb, fast_weights, network_config=network_config)
                local_outputbs.append(output)
                local_lossesb.append(euclidean_loss_layer(output, actionb, None, behavior_clone=True))
            local_fn_output = [local_outputa, local_outputbs, local_lossa, local_lossesb]
            return local_fn_output
        
        loss = euclidean_loss_layer(a=action, b=fc_output, precision=precision, behavior_clone=behavior_clone)
        val_loss = euclidean_loss_layer(a=action, b=test_output, precision=precision, behavior_clone=behavior_clone)
        
        nnet = TfMap.init_from_lists([nn_input, action, precision], [fc_output, test_output], [weights], [loss, val_loss], fp=fp, image=flat_image_input, debug=training_conv_layer_2) #this is training conv layer
        last_conv_vars = fc_inputs[0] #training fc input
    
        return nnet, fc_vars, last_conv_vars
    
    def extract_supervised_data(self, demo_file):
        demos = extract_demo_dict(demo_file)
        n_val = self._hyperparams['n_val'] # number of demos for testing
        if type(demo_file) is list:
            N = np.sum(demo['demoO'].shape[0] for i, demo in demos.iteritems())
            print "Number of demos: %d" % N
            update_batch_size = self.update_batch_size
            n_demo = demos[0]['demoX'].shape[0]
            idx_i = np.random.choice(np.arange(n_demo), replace=False, size=update_batch_size)
            X = demos[0]['demoX'][idx_i]
            U = demos[0]['demoU'][idx_i]
            O = demos[0]['demoO'][idx_i]
            cond = np.array(demos[0]['demoConditions'])[idx_i]
            for i in xrange(1, len(demo_file)):
                n_demo = demos[i]['demoX'].shape[0]
                idx_i = np.random.choice(np.arange(n_demo), replace=False, size=update_batch_size)
                X = np.concatenate((X, demos[i]['demoX'][idx_i]))
                U = np.concatenate((U, demos[i]['demoU'][idx_i]))
                O = np.concatenate((O, demos[i]['demoO'][idx_i]))
                cond = np.concatenate((cond, np.array(demos[i]['demoConditions'])[idx_i]))
            print "Number of few-shot demos is %d" % (X.shape[0])
        else:
            N = demos['demoO'].shape[0]
            print "Number of demos: %d" % N
            X = demos['demoX']
            U = demos['demoU']
            O = demos['demoO']
            cond = demos['demoConditions']
        if n_val != 0:
            valO = O[:n_val]
            valX = X[:n_val]
            valU = U[:n_val]
            val_cond = cond[:n_val]
            O = O[n_val:]
            X = X[n_val:]
            U = U[n_val:]
            cond = cond[n_val:]
        else:
            valO = None
            valX = None
            valU = None
            val_cond = None
        self.train_set = {"trainO": O, "trainX": X, "trainU": U, "train_conditions": cond}   
        self.val_set = {"valO": valO, "valX": valX, "valU": valU, "val_conditions": val_cond}

        return O, U, valO, valU

    def sample(self, agent, idx, conditions, N=1, testing=False):
        samples = []
        for i in xrange(len(conditions)):
            for j in xrange(N):
                if 'record_gif' in self._hyperparams:
                    gif_config = self._hyperparams['record_gif']
                    if j < gif_config.get('gifs_per_condition', float('inf')):
                        gif_fps = gif_config.get('fps', None)
                        if testing:
                            gif_dir = gif_config.get('test_gif_dir', self._hyperparams['plot_dir'] + 'test_gifs/')
                        else:
                            gif_dir = gif_config.get('gif_dir', self._hyperparams['plot_dir'] + 'gifs/')
                        mkdir_p(gif_dir)
                        gif_name = os.path.join(gif_dir,'color%d.cond%d.samp%d.gif' % (idx, i, j))
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
    
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt, iter_count=None, fc_only=False,\
                test_obs=None, test_acts=None, behavior_clone=False):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
            fc_only: If true, don't train end-to-end.
            test_obs: Numpy array of test observations, Ntest x T x dO.
            test_acts: Numpy array of test actions, Ntest x T x dU.
            behavior_clone: If true, run policy cloning
        Returns:
            A tensorflow object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO
        # import pdb; pdb.set_trace()
        if not behavior_clone:
            # TODO - Make sure all weights are nonzero?

            # Save original tgt_prc.
            tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

            # Renormalize weights.
            tgt_wt *= (float(N * T) / np.sum(tgt_wt))
            # Allow weights to be at most twice the robust median.
            mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
            for n in range(N):
                for t in range(T):
                    tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
            # Robust median should be around one.
            tgt_wt /= mn

            # Reshape inputs.
            tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
            tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

            # Fold weights into tgt_prc.
            tgt_prc = tgt_wt * tgt_prc

            # TODO: Find entries with very low weights?

        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        if test_obs is not None and test_acts is not None:
            test_obs = np.reshape(test_obs, (-1, dO))
            test_acts = np.reshape(test_acts, (-1, dU))
        # Normalize obs, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(
                1.0 / np.maximum(np.std(obs[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = - np.mean(
                obs[:, self.x_idx].dot(self.policy.scale), axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        conv_loss_history = []
        # conv_val_loss_history = []
        loss_history = []
        val_loss_history = []
        plot_dir = self._hyperparams.get('plot_dir', '/home/kevin/gps/')
        np.random.shuffle(idx)

        TOTAL_ITERS = self._hyperparams['iterations']
        # actual training.
        for i in range(TOTAL_ITERS):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            if not behavior_clone:
                feed_dict = {self.obs_tensor: obs[idx_i],
                             self.action_tensor: tgt_mu[idx_i],
                             self.precision_tensor: tgt_prc[idx_i]}
            else:
                feed_dict = {self.obs_tensor: obs[idx_i],
                             self.action_tensor: tgt_mu[idx_i]}
            train_loss = self.solver(feed_dict, self._sess, device_string=self.device_string)
            average_loss += train_loss
            if (i+1) % 50 == 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i+1, average_loss / 50)
                print ('supervised tf loss is ' + str(average_loss))
                average_loss = 0
                if behavior_clone:
                    loss_history.append(train_loss)
                    if test_obs is not None and test_acts is not None:
                        val_feed_dict = {self.obs_tensor: test_obs,
                                        self.action_tensor: test_acts}
                        with tf.device(self.device_string):
                            val_loss_history.append(self.run(self.val_loss_scalar, feed_dict=val_feed_dict))
        if behavior_clone:
            plt.figure()
            plt.plot(50*(np.arange(TOTAL_ITERS/50)+1), loss_history, color='red', linestyle='-')
            if test_obs is not None:
                plt.plot(50*(np.arange(TOTAL_ITERS/50)+1), val_loss_history, color='blue', linestyle=':')
            plot_dir = self._hyperparams.get('plot_dir', '/home/kevin/gps/')
            plt.savefig(plot_dir + 'actual_loss_history.png')
            plt.show()
        feed_dict = {self.obs_tensor: obs}
        num_values = obs.shape[0]
        if self.feat_op is not None:
            self.feat_vals = self.solver.get_var_values(self._sess, self.feat_op, feed_dict, num_values, self.batch_size)
        if self.debug is not None:
            self.debug_vals = self.solver.get_var_values(self._sess, self.debug, feed_dict, num_values, self.batch_size)
        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

    def eval_success_rate(self, test_agent):
        # TODO: sample on train and val sets and calucate the success rate.
        assert type(test_agent) is list
        success_thresh = test_agent[0]['filter_demos'].get('success_upper_bound', 0.05)
        state_idx = np.array(list(test_agent[0]['filter_demos'].get('state_idx', range(4, 7))))
        train_dists = []
        val_dists = []
        # Calculate the validation set success rate.
        for i in xrange(self._hyperparams['n_val']):
            agent = test_agent[i]['type'](test_agent[i])
            # Get the validation pos body offset idx!
            val_conditions = self.val_set['val_conditions'][i*self.update_batch_size:(i+1)*self.update_batch_size]
            # Sample on validation conditions.
            val_sample_list = self.sample(agent, i, val_conditions, N=1, testing=True)
            # Calculate val distances
            X_val = val_sample_list.get_X()
            target_eepts = np.squeeze(np.array(test_agent[i]['target_end_effector'])[val_conditions])[:3]
            val_dists.extend([np.nanmin(np.linalg.norm(X_val[j, :, state_idx].T - target_eepts, axis=1)) \
                                for j in xrange(X_val.shape[0])])

        # Calculate the training set success rate.
        for i in xrange(len(test_agent)-self._hyperparams['n_val']):
            agent = test_agent[i+self._hyperparams['n_val']]['type'](test_agent[i+self._hyperparams['n_val']])
            # Get the train pos body offset idx!
            train_conditions = self.train_set['train_conditions'][i*self.update_batch_size:(i+1)*self.update_batch_size]
            # Sample on train conditions.
            train_sample_list = self.sample(agent, i+self._hyperparams['n_val'], train_conditions, N=1)
            # Calculate train distances
            X_train = train_sample_list.get_X()
            target_eepts = np.squeeze(np.array(test_agent[i+self._hyperparams['n_val']]['target_end_effector'])[train_conditions])[:3]
            train_dists.extend([np.nanmin(np.linalg.norm(X_train[j, :, state_idx].T - target_eepts, axis=1)) \
                                for j in xrange(X_train.shape[0])])
        import pdb; pdb.set_trace()

        print "Training success rate is %.5f" % (np.array(train_dists) <= success_thresh).mean()
        print "Validation success rate is %.5f" % (np.array(val_dists) <= success_thresh).mean()
