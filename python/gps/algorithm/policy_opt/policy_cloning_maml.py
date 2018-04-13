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
from gps.algorithm.policy.tf_policy_maml import TfPolicyMAML
from gps.algorithm.policy_opt.config import POLICY_OPT_TF
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import *
from gps.algorithm.policy_opt.tf_utils import TfSolver
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
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)#1.0
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
        self.noisy_demo_gif_dir = self._hyperparams.get('noisy_demo_gif_dir', None)
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
        
        # For loading demos
        if hyperparams.get('agent', False):
            test_agent = hyperparams['agent']
            # test_agent = hyperparams['agent'][:300]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-150:])
            # test_agent = hyperparams['agent'][:1500]  # Required for sampling
            # test_agent.extend(hyperparams['agent'][-150:])
            if type(test_agent) is not list:
                test_agent = [test_agent]

        demo_file = hyperparams['demo_file']
        # demo_file = hyperparams['demo_file'][:300]
        # demo_file.extend(hyperparams['demo_file'][-150:])
        # demo_file = hyperparams['demo_file']#[:186]
        # demo_file.extend(hyperparams['demo_file'][-61:])
        if self._hyperparams.get('use_noisy_demos', False):
            noisy_demo_file = hyperparams['noisy_demo_file']
            # noisy_demo_file = hyperparams['noisy_demo_file'][:300]
            # noisy_demo_file.extend(hyperparams['noisy_demo_file'][-150:])
        
        # if hyperparams.get('agent', False):
        self.restore_iter = hyperparams.get('restore_iter', 0)
        if self._hyperparams.get('use_noisy_demos', False):
            self.extract_supervised_data(noisy_demo_file, noisy=True)
        self.extract_supervised_data(demo_file)
        for i in xrange(len(test_agent)):
            # agent = test_agent[i]['type'](test_agent[i])
            # import pdb; pdb.set_trace()
            models = test_agent[i]['models']
            # import pdb; pdb.set_trace()
            for j in xrange(len(models)):
                # if j in self.demos[i]['demoConditions']:
                print 'saving task %d xml %d' % (i, j)
                models[j].save('/home/kevin/gps/data/sim_vision_reach_test_xmls/task_%d_cond_%d.xml' % (i, j))
        os._exit(1)
        # if hyperparams.get('test', False):
        #     import pdb; pdb.set_trace()
        #     import pickle
        #     # with open('/home/kevin/gps/scale_and_bias_push_consistent_12.pkl', 'wb') as f:
        #     with open('/home/kevin/gps/scale_and_bias_reach_vr_3_obj.pkl', 'wb') as f:
        #         pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
        if not hyperparams.get('test', False):
            self.generate_batches(noisy=self._hyperparams.get('use_noisy_demos', False))

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
        os._exit(1)

        self.test_agent = None  # don't pickle agent
        self.val_demos = None # don't pickle demos
        self.train_demos = None
        self.demos = None
        self.policy.demos = None
        self.policy.selected_demoO = None
        self.policy.selected_demoU = None
        if self._hyperparams.get('agent', False):
            del self._hyperparams['agent']

    def init_network(self, graph, input_tensors=None, depth_tensors=None, restore_iter=0, prefix='Training_'):
		""" Helper method to initialize the tf networks used """
		with graph.as_default():
			image_tensors, depth_tensors = None, None
			if self._hyperparams.get('use_vision', True):
				if 'Training' in prefix:
					image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter)
					if self._hyperparams.get('use_depth', False):
						depth_tensors = self.make_batch_tensor(self._hyperparams['network_params'], use_depth=True, restore_iter=restore_iter)
				elif 'Validation' in prefix:
					image_tensors = self.make_batch_tensor(self._hyperparams['network_params'], restore_iter=restore_iter, train=False)
					if self._hyperparams.get('use_depth', False):
						depth_tensors = self.make_batch_tensor(self._hyperparams['network_params'], use_depth=True, restore_iter=restore_iter, train=False)
			if image_tensors is not None:
				# image_tensors = tf.reshape(image_tensors, [self.meta_batch_size, (self.update_batch_size+1)*self.T, -1])
				# inputa = tf.slice(image_tensors, [0, 0, 0], [-1, self.update_batch_size*self.T, -1])
				# inputb = tf.slice(image_tensors, [0, self.update_batch_size*self.T, 0], [-1, -1, -1])
				inputa = image_tensors[:, :self.update_batch_size*self.T, :]
				inputb = image_tensors[:, self.update_batch_size*self.T:, :]
				input_tensors = {'inputa': inputa, 'inputb': inputb}
				if depth_tensors is not None:
					deptha = depth_tensors[:, :self.update_batch_size*self.T, :]
					depthb = depth_tensors[:, self.update_batch_size*self.T:, :]
					input_tensors.update({'deptha': deptha, 'depthb': depthb})
			else:
				input_tensors = None
			with Timer('building TF network'):
				result = self.construct_model(input_tensors=input_tensors, prefix=prefix, dim_input=self._dO, dim_output=self._dU,
										  network_config=self._hyperparams['network_params'])
			# outputas, outputbs, test_outputa, lossesa, lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test = result
			stop_lossb, gripper_lossb = None, None
			if self._hyperparams.get('stop_signal', False) and self._hyperparams.get('gripper_command_signal', False):
				outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, control_lossesb, flat_img_inputa, flat_img_inputb, final_eept, fast_weights_values, gradients, acosine_lossb, pick_eept_lossa, pick_eept_lossb, task_label_loss, task_label_pred, stop_lossb, gripper_lossb = result
			elif self._hyperparams.get('gripper_command_signal', False):
				outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, control_lossesb, flat_img_inputa, flat_img_inputb, final_eept, fast_weights_values, gradients, acosine_lossb, pick_eept_lossa, pick_eept_lossb, task_label_loss, task_label_pred, gripper_lossb = result
			elif self._hyperparams.get('stop_signal', False):
				outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, control_lossesb, flat_img_inputa, flat_img_inputb, final_eept, fast_weights_values, gradients, acosine_lossb, pick_eept_lossa, pick_eept_lossb, task_label_loss, task_label_pred, stop_lossb = result
			else:                
				outputas, outputbs, test_output, lossesa, lossesb, final_eept_lossesb, control_lossesb, flat_img_inputa, flat_img_inputb, final_eept, fast_weights_values, gradients, acosine_lossb, pick_eept_lossa, pick_eept_lossb, task_label_loss, task_label_pred, = result
			if 'Testing' in prefix:
				self.obs_tensor = self.obsa
				self.state_tensor = self.statea
				self.action_tensor = self.actiona
				self.act_op = outputas
				self.outputbs = outputbs
				self.test_act_op = test_output # post-update output
				toy_output_variable = tf.add(test_output, 0, name='output_action')
				toy_final_eept_var = tf.add(final_eept, 0, name='final_ee')
				self.image_op = flat_img_inputb
				self.fast_weights = {key: fast_weights_values[i] for i, key in enumerate(self.sorted_weight_keys)}
			if 'Training' in prefix:
				self.train_act_op = test_output # post-update output
				self.task_label_pred = task_label_pred

			trainable_vars = tf.trainable_variables()
			if self._hyperparams.get('pred_pick_eept', False) and gripper_lossb is not None:
				pick_eept_loss_eps = self._hyperparams.get('pick_eept_loss_eps', 0.5)
				gripper_command_signal_eps = self._hyperparams.get('gripper_command_signal_eps', 5.0)
				if not self._hyperparams.get('use_noisy_demos', False):
					lossesa = lossesa - pick_eept_loss_eps * pick_eept_lossa - gripper_command_signal_eps * gripper_lossa
				lossesb[-1] = lossesb[-1] - pick_eept_loss_eps * pick_eept_lossb - gripper_command_signal_eps * gripper_lossb
				if 'Training' in prefix:
					self.total_pick_eept_loss2 = tf.reduce_sum(pick_eept_lossb) / tf.to_float(self.train_pick_batch_size)
					total_gripper_loss2 = tf.reduce_sum(gripper_lossb) / tf.to_float(self.train_pick_batch_size)
					total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
					if self._hyperparams.get('use_noisy_demos', False):            
						total_loss1 += pick_eept_loss_eps * tf.reduce_sum(pick_eept_lossa) / tf.to_float(self.train_pick_batch_size)
					total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
					total_losses2[-1] += pick_eept_loss_eps * self.total_pick_eept_loss2 + gripper_command_signal_eps * total_gripper_loss2
				elif 'Validation' in prefix:
					self.val_total_pick_eept_loss2 = tf.reduce_sum(pick_eept_lossb) / tf.to_float(self.val_pick_batch_size)
					val_total_gripper_loss2 = tf.reduce_sum(gripper_lossb) / tf.to_float(self.val_pick_batch_size)
					total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
					if self._hyperparams.get('use_noisy_demos', False):            
						total_loss1 += pick_eept_loss_eps * tf.reduce_sum(pick_eept_lossa) / tf.to_float(self.val_pick_batch_size)
					total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
					total_losses2[-1] += pick_eept_loss_eps * self.val_total_pick_eept_loss2 + gripper_command_signal_eps * val_total_gripper_loss2
			else:                
				total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
				total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
			total_final_eept_losses2 = [tf.reduce_sum(final_eept_lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
			total_control_losses2 = [tf.reduce_sum(control_lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
			total_task_label_loss2 = tf.reduce_sum(task_label_loss) / tf.to_float(self.meta_batch_size)
			total_stop_loss2 = None
			if stop_lossb is not None:
				total_stop_loss2 = tf.reduce_sum(stop_lossb) / tf.to_float(self.meta_batch_size)
			total_acosine_loss2 = tf.reduce_sum(acosine_lossb) / tf.to_float(self.meta_batch_size)
			
			if 'Training' in prefix:
				self.total_loss1 = total_loss1
				self.total_losses2 = total_losses2
				self.total_final_eept_losses2 = total_final_eept_losses2
				self.total_control_losses2 = total_control_losses2
				self.total_acosine_loss2 = total_acosine_loss2
				self.total_action_loss2 = total_control_losses2[-1]
				self.total_task_label_loss2 = total_task_label_loss2
				if total_stop_loss2 is not None:
					self.total_stop_loss2 = total_stop_loss2
					self.total_action_loss2 = self.total_action_loss2 - self._hyperparams.get('stop_signal_eps', 10.0)*total_stop_loss2
				if total_gripper_loss2 is not None:
					self.total_gripper_loss2 = total_gripper_loss2
					self.total_action_loss2 = self.total_action_loss2 - self._hyperparams.get('gripper_command_signal_eps', 5.0)*total_gripper_loss2
				self.lossesa = lossesa # for testing
				self.lossesb = lossesb[-1] # for testing
			elif 'Validation' in prefix:
				self.val_total_loss1 = total_loss1
				self.val_total_losses2 = total_losses2
				self.val_total_final_eept_losses2 = total_final_eept_losses2
				self.val_total_pick_eept_loss2 = tf.reduce_sum(pick_eept_lossb) / tf.to_float(self.val_pick_batch_size)
				self.val_total_control_losses2 = total_control_losses2
				self.val_total_acosine_loss2 = total_acosine_loss2
				self.val_total_action_loss2 = total_control_losses2[-1]
				self.val_total_task_label_loss2 = total_task_label_loss2
				if total_stop_loss2 is not None:
					self.val_total_stop_loss2 = total_stop_loss2
					self.val_total_action_loss2 = self.val_total_action_loss2 - self._hyperparams.get('stop_signal_eps', 10.0)*total_stop_loss2
				if val_total_gripper_loss2 is not None:
					self.val_total_gripper_loss2 = val_total_gripper_loss2
					self.val_total_action_loss2 = self.val_total_action_loss2 - self._hyperparams.get('gripper_command_signal_eps', 5.0)*val_total_gripper_loss2
			# self.val_total_loss1 = tf.contrib.copy_graph.get_copied_op(total_loss1, self.graph)
			# self.val_total_losses2 = [tf.contrib.copy_graph.get_copied_op(total_losses2[i], self.graph) for i in xrange(len(total_losses2))]
 
			# Initialize solver
			# mom1, mom2 = 0.9, 0.999 # adam defaults
			# self.global_step = tf.Variable(0, trainable=False)
			# learning_rate = tf.train.exponential_decay(self.meta_lr, self.global_step, ANNEAL_INTERVAL, 0.5, staircase=True)
			# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# with tf.control_dependencies(update_ops):
			# self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.total_losses2[self.num_updates - 1], global_step=self.global_step)
			if self._hyperparams.get('debug_align', False):
				im_height = self._hyperparams['network_params']['image_height']
				im_width = self._hyperparams['network_params']['image_width']
				num_channels = self._hyperparams['network_params']['image_channels']
				flat_img_inputa = tf.reshape(flat_img_inputa, [self.meta_batch_size, self.T, num_channels, im_width, im_height])
				flat_img_inputa = tf.transpose(flat_img_inputa, perm=[0,1,4,3,2])
				flat_img_inputb = tf.reshape(flat_img_inputb, [self.meta_batch_size, self.T, num_channels, im_width, im_height])
				flat_img_inputb = tf.transpose(flat_img_inputb, perm=[0,1,4,3,2])
				if self._hyperparams.get('use_depth', False) and 'Testing' not in prefix:
					depth_img_inputa = tf.reshape(deptha, [self.meta_batch_size, self.T, 1, im_width, im_height])
					depth_img_inputa = tf.transpose(depth_img_inputa, perm=[0,1,4,3,2])
					depth_img_inputb = tf.reshape(depthb, [self.meta_batch_size, self.T, 1, im_width, im_height])
					depth_img_inputb = tf.transpose(depth_img_inputb, perm=[0,1,4,3,2])
			if 'Training' in prefix:
				optimizer_name = self._hyperparams.get('optimizer', 'adam')
				if optimizer_name == 'adam':
					optimizer = tf.train.AdamOptimizer(self.meta_lr)
				elif optimizer_name == 'momentum':
					momentum = self._hyperparams.get('momentum', 0.9)
					optimizer = tf.train.MomentumOptimizer(self.meta_lr, momentum)
				else:
					raise NotImplementedError
				if self._hyperparams.get('clip_metagradient', False):
					clip_meta_max, clip_meta_min = self._hyperparams['clip_meta_max'], self._hyperparams['clip_meta_min']
					self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[self.num_updates-1])
					gvs = [(tf.clip_by_value(grad, clip_meta_min, clip_meta_max), var) for grad, var in gvs]
					self.train_op = optimizer.apply_gradients(gvs)
				else:
					self.train_op = optimizer.minimize(self.total_losses2[self.num_updates - 1])
				# Add summaries
				summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.total_loss1)] # tf.scalar_summary('Learning rate', learning_rate)
				# train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
				# train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
				# train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
				# train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
				if self._hyperparams.get('debug_align', False):
					for i in xrange(self.meta_batch_size):
						summ.append(tf.summary.image('Training_imagea_%d' % i, flat_img_inputa[i, -2:, :, :, :]*255.0, max_outputs=1))
						summ.append(tf.summary.image('Training_imageb_%d' % i, flat_img_inputb[i, -2:, :, :, :]*255.0, max_outputs=1))
						if self._hyperparams.get('use_depth', False):
							summ.append(tf.summary.image('Training_depth_imagea_%d' % i, depth_img_inputa[i, -2:, :, :, :], max_outputs=1))
							summ.append(tf.summary.image('Training_depth_imageb_%d' % i, depth_img_inputb[i, -2:, :, :, :], max_outputs=1))
				for j in xrange(self.num_updates):
					summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.total_losses2[j]))
					summ.append(tf.summary.scalar(prefix + 'Post-update_control_loss_step_%d' % j, self.total_control_losses2[j]))
					summ.append(tf.summary.scalar(prefix + 'Post-update_final_eept_loss_step_%d' % j, self.total_final_eept_losses2[j]))
					for k in xrange(len(self.sorted_weight_keys)):
						summ.append(tf.summary.histogram('Gradient_of_%s_step_%d' % (self.sorted_weight_keys[k], j), gradients[j][k]))
						# summ.append(tf.summary.scalar('Gradient_norm_of_%s_step_%d' % (self.sorted_weight_keys[k], j), tf.sqrt(tf.reduce_sum(tf.square(gradients[j][k])))))
				if stop_lossb is not None:
					summ.append(tf.summary.scalar(prefix + 'Post-update_stop_loss', self.total_stop_loss2))
					summ.append(tf.summary.scalar(prefix + 'Post-update_action_loss', self.total_action_loss2))
				if gripper_lossb is not None:
					summ.append(tf.summary.scalar(prefix + 'Post-update_gripper_loss', self.total_gripper_loss2))
				if self._hyperparams.get('use_acosine_loss', False):
					summ.append(tf.summary.scalar(prefix + 'Post-update_acosine_loss', self.total_acosine_loss2))
				if self._hyperparams.get('pred_pick_eept', False):
					summ.append(tf.summary.scalar(prefix + 'Post-update_pick_eept_loss', self.total_pick_eept_loss2))
				if 'num_learned_loss' in self._hyperparams:
					summ.append(tf.summary.scalar(prefix + 'task_label_loss', self.total_task_label_loss2))
				if self._hyperparams.get('clip_metagradient', False):
					for grad, var in self.gvs:
						summ.append(tf.summary.histogram('meta_grad_'+ var.name.split('/')[-1][:-2], grad))
				# summ.append(tf.summary.scalar('Step size', self.step_size))
				self.train_summ_op = tf.summary.merge(summ)
			elif 'Validation' in prefix:
				# Add summaries
				summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.val_total_loss1)] # tf.scalar_summary('Learning rate', learning_rate)
				# train_summ.append(tf.scalar_summary('Moving Mean', self.moving_mean))
				# train_summ.append(tf.scalar_summary('Moving Variance', self.moving_variance))
				# train_summ.append(tf.scalar_summary('Moving Mean Test', self.moving_mean_test))
				# train_summ.append(tf.scalar_summary('Moving Variance Test', self.moving_variance_test))
				if self._hyperparams.get('debug_align', False):
					for i in xrange(self.meta_batch_size):
						summ.append(tf.summary.image('Validation_imagea_%d' % i, flat_img_inputa[i, -2:, :, :, :]*255.0, max_outputs=1))
						summ.append(tf.summary.image('Validation_imageb_%d' % i, flat_img_inputb[i, -2:, :, :, :]*255.0, max_outputs=1))
						if self._hyperparams.get('use_depth', False):
							summ.append(tf.summary.image('Validation_depth_imagea_%d' % i, depth_img_inputa[i, -2:, :, :, :], max_outputs=1))
							summ.append(tf.summary.image('Validation_depth_imageb_%d' % i, depth_img_inputb[i, -2:, :, :, :], max_outputs=1))

				for j in xrange(self.num_updates):
					summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.val_total_losses2[j]))
					summ.append(tf.summary.scalar(prefix + 'Post-update_control_loss_step_%d' % j, self.val_total_control_losses2[j]))
					summ.append(tf.summary.scalar(prefix + 'Post-update_final_eept_loss_step_%d' % j, self.val_total_final_eept_losses2[j]))
				if stop_lossb is not None:
					summ.append(tf.summary.scalar(prefix + 'Post-update_stop_loss', self.val_total_stop_loss2))
					summ.append(tf.summary.scalar(prefix + 'Post-update_action_loss', self.val_total_action_loss2))
				if gripper_lossb is not None:
					summ.append(tf.summary.scalar(prefix + 'Post-update_gripper_loss', self.val_total_gripper_loss2))
				if self._hyperparams.get('use_acosine_loss', False):
					summ.append(tf.summary.scalar(prefix + 'Post-update_acosine_loss', self.val_total_acosine_loss2))
				if self._hyperparams.get('pred_pick_eept', False):
					summ.append(tf.summary.scalar(prefix + 'Post-update_pick_eept_loss', self.val_total_pick_eept_loss2))
				if 'num_learned_loss' in self._hyperparams:
					summ.append(tf.summary.scalar(prefix + 'task_label_loss', self.val_total_task_label_loss2))
				self.val_summ_op = tf.summary.merge(summ)
				
	def construct_image_input(self, nn_input, x_idx, img_idx, use_depth=False, light_augs=None, network_config=None):
		state_input = nn_input[:, 0:x_idx[-1]+1]
		flat_image_input = nn_input[:, x_idx[-1]+1:img_idx[-1]+1]
		if use_depth:
			depth_input = nn_input[:, img_idx[-1]+1:]
		else:
			depth_input = None
	
		# image goes through 3 convnet layers
		num_filters = network_config['num_filters']
	
		im_height = network_config['image_height']
		im_width = network_config['image_width']
		num_channels = network_config['image_channels']
		image_input = tf.reshape(flat_image_input, [-1, num_channels, im_width, im_height])
		image_input = tf.transpose(image_input, perm=[0,3,2,1])
		if use_depth:
			depth_input = tf.reshape(depth_input, [-1, 1, im_width, im_height])
			depth_input = tf.transpose(depth_input, perm=[0,3,2,1])
			
		if self._hyperparams.get('aggressive_light_aug', False) and light_augs is not None:
			img_hsv = tf.image.rgb_to_hsv(image_input)
			img_h = img_hsv[..., 0]
			img_s = img_hsv[..., 1]
			img_v = img_hsv[..., 2]
			# img_h = tf.clip_by_value(img_h + light_augs[0], 0., 1.)
			# img_s = tf.clip_by_value(img_s + light_augs[1], 0., 1.)
			img_v = tf.clip_by_value(img_v + light_augs, 0., 1.)
			img_hsv = tf.stack([img_h, img_s, img_v], 3)
			image_rgb = tf.image.hsv_to_rgb(img_hsv)
			image_input = image_rgb
			flat_image_input = tf.reshape(tf.transpose(image_input, perm=[0,3,2,1]), [-1, num_channels*im_width*im_height])
		if self._hyperparams.get('pretrain', False):
			image_input = image_input * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
			# 'RGB'->'BGR'
			image_input = image_input[:, :, :, ::-1]
		if use_depth:
			image_input = tf.concat(axis=3, values=[image_input, depth_input])
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
				# if self._hyperparams.get('use_depth', False):
				#     weights['depth_context'] = safe_get('depth_context', initializer=tf.zeros([im_height, im_width, 1], dtype=tf.float32))
				if self._hyperparams.get('normalize_img_context', False):
					weights['img_context'] = tf.clip_by_value(weights['img_context'], 0., 1.)
					# if self._hyperparams.get('use_depth', False):
					#     weights['depth_context'] = tf.clip_by_value(weights['depth_context'], 0., 1.)
			for i in xrange(n_conv_layers):
				# if not pretrain or (i != 0 and i != 1):
				if i == 0 and self._hyperparams.get('use_depth', False):
					num_filters_depth = 16
					if self.norm_type == 'selu':
						weights['wc_depth_%d' % (i+1)] = init_conv_weights_snn([filter_sizes[i], filter_sizes[i], 1, num_filters_depth], name='wc_depth_%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
					elif initialization == 'xavier':                
						weights['wc_depth_%d' % (i+1)] = init_conv_weights_xavier([filter_sizes[i], filter_sizes[i], 1, num_filters_depth], name='wc_depth_%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
					elif initialization == 'random':
						weights['wc_depth_%d' % (i+1)] = init_weights([filter_sizes[i], filter_sizes[i], 1, num_filters_depth], name='wc_depth_%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
					else:
						raise NotImplementedError
					weights['bc_depth_%d' % (i+1)] = init_bias([num_filters_depth], name='bc_depth_%d' % (i+1))
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
					if i == 0 and not pretrain and self._hyperparams.get('use_depth', False):
						fan_in += 16
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
					if self._hyperparams.get('use_depth', False):
						fan_in += 16

			# fc weights
			# in_shape = 40 # dimension after feature computation
			in_shape = self.conv_out_size
			if not self._hyperparams.get('no_state', False) and not (self._hyperparams.get('zero_state', False) and self._hyperparams.get('use_lstm', False)):
				in_shape += len(self.x_idx) # hard-coded for last conv layer output
			if self._hyperparams.get('learn_final_eept', False) and not self._hyperparams.get('final_state', False):
				final_eept_range = self._hyperparams['final_eept_range']
				final_eept_in_shape = self.conv_out_size
				if self._hyperparams.get('use_state_context', False):
					weights['context_final_eept'] = safe_get('context_final_eept', initializer=tf.zeros([self._hyperparams.get('context_dim', 10)], dtype=tf.float32))
					final_eept_in_shape += self._hyperparams.get('context_dim', 10)
				n_layers_ee = network_config.get('n_layers_ee', 1)
				layer_size_ee = [self._hyperparams.get('layer_size_ee', 40)]*(n_layers_ee-1)
				layer_size_ee.append(len(final_eept_range))
				for i in xrange(n_layers_ee):
					weights['w_ee_%d' % i] = init_weights([final_eept_in_shape, layer_size_ee[i]], name='w_ee_%d' % i)
					weights['b_ee_%d' % i] = init_bias([layer_size_ee[i]], name='b_ee_%d' % i)
					if i == n_layers_ee - 1 and self._hyperparams.get('two_heads', False) and self._hyperparams.get('no_final_eept', False):
						two_head_in_shape = final_eept_in_shape
						if network_config.get('temporal_conv_2_head_ee', False):
							temporal_kernel_size = network_config.get('1d_kernel_size', 2)
							temporal_num_filters = network_config.get('1d_num_filters_ee', [32, 32, 32])
							temporal_num_filters[-1] = len(final_eept_range)
							for j in xrange(len(temporal_num_filters)):
								if j != len(temporal_num_filters) - 1:
									weights['w_1d_conv_2_head_ee_%d' % j] = init_weights([temporal_kernel_size, two_head_in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_ee_%d' % j)
									weights['b_1d_conv_2_head_ee_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_ee_%d' % j)
									two_head_in_shape = temporal_num_filters[j]
								else:
									weights['w_1d_conv_2_head_ee_%d' % j] = init_weights([1, two_head_in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_ee_%d' % j)
									weights['b_1d_conv_2_head_ee_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_ee_%d' % j)
									if 'num_learned_loss' in self._hyperparams:
										num_learned_loss = self._hyperparams['num_learned_loss']
										weights['w_1d_conv_2_head_tasklabel_%d' % j] = init_weights([1, two_head_in_shape, num_learned_loss], name='w_1d_conv_2_head_tasklabel_%d' % j)
										weights['b_1d_conv_2_head_tasklabel_%d' % j] = init_bias([num_learned_loss], name='b_1d_conv_2_head_tasklabel_%d' % j)
						else:
							n_layers_ee_2_head = self._hyperparams.get('n_layers_ee_2_head', 1)
							layer_size_ee_2_head = [self._hyperparams.get('layer_size_ee', 40)]*(n_layers_ee_2_head-1)
							layer_size_ee_2_head.append(len(final_eept_range))
							for j in xrange(n_layers_ee_2_head):
								weights['w_ee_two_heads_%d' % j] = init_weights([two_head_in_shape, layer_size_ee_2_head[j]], name='w_ee_two_heads_%d' % j)
								weights['b_ee_two_heads_%d' % j] = init_bias([layer_size_ee_2_head[j]], name='b_ee_two_heads_%d' % j)
								two_head_in_shape = layer_size_ee_2_head[j]
					final_eept_in_shape = layer_size_ee[i]
				in_shape += (len(final_eept_range))
			if self._hyperparams.get('pred_pick_eept', False) and not self._hyperparams.get('final_state', False):
				pick_eept_range = self._hyperparams['pick_eept_range']
				pick_eept_in_shape = self.conv_out_size
				if self._hyperparams.get('use_state_context', False):
					weights['context_pick_eept'] = safe_get('context_pick_eept', initializer=tf.zeros([self._hyperparams.get('context_dim', 10)], dtype=tf.float32))
					pick_eept_in_shape += self._hyperparams.get('context_dim', 10)
				n_layers_pick_ee = network_config.get('n_layers_pick_ee', 1)
				layer_size_pick_ee = [self._hyperparams.get('layer_size_pick_ee', 40)]*(n_layers_pick_ee-1)
				layer_size_pick_ee.append(len(pick_eept_range))
				for i in xrange(n_layers_pick_ee):
					weights['w_pick_ee_%d' % i] = init_weights([pick_eept_in_shape, layer_size_pick_ee[i]], name='w_pick_ee_%d' % i)
					weights['b_pick_ee_%d' % i] = init_bias([layer_size_pick_ee[i]], name='b_pick_ee_%d' % i)
					if i == n_layers_pick_ee - 1 and self._hyperparams.get('two_heads', False) and self._hyperparams.get('no_pick_eept', False):
						two_head_in_shape = pick_eept_in_shape
						if network_config.get('temporal_conv_2_head_ee', False):
							temporal_kernel_size = network_config.get('1d_kernel_size', 2)
							temporal_num_filters = network_config.get('1d_num_filters_pick_ee', [32, 32, 32])
							temporal_num_filters[-1] = len(pick_eept_range)
							for j in xrange(len(temporal_num_filters)):
								if j != len(temporal_num_filters) - 1:
									weights['w_1d_conv_2_head_pick_ee_%d' % j] = init_weights([temporal_kernel_size, two_head_in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_pick_ee_%d' % j)
									weights['b_1d_conv_2_head_pick_ee_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_pick_ee_%d' % j)
									two_head_in_shape = temporal_num_filters[j]
								else:
									weights['w_1d_conv_2_head_pick_ee_%d' % j] = init_weights([1, two_head_in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_pick_ee_%d' % j)
									weights['b_1d_conv_2_head_pick_ee_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_pick_ee_%d' % j)
						else:
							n_layers_pick_ee_2_head = self._hyperparams.get('n_layers_pick_ee_2_head', 1)
							layer_size_pick_ee_2_head = [self._hyperparams.get('layer_size_pick_ee', 40)]*(n_layers_pick_ee_2_head-1)
							layer_size_pick_ee_2_head.append(len(pick_eept_range))
							for j in xrange(n_layers_pick_ee_2_head):
								weights['w_pick_ee_two_heads_%d' % j] = init_weights([two_head_in_shape, layer_size_pick_ee_2_head[j]], name='w_pick_ee_two_heads_%d' % j)
								weights['b_pick_ee_two_heads_%d' % j] = init_bias([layer_size_pick_ee_2_head[j]], name='b_pick_ee_two_heads_%d' % j)
								two_head_in_shape = layer_size_pick_ee_2_head[j]
					pick_eept_in_shape = layer_size_pick_ee[i]
				in_shape += (len(pick_eept_range))
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
		state_in_shape = len(self.x_idx)
		rem = 0
		if self._hyperparams.get('stop_signal', False):
			rem += 1
		if self._hyperparams.get('gripper_command_signal', False):
			rem += 1
		for i in xrange(n_layers):
			if i == 0:
				if 'num_concat_last_states' in self._hyperparams:
					in_shape += self._hyperparams['num_concat_last_states']*dim_hidden[i]
					state_in_shape += self._hyperparams['num_concat_last_states']*dim_hidden[i]
					self.conv_out_size_final = in_shape
				if self._hyperparams.get('sep_state', False):
					if self.norm_type == 'selu':
						weights['w_%d_img' % i] = init_fc_weights_snn([in_shape-len(self.x_idx), dim_hidden[i]], name='w_%d_img' % i)
						weights['w_%d_state' % i] = init_fc_weights_snn([state_in_shape, dim_hidden[i]], name='w_%d_state' % i)
					else:
						weights['w_%d_img' % i] = init_weights([in_shape-len(self.x_idx), dim_hidden[i]], name='w_%d_img' % i)
						weights['w_%d_state' % i] = init_weights([state_in_shape, dim_hidden[i]], name='w_%d_state' % i)
						if self._hyperparams.get('two_arms', False):
							weights['b_%d_state_two_arms' % i] = init_bias([dim_hidden[i]], name='b_%d_state_two_arms' % i)
							if self._hyperparams.get('free_state', False):
								weights['w_%d_state_two_arms' % i] = init_weights([state_in_shape, dim_hidden[i]], name='w_%d_state_two_arms' % i)
					weights['b_%d_img' % i] = init_bias([dim_hidden[i]], name='b_%d_img' % i)
					weights['b_%d_state' % i] = init_bias([dim_hidden[i]], name='b_%d_state' % i)
					in_shape = dim_hidden[i]
					continue
			if i > 0 and self._hyperparams.get('bt_all_fc', False) and not (i == n_layers-1 and network_config.get('predict_feature', False)):
				in_shape += self._hyperparams.get('context_dim', 10)
				weights['context_%d' % i] = init_bias([self._hyperparams.get('context_dim', 10)], name='context_%d' % i)
			if self.norm_type == 'selu':
				weights['w_%d' % i] = init_fc_weights_snn([in_shape, dim_hidden[i]], name='w_%d' % i)
			else:
				if i == n_layers - 1:
					if self._hyperparams.get('mixture_density', False):
						num_mixtures = self._hyperparams.get('num_mixtures', 20)
						dim_out_ = dim_hidden[i]
						if self._hyperparams.get('stop_signal', False):
							dim_out_ -= 1
						if self._hyperparams.get('gripper_command_signal', False):
							dim_out_ -= 1
						weights['w_%d' % i] = init_weights([in_shape, (dim_out_ + 2)*num_mixtures + rem], name='w_%d' % i)
					elif self._hyperparams.get('use_discretization', False):
						weights['w_%d' % i] = init_weights([in_shape, self.n_actions*self.n_bins + rem], name='w_%d' % i)
				else:
					weights['w_%d' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d' % i)
				# weights['w_%d' % i] = init_fc_weights_xavier([in_shape, dim_hidden[i]], name='w_%d' % i)
			if i == n_layers - 1:
				if self._hyperparams.get('mixture_density', False):
					num_mixtures = self._hyperparams.get('num_mixtures', 20)
					weights['b_%d' % i] = init_weights([(dim_out_ + 2)*num_mixtures + rem], name='b_%d' % i)
					self.mixture_dim = dim_out_ + 2
				elif self._hyperparams.get('use_discretization', False):
					weights['b_%d' % i] = init_weights([self.n_actions*self.n_bins + rem], name='b_%d' % i)
			else:
				weights['b_%d' % i] = init_bias([dim_hidden[i]], name='b_%d' % i)
			if (i == n_layers - 1 or (i == 0 and self._hyperparams.get('zero_state', False) and not self._hyperparams.get('sep_state', False))) and \
				self._hyperparams.get('two_heads', False):
				if i == 0:
					weights['w_%d_two_heads' % i] = init_weights([in_shape, dim_hidden[i]], name='w_%d_two_heads' % i)
					weights['b_%d_two_heads' % i] = init_bias([dim_hidden[i]], name='b_%d_two_heads' % i)
				else:
					if self._hyperparams.get('learn_action_layer', False):
						# TODO: figure out how to do this with mdn
						if self._hyperparams.get('mixture_density', False):
							num_mixtures = self._hyperparams.get('num_mixtures', 20)
							out_dim = num_mixtures * self.mixture_dim + rem
						if self._hyperparams.get('use_discretization', False):
							out_dim = self.n_bins * self.n_actions + rem
						else:
							out_dim = dim_output
						in_shape += out_dim
					if network_config.get('temporal_conv_2_head', False):
						initialization = self._hyperparams.get('initialization', 'xavier')
						if initialization == 'xavier':
							conv1d_init_weight = init_conv_weights_xavier
						else:
							conv1d_init_weight = init_weights
						temporal_kernel_size = network_config.get('1d_kernel_size', 2)
						temporal_num_filters = network_config.get('1d_num_filters', [32, 32, 32])
						temporal_num_filters[-1] = dim_output
						num_learned_loss = self._hyperparams.get('num_learned_loss', 1)
						if num_learned_loss > 1:
							for loss_idx in xrange(num_learned_loss):
								in_shape_tmp = in_shape
								for j in xrange(len(temporal_num_filters)):
									if j != len(temporal_num_filters) - 1:
										weights['w_1d_conv_2_head_loss%d_%d' % (loss_idx, j)] = init_weights([temporal_kernel_size, in_shape_tmp, temporal_num_filters[j]], name='w_1d_conv_2_head_loss%d_%d' % (loss_idx, j))
										weights['b_1d_conv_2_head_loss%d_%d' % (loss_idx, j)] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_loss%d_%d' % (loss_idx, j))
										in_shape_tmp = temporal_num_filters[j]
									else:
										weights['w_1d_conv_2_head_loss%d_%d' % (loss_idx, j)] = init_weights([1, in_shape_tmp, temporal_num_filters[j]], name='w_1d_conv_2_head_loss%d_%d' % (loss_idx, j))
										weights['b_1d_conv_2_head_loss%d_%d' % (loss_idx, j)] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_loss%d_%d' % (loss_idx, j))
						else:
							for j in xrange(len(temporal_num_filters)):
								if j != len(temporal_num_filters) - 1:
									weights['w_1d_conv_2_head_%d' % j] = init_weights([temporal_kernel_size, in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_%d' % j)
									weights['b_1d_conv_2_head_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_%d' % j)
									in_shape = temporal_num_filters[j]
								else:
									weights['w_1d_conv_2_head_%d' % j] = init_weights([1, in_shape, temporal_num_filters[j]], name='w_1d_conv_2_head_%d' % j)
									weights['b_1d_conv_2_head_%d' % j] = init_bias([temporal_num_filters[j]], name='b_1d_conv_2_head_%d' % j)
					else:                    
						n_layers_2_head = network_config.get('n_layers_2_head', 1)
						layer_size_2_head = network_config.get('layer_size_2_head', 40)
						dim_hidden_2_head = (n_layers_2_head-1)*[layer_size_2_head]
						if network_config.get('predict_feature', False):
							dim_hidden_2_head.append(in_shape)
						else:
							dim_hidden_2_head.append(dim_output) # single output for loss
						if self._hyperparams.get('flatten_2_head', False):
							in_shape *= self.T
						for j in xrange(n_layers_2_head):
							weights['w_%d_two_heads' % (i+j)] = init_weights([in_shape, dim_hidden_2_head[j]], name='w_%d_two_heads' % (i+j))
							weights['b_%d_two_heads' % (i+j)] = init_bias([dim_hidden_2_head[j]], name='b_%d_two_heads' % (i+j))
							in_shape = dim_hidden_2_head[j]
						if network_config.get('regress_actions', False) and network_config.get('predict_feature', False):
							weights['w_%d_two_heads_act' % (i+n_layers_2_head)] = init_weights([in_shape, dim_output], name='w_%d_two_heads_act' % (i+n_layers_2_head))
							weights['b_%d_two_heads_act' % (i+n_layers_2_head)] = init_bias([dim_output], name='b_%d_two_heads_act' % (i+n_layers_2_head))
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
		dim_input = num_units
		if self._hyperparams.get('zero_state', False):
			dim_input += len(self.x_idx) 
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

	def forward(self, image_input, state_input, weights, loss_idx=None, pick_labels=None, meta_testing=False, update=False, is_training=True, testing=False, network_config=None):
		# tile up context variable
		use_depth = self._hyperparams.get('use_depth', False)
		if self._hyperparams.get('use_vision', False) and use_depth:
			depth_input = tf.expand_dims(image_input[:, :, :, -1], axis=3)
			image_input = image_input[:, :, :, :-1]
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
			if self._hyperparams.get('learn_final_eept', False):
				context_final_eept = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(self._hyperparams.get('context_dim', 10))))
				context_final_eept += weights['context_final_eept']
			if self._hyperparams.get('pred_pick_eept', False):
				context_pick_eept = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), range(self._hyperparams.get('context_dim', 10))))
				context_pick_eept += weights['context_pick_eept']
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
				# if not testing:
				#     if not meta_testing:
				#         batch_size = self.update_batch_size*self.T
				#     else:
				#         batch_size = self.test_batch_size*self.T
				# else:
				#     batch_size = 1
				# img_context = tf.reshape(tf.tile(tf.reshape(weights['img_context'], [-1]), [batch_size]), [-1, im_height, im_width, num_channels])
				img_context = tf.zeros_like(conv_layer)
				img_context += weights['img_context']
				conv_layer = tf.concat(axis=3, values=[conv_layer, img_context])
				# if use_depth:
				#     depth_context = tf.zeros_like(depth_input)
				#     depth_context += weights['depth_context']
				#     depth_input = tf.concat(axis=3, values=[depth_input, depth_context])
			for i in xrange(n_conv_layers):
				if norm_type == 'vbn':
					if not use_dropout:
						conv_layer = self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
										name='vbn_%d' % (i+1), update=update)
					else:
						conv_layer = dropout(self.vbn(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
										name='vbn_%d' % (i+1), update=update), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
				else:
					conv_layer = conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated)
					if i == 0 and use_depth:
						depth_conv_out = conv2d(img=depth_input, w=weights['wc_depth_%d' % (i+1)], b=weights['bc_depth_%d' % (i+1)], strides=strides[i], is_dilated=is_dilated)
						conv_layer = tf.concat(axis=3, values=[conv_layer, depth_conv_out])
					if not use_dropout:
						conv_layer = norm(conv_layer, norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn)
					else:
						conv_layer = dropout(norm(conv_layer, norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))
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
		
				fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
				fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
		
				conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
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
				final_ee_inp = tf.reshape(conv_out_flat, [-1, T, self.conv_out_size])
				conv_size = self.conv_out_size
				if self._hyperparams.get('use_state_context', False):
					context_dim = self._hyperparams.get('context_dim', 10)
					final_ee_inp = tf.concat(axis=2, values=[final_ee_inp, tf.reshape(context_final_eept, [-1, T, context_dim])])
					conv_size += context_dim
				# only predict the final eept using the initial image
				# use video for preupdate only if no_final_eept
				if (not self._hyperparams.get('learn_final_eept_whole_traj', False)) or meta_testing:
					final_ee_inp = final_ee_inp[:, 0, :]
				else:
					final_ee_inp = tf.reshape(final_ee_inp, [-1, conv_size])
				n_layers_ee = network_config.get('n_layers_ee', 1)
				final_eept_pred = final_ee_inp
				for i in xrange(n_layers_ee):
					if i != n_layers_ee - 1:
						final_eept_pred = tf.nn.relu(tf.matmul(final_eept_pred, weights['w_ee_%d' % i]) + weights['b_ee_%d' % i])
					else:
						if self._hyperparams.get('two_heads', False) and not meta_testing and self._hyperparams.get('no_final_eept', False):
							if network_config.get('temporal_conv_2_head_ee', False):
								final_eept_pred = tf.reshape(final_eept_pred, [-1, self.T, final_eept_pred.get_shape().dims[-1].value])
								task_label_pred = None
								temporal_num_filters = network_config.get('1d_num_filters_ee', [32, 32, 32])
								for j in xrange(len(temporal_num_filters)):
									if j != len(temporal_num_filters) - 1:
										final_eept_pred = norm(conv1d(img=final_eept_pred, w=weights['w_1d_conv_2_head_ee_%d' % j], b=weights['b_1d_conv_2_head_ee_%d' % j]), \
														norm_type=self.norm_type, id=n_conv_layers+j, is_training=is_training, activation_fn=self.activation_fn)
									else:
										if 'num_learned_loss' in self._hyperparams and loss_idx is None:
											task_label_pred = conv1d(img=final_eept_pred, w=weights['w_1d_conv_2_head_tasklabel_%d' % j], b=weights['b_1d_conv_2_head_tasklabel_%d' % j])
										final_eept_pred = conv1d(img=final_eept_pred, w=weights['w_1d_conv_2_head_ee_%d' % j], b=weights['b_1d_conv_2_head_ee_%d' % j])
								final_eept_pred = tf.reshape(final_eept_pred, [-1, len(final_eept_range)])
							else:
								n_layers_ee_2_head = self._hyperparams.get('n_layers_ee_2_head', 1)
								for j in xrange(n_layers_ee_2_head):
									final_eept_pred = tf.matmul(final_eept_pred, weights['w_ee_two_heads_%d' % j]) + weights['b_ee_two_heads_%d' % j]
									if j != n_layers_ee_2_head - 1:
										final_eept_pred = tf.nn.relu(final_eept_pred)
						else:
							final_eept_pred = tf.matmul(final_eept_pred, weights['w_ee_%d' % i]) + weights['b_ee_%d' % i]
				if (not self._hyperparams.get('learn_final_eept_whole_traj', False)) or meta_testing:
					# this seems to only work when ubs=tbs=1
					# final_eept_pred = tf.reshape(tf.tile(tf.reshape(final_eept_pred, [-1]), [T]), [-1, len(final_eept_range)])
					final_eept_pred = tf.reshape(tf.tile(tf.expand_dims(final_eept_pred, axis=1), [1, T, 1]), [-1, len(final_eept_range)])
					final_eept_concat = tf.identity(final_eept_pred)
				else:
					# Assume tbs == 1
					# Only provide the FC layers with final_eept_pred at first time step
					final_eept_concat = tf.reshape(final_eept_pred, [-1, T, len(final_eept_range)])[:, 0, :]
					final_eept_concat = tf.reshape(tf.tile(tf.expand_dims(final_eept_concat, axis=1), [1, T, 1]), [-1, len(final_eept_range)])
					if 'num_learned_loss' in self._hyperparams and loss_idx is None:
						task_label_pred = tf.reshape(task_label_pred, [-1, self._hyperparams['num_learned_loss']])[0, :]
					# final_eept_concat = tf.reshape(tf.tile(tf.reshape(final_eept_concat, [-1]), [T]), [-1, len(final_eept_range)])
				if self._hyperparams.get('stop_grad_ee', False):
					final_eept_concat = tf.stop_gradient(final_eept_concat)
				fc_input = tf.concat(axis=1, values=[fc_input, final_eept_concat])
				# fc_input = tf.concat(concat_dim=1, values=[fc_input, final_eept_pred])
			else:
				final_eept_pred = None
			if self._hyperparams.get('pred_pick_eept', False):
				pick_eept_range = self._hyperparams['pick_eept_range']
				if testing:
					T = 1
				else:
					T = self.T
				pick_ee_inp = tf.reshape(conv_out_flat, [-1, T, self.conv_out_size])
				conv_size = self.conv_out_size
				if self._hyperparams.get('use_state_context', False):
					context_dim = self._hyperparams.get('context_dim', 10)
					pick_ee_inp = tf.concat(axis=2, values=[pick_ee_inp, tf.reshape(context_pick_eept, [-1, T, context_dim])])
					conv_size += context_dim
				# only predict the final eept using the initial image
				# use video for preupdate only if no_final_eept
				if meta_testing:
					pick_ee_inp = pick_ee_inp[:, 0, :]
				else:
					pick_ee_inp = tf.reshape(pick_ee_inp, [-1, conv_size])
				n_layers_pick_ee = network_config.get('n_layers_pick_ee', 1)
				pick_eept_pred = pick_ee_inp
				for i in xrange(n_layers_pick_ee):
					if i != n_layers_pick_ee - 1:
						pick_eept_pred = tf.nn.relu(tf.matmul(pick_eept_pred, weights['w_ee_%d' % i]) + weights['b_ee_%d' % i])
					else:
						if self._hyperparams.get('two_heads', False) and not meta_testing and self._hyperparams.get('no_final_eept', False):
							if network_config.get('temporal_conv_2_head_ee', False):
								pick_eept_pred = tf.reshape(pick_eept_pred, [-1, self.T, pick_eept_pred.get_shape().dims[-1].value])
								temporal_num_filters = network_config.get('1d_num_filters_pick_ee', [32, 32, 32])
								n_conv_layers += len(network_config.get('1d_num_filters_ee', [32, 32, 32]))
								for j in xrange(len(temporal_num_filters)):
									if j != len(temporal_num_filters) - 1:
										pick_eept_pred = norm(conv1d(img=pick_eept_pred, w=weights['w_1d_conv_2_head_pick_ee_%d' % j], b=weights['b_1d_conv_2_head_pick_ee_%d' % j]), \
														norm_type=self.norm_type, id=n_conv_layers+j, is_training=is_training, activation_fn=self.activation_fn)
									else:
										pick_eept_pred = conv1d(img=pick_eept_pred, w=weights['w_1d_conv_2_head_pick_ee_%d' % j], b=weights['b_1d_conv_2_head_pick_ee_%d' % j])
								pick_eept_pred = tf.reshape(pick_eept_pred, [-1, len(pick_eept_range)])
							else:
								n_layers_pick_ee_2_head = self._hyperparams.get('n_layers_pick_ee_2_head', 1)
								for j in xrange(n_layers_pick_ee_2_head):
									pick_eept_pred = tf.matmul(pick_eept_pred, weights['w_pick_ee_two_heads_%d' % j]) + weights['b_pick_ee_two_heads_%d' % j]
									if j != n_layers_pick_ee_2_head - 1:
										pick_eept_pred = tf.nn.relu(pick_eept_pred)
						else:
							pick_eept_pred = tf.matmul(pick_eept_pred, weights['w_pick_ee_%d' % i]) + weights['b_pick_ee_%d' % i]
				if meta_testing:
					# this seems to only work when ubs=tbs=1
					# final_eept_pred = tf.reshape(tf.tile(tf.reshape(pick_eept_pred, [-1]), [T]), [-1, len(final_eept_range)])
					pick_eept_pred = tf.reshape(tf.tile(tf.expand_dims(pick_eept_pred, axis=1), [1, T, 1]), [-1, len(pick_eept_range)])
					pick_eept_concat = tf.identity(pick_eept_pred)
				else:
					# Assume tbs == 1
					# Only provide the FC layers with pick_eept_pred at first time step
					pick_eept_concat = tf.reshape(pick_eept_pred, [-1, T, len(pick_eept_range)])[:, 0, :]
					pick_eept_concat = tf.reshape(tf.tile(tf.expand_dims(pick_eept_concat, axis=1), [1, T, 1]), [-1, len(pick_eept_range)])
					# final_eept_concat = tf.reshape(tf.tile(tf.reshape(final_eept_concat, [-1]), [T]), [-1, len(final_eept_range)])
				pick_eept_concat = pick_labels * pick_eept_concat
				if self._hyperparams.get('stop_grad_ee', False):
					pick_eept_concat = tf.stop_gradient(pick_eept_concat)
				else:
					pick_eept_concat = tf.cond(tf.reduce_all(tf.equal(pick_labels, tf.ones_like(pick_labels))), lambda: pick_eept_concat, lambda: tf.stop_gradient(pick_eept_concat))
				fc_input = tf.concat(axis=1, values=[fc_input, pick_eept_concat])
				# fc_input = tf.concat(concat_dim=1, values=[fc_input, pick_eept_pred])
			else:
				pick_eept_pred = None
		else:
			fc_input = image_input
			final_eept_pred, pick_eept_pred = None, None
		if self._hyperparams.get('use_state_context', False):
			fc_input = tf.concat(axis=1, values=[fc_input, context])
		if 'num_learned_loss' in self._hyperparams and not meta_testing:
			num_learned_loss = self._hyperparams['num_learned_loss']
			if loss_idx is None:
				task_label_prob = tf.nn.softmax(task_label_pred)
				loss_idx_out = tf.one_hot(indices=tf.argmax(task_label_prob), depth=num_learned_loss)
			else:
				loss_idx_out = loss_idx
		else:
			loss_idx_out = None
		if self._hyperparams.get('pred_pick_eept', False):
			if 'num_learned_loss' in self._hyperparams and not meta_testing:
				if loss_idx is None:
					aux_out = (final_eept_pred, pick_eept_pred, task_label_pred, loss_idx_out)
				else:
					aux_out = (final_eept_pred, pick_eept_pred, loss_idx_out)
			else:
				aux_out = (final_eept_pred, pick_eept_pred)
		else:
			aux_out = final_eept_pred
		if self._hyperparams.get('use_rnn', False):
			return self.rnn_forward(fc_input, weights, state_input=state_input, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), aux_out
		if self._hyperparams.get('use_lstm', False):
			return self.lstm_forward(fc_input, weights, state_input=state_input, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), aux_out
		return self.fc_forward(fc_input, weights, loss_idx=loss_idx_out, state_input=state_input, meta_testing=meta_testing, is_training=is_training, testing=testing, network_config=network_config), aux_out

	def fc_forward(self, fc_input, weights, loss_idx=None, state_input=None, meta_testing=False, is_training=True, testing=False, network_config=None):
		n_layers = network_config.get('n_layers', 4) # 3
		use_dropout = self._hyperparams.get('use_dropout', False)
		prob = self._hyperparams.get('keep_prob', 0.5)
		fc_output = tf.add(fc_input, 0)
		use_selu = self.norm_type == 'selu'
		use_ln = self._hyperparams.get('ln_for_fc', False)
		norm_type = self.norm_type
		dim_hidden = network_config.get('layer_size', [100]*(n_layers-1))
		if type(dim_hidden) is not list:
			dim_hidden = (n_layers - 1)*[dim_hidden]
		if self._hyperparams.get('use_vision', False) and state_input is not None:
			if not self._hyperparams.get('sep_state', False):
				fc_output = tf.concat(axis=1, values=[fc_output, state_input])
			elif self._hyperparams.get('use_context', False) and meta_testing:
				context_dim = self._hyperparams.get('context_dim', 10)
				context = state_input[:, :context_dim]
				state_input = state_input[:, context_dim:]
				fc_output = tf.concat(axis=1, values=[fc_output, context])
		for i in xrange(n_layers):
			if i > 0 and self._hyperparams.get('bt_all_fc', False) and not (i == n_layers-1 and network_config.get('predict_feature', False)):
				context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(fc_output)), range(self._hyperparams.get('context_dim', 10))))
				context += weights['context_%d' % i]
				fc_output = tf.concat(axis=1, values=[fc_output, context])
			if (i == n_layers - 1 or (i == 0 and self._hyperparams.get('zero_state', False) and not self._hyperparams.get('sep_state', False))) and \
				self._hyperparams.get('two_heads', False) and not meta_testing:
				if i == 0:
					if 'num_concat_last_states' in self._hyperparams:
						num_actions = self._hyperparams['num_concat_last_states']
						# assume tbs=1
						init_actions = [tf.zeros((1, dim_hidden[i])) for _ in xrange(num_actions)]
						fc_output = self.rnn_actions_forward(fc_output, init_actions, weights['w_%d_two_heads' % i],
															weights['b_%d_two_heads' % i], in_shape=self.conv_out_size_final,
															dim_out=dim_hidden[i], T=self.T, num_actions=num_actions)
					else:
						fc_output = tf.matmul(fc_output, weights['w_%d_two_heads' % i]) + weights['b_%d_two_heads' % i]
				else:
					if self._hyperparams.get('learn_action_layer', False):
						# TODO: figure out how to do this with mdn
						if self._hyperparams.get('mixture_density', False) and meta_testing: # just for postupdate for now
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
							mixture_params[:, -2, :] = tf.exp(mixture_params[:, -2, :])
							fc_output_action = tf.reshape(mixture_params, [-1, self.mixture_dim*num_mixtures])
							if self._hyperparams.get('gripper_command_signal', False):
								fc_output_action = tf.concat(axis=1, values=[fc_output_action, gripper_command])
							if self._hyperparams.get('stop_signal', False):
								fc_output_action = tf.concat(axis=1, values=[fc_output_action, stop_signal])
						else:
							fc_output_action = tf.matmul(fc_output, weights['w_%d' % i]) + weights['b_%d' % i]
						fc_output = tf.concat(axis=1, values=[fc_output, fc_output_action])
					if network_config.get('temporal_conv_2_head', False):
						fc_output = tf.reshape(fc_output, [-1, self.T, fc_output.get_shape().dims[-1].value])
						strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
						temporal_num_filters = network_config.get('1d_num_filters', [32, 32, 32])
						n_conv_layers = len(strides)
						if network_config.get('temporal_conv_2_head_ee', False):
							n_conv_layers += len(network_config.get('1d_num_filters_ee', [32, 32, 32]))
							if self._hyperparams.get('pred_pick_eept', False):
								n_conv_layers += len(network_config.get('1d_num_filters_pick_ee', [32, 32, 32]))
						if 'num_learned_loss' in self._hyperparams:
							# def learned_loss(inp, idx=0):
							#     for j in xrange(len(temporal_num_filters)):
							#         if j != len(temporal_num_filters) - 1:
							#             inp = norm(conv1d(img=inp, w=weights['w_1d_conv_2_head_loss%d_%d' % (idx, j)], b=weights['b_1d_conv_2_head_loss%d_%d' % (idx, j)]), \
							#                             norm_type=self.norm_type, id=j, is_training=is_training, activation_fn=self.activation_fn, prefix='learned_loss%d_' % idx)
							#         else:
							#             inp = conv1d(img=inp, w=weights['w_1d_conv_2_head_loss%d_%d' % (idx, j)], b=weights['b_1d_conv_2_head_loss%d_%d' % (idx, j)])
							#     return inp
							# fc_output = tf.case({tf.equal(loss_idx, tf.constant(0, dtype=tf.int64)): lambda:learned_loss(fc_output, idx=0), 
							#                     tf.equal(loss_idx, tf.constant(1, dtype=tf.int64)): lambda:learned_loss(fc_output, idx=1),
							#                     tf.equal(loss_idx, tf.constant(2, dtype=tf.int64)): lambda:learned_loss(fc_output, idx=2)},
							#                     exclusive=True, default=lambda:learned_loss(fc_output, idx=0))
							assert loss_idx is not None
							learned_losses = []
							for idx in xrange(self._hyperparams['num_learned_loss']):
								inp = tf.identity(fc_output)
								for j in xrange(len(temporal_num_filters)):
									if j != len(temporal_num_filters) - 1:
										inp = norm(conv1d(img=inp, w=weights['w_1d_conv_2_head_loss%d_%d' % (idx, j)], b=weights['b_1d_conv_2_head_loss%d_%d' % (idx, j)]), \
														norm_type=self.norm_type, id=j, is_training=is_training, activation_fn=self.activation_fn, prefix='learned_loss%d_' % idx)
									else:
										inp = conv1d(img=inp, w=weights['w_1d_conv_2_head_loss%d_%d' % (idx, j)], b=weights['b_1d_conv_2_head_loss%d_%d' % (idx, j)])
								learned_loss = euclidean_loss_layer(inp, 0.0, None, multiplier=self._hyperparams.get('loss_multiplier', 100.0), behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
								learned_losses.append(learned_loss)
							fc_output = tf.reduce_sum(tf.stack(learned_losses)*loss_idx)
							# fc_output = tf.stack(learned_losses)[tf.argmax(loss_idx)]
						else:
							for j in xrange(len(temporal_num_filters)):
								if j != len(temporal_num_filters) - 1:
									fc_output = norm(conv1d(img=fc_output, w=weights['w_1d_conv_2_head_%d' % j], b=weights['b_1d_conv_2_head_%d' % j]), \
													norm_type=self.norm_type, id=n_conv_layers+j, is_training=is_training, activation_fn=self.activation_fn)
								else:
									fc_output = conv1d(img=fc_output, w=weights['w_1d_conv_2_head_%d' % j], b=weights['b_1d_conv_2_head_%d' % j])
					else:
						n_layers_2_head = network_config.get('n_layers_2_head', 1)
						feature_dim = fc_output.get_shape().dims[-1].value
						if self._hyperparams.get('flatten_2_head', False):
							fc_output = tf.reshape(fc_output, [-1, self.T*feature_dim])
						if network_config.get('predict_feature', False):
							fc_output = tf.reshape(fc_output, [-1, self.T, feature_dim])
							# next_features = tf.reshape(fc_output[:, 1:, :], [-1, feature_dim])
							next_features = fc_output[:, -1, :]
							fc_output = tf.reshape(fc_output[:, :-1, :], [-1, feature_dim])
						for j in xrange(n_layers_2_head):
							fc_output = tf.matmul(fc_output, weights['w_%d_two_heads' % (i+j)]) + weights['b_%d_two_heads' % (i+j)]
							if j != n_layers_2_head - 1:
								fc_output = tf.nn.relu(fc_output)
							else:
								if network_config.get('predict_feature', False):
									fc_output = fc_output - next_features
									if network_config.get('regress_actions', False):
										fc_output = tf.matmul(fc_output, weights['w_%d_two_heads_act' % (i+n_layers_2_head)]) + \
																weights['b_%d_two_heads_act' % (i+n_layers_2_head)]
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
					state_part = tf.constant(0.0)
				if 'num_concat_last_states' in self._hyperparams:
					num_actions = self._hyperparams['num_concat_last_states']
					if meta_testing:
						init_state_actions = [tf.zeros((1, dim_hidden[i])) for _ in xrange(num_actions)]
						state_part = self.rnn_actions_forward(state_input, init_state_actions, weights['w_%d_state' % i], 
															weights['b_%d_state' % i], in_shape=len(self.x_idx),
															dim_out=dim_hidden[i], T=self.T, num_actions=num_actions)
					init_actions = [tf.zeros((1, dim_hidden[i])) for _ in xrange(num_actions)]
					fc_output = self.rnn_actions_forward(fc_output, init_actions, weights['w_%d_img' % i], 
														weights['b_%d_img' % i], in_shape=self.conv_out_size_final-len(self.x_idx),
														dim_out=dim_hidden[i], T=self.T, num_actions=num_actions)
				else:
					if meta_testing:
						state_part = tf.matmul(state_input, weights['w_%d_state' % i]) + weights['b_%d_state' % i]
					fc_output = tf.matmul(fc_output, weights['w_%d_img' % i]) + weights['b_%d_img' % i]
				fc_output = fc_output + state_part
			else:
				if i == 0 and 'num_concat_last_states' in self._hyperparams:
					num_actions = self._hyperparams['num_concat_last_states']
					init_actions = [tf.zeros((1, dim_hidden[i])) for _ in xrange(num_actions)]
					fc_output = self.rnn_actions_forward(fc_output, init_actions, weights['w_%d' % i], 
														weights['b_%d' % i], in_shape=self.conv_out_size_final,
														dim_out=dim_hidden[i], T=self.T, num_actions=num_actions)
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
			if i == n_layers - 1:
				if self._hyperparams.get('non_linearity_out', False):
					fc_output = self.activation_fn(fc_output)
				elif self._hyperparams.get('mixture_density', False) and meta_testing: # just for postupdate for now
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
		# return fc_output, fp, moving_mean, moving_variance
		return fc_output
	
	def learn_loss(self, act_hat, act_tgt, weights, network_config=False):
		n_loss_layers = network_config.get('n_loss_layers', 2)
		loss_multiplier = self._hyperparams.get('loss_multiplier', 100.0)
		loss = tf.concat(axis=1, values=[act_hat, act_tgt])
		for j in xrange(n_loss_layers):
			loss = tf.matmul(loss, weights['w_loss_%d' % j]) + weights['b_loss_%d' % j]
			if j != n_loss_layers - 1:
				loss = self.activation_fn(loss)
		# return (loss_multiplier*loss)**2
		return loss**2
		
	def rnn_forward(self, rnn_input, weights, state_input=None, meta_testing=False, is_training=True, testing=False, network_config=None):
		if state_input is not None and self._hyperparams.get('use_vision', False) and not self._hyperparams.get('zero_state', False):
			rnn_input = tf.concat(axis=1, values=[rnn_input, state_input])
		# LSTM forward
		state = weights['rnn_state']
		rnn_input = tf.reshape(rnn_input, [-1, self.T, self.conv_out_size_final])
		rnn_outputs = []
		for t in xrange(self.T):
			inp = tf.concat(axis=1, values=[rnn_input[:, t, :], state])
			rnn_output = tf.matmul(inp, weights['rnn_weight']) + weights['rnn_bias']
			if self._hyperparams.get('ln_for_rnn', False):
				rnn_output = norm(rnn_output, norm_type='layer_norm', id=t, is_training=is_training, prefix='rnn_')
			else:
				rnn_output = self.activation_fn(rnn_output)
			rnn_outputs.append(rnn_output)
			state = tf.identity(rnn_output)
		rnn_output = tf.concat(axis=0, values=rnn_outputs)
		return self.fc_forward(rnn_output, weights, is_training=is_training, testing=testing, network_config=network_config)
		
	def rnn_actions_forward(self, rnn_input, init_actions, weight, bias, in_shape, dim_out, T=40, num_actions=3):
		past_actions = deque(init_actions, maxlen=num_actions)
		rnn_input = tf.reshape(rnn_input, [-1, T, in_shape])
		outputs = []
		for t in xrange(T):
			inp = rnn_input[:, t, :]
			actions = tf.stop_gradient(tf.reshape(tf.stack(list(past_actions)), [-1, num_actions*dim_out]))
			inp = tf.concat(axis=1, values=[inp, actions])
			out = tf.nn.relu(tf.matmul(inp, weight) + bias)
			if t < num_actions:
				past_actions[t] = out
			else:
				past_actions.append(out)
			outputs.append(out)
		return tf.reshape(tf.transpose(tf.stack(outputs), perm=[1, 0, 2]), [-1, dim_out])
	
	def lstm_forward(self, lstm_input, weights, state_input=None, meta_testing=False, is_training=True, testing=False, network_config=None):
		if state_input is not None and self._hyperparams.get('use_vision', False) and not self._hyperparams.get('zero_state', False):
			lstm_input = tf.concat(axis=1, values=[lstm_input, state_input])
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
			cell_inputs = tf.concat(axis=1, values=[inp, m_prev])
			lstm_matrix = tf.matmul(cell_inputs, weights['lstm_weight']) + weights['lstm_bias']
			i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=lstm_matrix)
			if use_norm:
				i = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_i_')
				j = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_j_')
				f = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_f_')
				o = norm(i, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_o_')
			c = (tf.sigmoid(f + forget_bias) * c_prev + tf.sigmoid(i) * activation(j))
			if use_norm:
				c = norm(c, norm_type='layer_norm', id=t, is_training=is_training, activation_fn=None, prefix='lstm_c_')
			m = tf.sigmoid(o) * activation(c)
			state = tf.concat(axis=1, values=[c, m])
			lstm_outputs.append(m)
		lstm_output = tf.concat(axis=0, values=lstm_outputs)
		return self.fc_forward(lstm_output, weights, is_training=is_training, meta_testing=meta_testing, testing=testing, network_config=network_config)

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
				if self._hyperparams.get('use_depth', False):
					self.deptha = deptha = tf.placeholder(tf.float32, name='deptha')# meta_batch_size x update_batch_size x dim_input
					self.depthb = depthb = tf.placeholder(tf.float32, name='depthb')
			else:
				self.obsa = obsa = input_tensors['inputa'] # meta_batch_size x update_batch_size x dim_input
				self.obsb = obsb = input_tensors['inputb']
				if self._hyperparams.get('use_depth', False):
					self.deptha = deptha = input_tensors['deptha'] # meta_batch_size x update_batch_size x dim_input
					self.depthb = depthb = input_tensors['depthb']
		else:
			self.obsa, self.obsb = None, None
		# Temporary in order to make testing work
		if not hasattr(self, 'stateb'):
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
			if self._hyperparams.get('aggressive_light_aug', False) and 'Testing' not in prefix and not hasattr(self, 'lighting_tensor'):
				self.lighting_tensor = lighting = tf.placeholder(tf.float32, name='lighting')
			elif hasattr(self, 'lighting_tensor'):
				lighting = self.lighting_tensor
			inputa = tf.concat(axis=2, values=[statea, obsa])
			inputb = tf.concat(axis=2, values=[stateb, obsb])
			if self._hyperparams.get('use_depth', False):
				inputa = tf.concat(axis=2, values=[inputa, deptha])
				inputb = tf.concat(axis=2, values=[inputb, depthb])
		else:
			inputa = statea
			inputb = stateb
			
		if self._hyperparams.get('pred_pick_eept', False):
			if not hasattr(self, 'pick_labels'):
				self.pick_labels = pick_labels = tf.placeholder(tf.float32, name='pick_labels')
			else:
				pick_labels = self.pick_labels
			if 'num_learned_loss' in self._hyperparams:
				if not hasattr(self, 'task_labels'):
					self.task_labels = task_labels = tf.placeholder(tf.float32, name='task_labels')
				else:
					task_labels = self.task_labels
					
		if 'num_concat_last_states' in self._hyperparams and 'Testing' in prefix:
			self.curr_t = curr_t = tf.placeholder(tf.int64, name='curr_t')
		
		with tf.variable_scope('model', reuse=None) as training_scope:
			# Construct layers weight & bias
			# TODO: since we flip to reuse automatically, this code below is unnecessary
			if 'weights' not in dir(self):
				if 'final_eept_range' in self._hyperparams:
					final_eept_range = self._hyperparams['final_eept_range']
					dim_output_new = dim_output
					# if not self._hyperparams.get('sawyer', False):
					dim_output_new = dim_output-len(final_eept_range)
					if 'pick_eept_range' in self._hyperparams:
						pick_eept_range = self._hyperparams['pick_eept_range']
						dim_output_new -= len(pick_eept_range)
					# if 'num_learned_loss' in self._hyperparams:
					#     dim_output_new -= self._hyperparams['num_learned_loss']
					self.weights = weights = self.construct_weights(dim_input, dim_output_new, network_config=network_config)
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
			self.weight_reg = self._hyperparams.get('weight_reg', 0.0001)
			act_noise_std = self._hyperparams.get('act_noise_std', 0.5)
			loss_multiplier = self._hyperparams.get('loss_multiplier', 100.0)
			final_eept_loss_eps = self._hyperparams.get('final_eept_loss_eps', 0.01)
			pick_eept_loss_eps = self._hyperparams.get('pick_eept_loss_eps', 0.01)
			act_loss_eps = self._hyperparams.get('act_loss_eps', 1.0)
			hinge_loss_eps = self._hyperparams.get('hinge_loss_eps', 1.0)
			use_whole_traj = self._hyperparams.get('learn_final_eept_whole_traj', False)
			stop_signal_eps = self._hyperparams.get('stop_signal_eps', 1.0)
			gripper_command_signal_eps = self._hyperparams.get('gripper_command_signal_eps', 1.0)
			acosine_loss_eps = self._hyperparams.get('acosine_loss_eps', 0.5)
			task_label_loss_eps = self._hyperparams.get('task_label_loss_eps', 0.5)
			use_depth = self._hyperparams.get('use_depth', False)
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
				pick_labels, task_labels = None, None
				if hasattr(self, 'lighting_tensor'):
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							inputa, inputb, actiona, actionb, pick_labels, task_labels, lighting = inp
						else:
							inputa, inputb, actiona, actionb, pick_labels, lighting = inp
					else:
						inputa, inputb, actiona, actionb, lighting = inp
				else:
					lighting = None
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							inputa, inputb, actiona, actionb, pick_labels, task_labels = inp
						else:
							inputa, inputb, actiona, actionb, pick_labels = inp #image input
					else:
						inputa, inputb, actiona, actionb = inp
				inputa = tf.reshape(inputa, [-1, dim_input])
				inputb = tf.reshape(inputb, [-1, dim_input])
				actiona = tf.reshape(actiona, [-1, dim_output])
				actionb = tf.reshape(actionb, [-1, dim_output])
				# if pick_labels is not None:
				#     pick_labels = tf.squeeze(pick_labels)
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
				
				final_eepta, final_eeptb, pick_eepta, pick_eeptb = None, None, None, None

				# if 'num_learned_loss' in self._hyperparams:
				#     num_learned_loss = self._hyperparams['num_learned_loss']
				#     # TODO: This fix won't work during test time. Figure out a way soon.
				#     task_label = actionb[:, -num_learned_loss:]
				#     task_label = tf.reshape(task_label, [-1, num_learned_loss])[0, :]
				#     actiona = actiona[:, :-num_learned_loss]
				#     actionb = actionb[:, :-num_learned_loss]
				
				if 'pick_eept_range' in self._hyperparams:
					pick_eept_range = self._hyperparams['pick_eept_range']
					if self._hyperparams.get('pred_pick_eept', False):
						pick_eepta = actiona[:, pick_eept_range[0]:pick_eept_range[-1]+1]
						pick_eeptb = actionb[:, pick_eept_range[0]:pick_eept_range[-1]+1]
					actiona = actiona[:, :pick_eept_range[0]]
					actionb = actionb[:, :pick_eept_range[0]]
					if self._hyperparams.get('no_pick_eept', False):
						pick_eepta = tf.zeros_like(pick_eepta)
				
				if 'final_eept_range' in self._hyperparams:
					final_eept_range = self._hyperparams['final_eept_range']
					# assumes update_batch_size == 1
					# final_eepta = tf.reshape(tf.tile(actiona[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
					# final_eeptb = tf.reshape(tf.tile(actionb[-1, final_eept_range[0]:], [self.update_batch_size*self.T]), [-1, len(final_eept_range)])
					# if not self._hyperparams.get('sawyer', False):
					if self._hyperparams.get('learn_final_eept', False):
						final_eepta = actiona[:, final_eept_range[0]:final_eept_range[-1]+1]
						final_eeptb = actionb[:, final_eept_range[0]:final_eept_range[-1]+1]
					# import pdb; pdb.set_trace()
					actiona = actiona[:, :final_eept_range[0]]
					actionb = actionb[:, :final_eept_range[0]]
					# else:
					#     if self._hyperparams.get('learn_final_eept', False):
					#         inputa_ = tf.reshape(inputa, [-1, self.T, dim_input])
					#         if testing:
					#             Tb = self.T
					#         else:
					#             Tb = 1
					#         inputb_ = tf.reshape(inputb, [-1, Tb, dim_input])
					#         final_eepta = tf.reshape(tf.tile(tf.expand_dims(inputa_[:, -1, final_eept_range[0]:final_eept_range[-1]+1], axis=1), [1, self.T, 1]), [-1, len(final_eept_range)])
					#         final_eeptb = tf.reshape(tf.tile(tf.expand_dims(inputb_[:, -1, final_eept_range[0]:final_eept_range[-1]+1], axis=1), [1, Tb, 1]), [-1, len(final_eept_range)])

					if self._hyperparams.get('no_final_eept', False):
						final_eepta = tf.zeros_like(final_eepta)
				# else:
					# actiona = actiona[:, 6:]
					# actionb = actionb[:, 6:]
				if self._hyperparams.get('use_discretization', False):
					if self._hyperparams.get('stop_signal', False):
						if not self._hyperparams.get('no_action', False):
							stopa = tf.expand_dims(actiona[:, -1], axis=1)
							actiona = actiona[:, :-1]
						stopb = tf.expand_dims(actionb[:, -1], axis=1)
						actionb = actionb[:, :-1]
					if self._hyperparams.get('gripper_command_signal', False):
						if not self._hyperparams.get('no_action', False):
							grippera = tf.expand_dims(actiona[:, -1], axis=1)
							actiona = actiona[:, :-1]
						gripperb = tf.expand_dims(actionb[:, -1], axis=1)
						actionb = actionb[:, :-1]
					U_min = tf.convert_to_tensor(self.U_min, np.float32)
					avg_bin_action = tf.convert_to_tensor(self.avg_bin_action, np.float32)
					bin_size = tf.convert_to_tensor(self.bin_size, np.float32)
					if not self._hyperparams.get('no_action', False):
						bina = tf.floor_div((actiona - U_min), bin_size)
					binb = tf.floor_div((actionb - U_min), bin_size)
					if not self._hyperparams.get('no_action', False):
						actiona = tf.reshape(tf.one_hot(tf.cast(bina, tf.int32), depth=self.n_bins), [-1, self.n_actions*self.n_bins])
					actionb = tf.reshape(tf.one_hot(tf.cast(binb, tf.int32), depth=self.n_bins), [-1, self.n_actions*self.n_bins])
					if self._hyperparams.get('gripper_command_signal', False):
						if not self._hyperparams.get('no_action', False):
							actiona = tf.concat(axis=1, values=[actiona, grippera])
						actionb = tf.concat(axis=1, values=[actionb, gripperb])
					if self._hyperparams.get('stop_signal', False):
						if not self._hyperparams.get('no_action', False):
							actiona = tf.concat(axis=1, values=[actiona, stopa])
						actionb = tf.concat(axis=1, values=[actionb, stopb])
					
				if self._hyperparams.get('no_action', False):
					actiona = tf.zeros_like(actiona)

				if self._hyperparams.get('add_noise', False):
					actiona += tf.random_normal([self.T*self.update_batch_size, dim_output], stddev=act_noise_std)
				
				local_outputbs, local_lossesb, final_eept_lossesb, control_lossesb = [], [], [], []
				# Assume fixed data for each update
				actionas = [actiona]*num_updates
				
				# Convert to image dims
				if self._hyperparams.get('use_vision', True):
					inputa, flat_img_inputa, state_inputa = self.construct_image_input(inputa, x_idx, img_idx, use_depth=use_depth, light_augs=lighting, network_config=network_config)
					inputb, flat_img_inputb, state_inputb = self.construct_image_input(inputb, x_idx, img_idx, use_depth=use_depth, light_augs=lighting, network_config=network_config)
					inputas = [inputa]*num_updates
					inputbs = [inputb]*num_updates
					if self._hyperparams.get('zero_state', False):
						state_inputa = tf.zeros_like(state_inputa)
					state_inputas = [state_inputa]*num_updates
					if self._hyperparams.get('use_img_context', False):
						img_context = tf.zeros_like(inputa)
						img_context += self.img_context_var
						inputa = tf.concat(axis=3, values=[inputa, img_context])
					state_inputa_new = state_inputa
					if self._hyperparams.get('use_context', False):
						context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputa)), range(self._hyperparams.get('context_dim', 10))))
						context += self.context_var
						if self._hyperparams.get('no_state'):
							state_inputa_new = context
						else:
							state_inputa_new = tf.concat(axis=1, values=[context, state_inputa_new])
					elif self._hyperparams.get('no_state'):
						state_inputa_new = None
				else:
					inputas = [inputa]*num_updates
					if self._hyperparams.get('use_context', False):
						context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputa)), range(self._hyperparams.get('context_dim', 10))))
						context += self.context_var
						inputa = tf.concat(axis=1, values=[context, inputa])
					state_inputb = None
					state_inputa_new = None
					flat_img_inputa = tf.add(inputa, 0)
					flat_img_inputb = tf.add(inputb, 0) # pseudo-tensor
				
				if self._hyperparams.get('learn_final_eept', False):
					final_eeptas = [final_eepta]*num_updates
				if 'Training' in prefix:
					# local_outputa, fp, moving_mean, moving_variance = self.forward(inputa, state_inputa, weights, network_config=network_config)
					local_outputa, final_eept_preda = self.forward(inputa, state_inputa_new, weights, pick_labels=pick_labels, network_config=network_config)
				else:
					# local_outputa, _, moving_mean_test, moving_variance_test = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
					# TODO: modify this to choose which learned loss to use
					local_outputa, final_eept_preda = self.forward(inputa, state_inputa_new, weights, pick_labels=pick_labels, update=update, is_training=False, network_config=network_config)
				task_label_loss = tf.constant(0.0) # for learning multiple learned losses
				loss_idx = tf.constant(0.0) # for learning multiple learned losses
				if self._hyperparams.get('learn_final_eept', False):
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							final_eept_preda, pick_eept_preda, task_label_pred, loss_idx = final_eept_preda
							task_label_loss = tf.losses.softmax_cross_entropy(task_labels, task_label_pred)
						else:
							final_eept_preda, pick_eept_preda = final_eept_preda
						pick_eept_lossa = pick_labels * euclidean_loss_layer(pick_eept_preda, pick_eepta, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					else:
						pick_eept_lossa = tf.constant(0.0)
					final_eept_lossa = euclidean_loss_layer(final_eept_preda, final_eepta, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
				else:
					final_eept_lossa = tf.constant(0.0)
				if (self._hyperparams.get('flatten_2_head', False) or network_config.get('predict_feature', False)) and self._hyperparams.get('no_action', False):
					actiona = tf.zeros_like(local_outputa)
					actionas = [actiona]*num_updates
				if self._hyperparams.get('learn_loss', False):
					local_lossa = act_loss_eps * self.learn_loss(local_outputa, actiona, weights, network_config=network_config)
				elif 'num_learned_loss' in self._hyperparams:
					local_lossa = local_outputa
				else:
					if not self._hyperparams.get('no_action', False):
						discrete_loss = 0.0
						outputa_ = tf.identity(local_outputa)
						actiona_ = tf.identity(actiona)
						if self._hyperparams.get('stop_signal', False):
							stop_lossa = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actiona[:, -1], axis=1), logits=tf.expand_dims(local_outputa[:, -1], axis=1))
							discrete_loss += stop_signal_eps * tf.reduce_mean(stop_lossa)
							outputa_ = outputa_[:, :-1]
							actiona_ = actiona_[:, :-1]
						if self._hyperparams.get('gripper_command_signal', False):
							gripper_lossa = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actiona[:, -2], axis=1), logits=tf.expand_dims(local_outputa[:, -2], axis=1))
							discrete_loss += pick_labels * gripper_command_signal_eps * tf.reduce_mean(gripper_lossa)
							outputa_ = outputa_[:, :-1]
							actiona_ = actiona_[:, :-1]
							# still use mse for learning with human demos
							local_lossa = act_loss_eps * euclidean_loss_layer(outputa_, actiona_, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False)) + discrete_loss
					else:
						local_lossa = act_loss_eps * euclidean_loss_layer(local_outputa, actiona, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
				if self._hyperparams.get('learn_final_eept', False):
					local_lossa += final_eept_loss_eps * final_eept_lossa
					if self._hyperparams.get('pred_pick_eept', False):
						local_lossa += pick_eept_loss_eps * pick_eept_lossa
				acosine_lossa = 0.
				if self._hyperparams.get('use_acosine_loss', False) and not self._hyperparams.get('no_action', False):
					acosine_lossa = acosine_loss(outputa, actiona)
					local_lossa += acosine_loss_eps * acosine_lossa
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
					if self._hyperparams.get('clip_norm', False):
						clip_norm_max = self._hyperparams['clip_norm_max']
					for key in gradients.keys():
						# if 'context' not in key:
						if not self._hyperparams.get('clip_norm', False):
							gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
						else:
							gradients[key] = tf.clip_by_norm(gradients[key], clip_norm_max)
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
						inputb = tf.concat(axis=3, values=[inputb, img_contextb])
					state_inputb_new = state_inputb
					if self._hyperparams.get('use_context', False):
						contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputb)), range(self._hyperparams.get('context_dim', 10))))
						contextb += fast_context
						if self._hyperparams.get('no_state'):
							state_inputb_new = contextb
						else:
							state_inputb_new = tf.concat(axis=1, values=[contextb, state_inputb_new])
					elif self._hyperparams.get('no_state'):
						state_inputb_new = None
				else:
					if self._hyperparams.get('use_context', False):
						contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputb)), range(self._hyperparams.get('context_dim', 10))))
						contextb += fast_context
						inputb = tf.concat(axis=1, values=[contextb, inputb])
					state_inputb_new = None
				# Is mask used here?
				if self.step_size == 0.:
					fast_weights = weights
				elif update_rule == 'adam':
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
					outputb, final_eept_predb = self.forward(inputb, state_inputb_new, fast_weights, pick_labels=pick_labels, meta_testing=True, network_config=network_config)
					if self._hyperparams.get('learn_loss_reg', False):
						outputb_2_head, _ = self.forward(inputb, state_inputb_new, fast_weights, network_config=network_config)
						outputa_2_head, _ = self.forward(inputa, state_inputa_new, fast_weights, network_config=network_config)
				else:
					outputb, final_eept_predb = self.forward(inputb, state_inputb_new, fast_weights, pick_labels=pick_labels, meta_testing=True, update=update, is_training=False, testing=testing, network_config=network_config)
					if self._hyperparams.get('learn_loss_reg', False):
						outputb_2_head, _ = self.forward(inputb, state_inputb_new, fast_weights, update=update, is_training=False, testing=testing, network_config=network_config)
						outputa_2_head, _ = self.forward(inputa, state_inputa_new, fast_weights, update=update, is_training=False, testing=testing, network_config=network_config)
				# fast_weights_reg = tf.reduce_sum([self.weight_decay*tf.nn.l2_loss(var) for var in fast_weights.values()]) / tf.to_float(self.T)
				if self._hyperparams.get('mixture_density', False):
					outputb, gripper_command, stop_signal = outputb
					if num_updates == 1 and testing:
						samples = tf.stack([outputb.sample() for _ in range(20)])
						sample_probs = tf.squeeze(output.prob(samples))
						output_sample = samples[tf.argmax(sample_probs)]
					else:
						output_sample = outputb.sample()
					if gripper_command is not None:
						gripper_command_out = tf.cast(tf.sigmoid(gripper_command)>0.5, tf.float32)
						output_sample = tf.concat([output_sample, gripper_command_out], axis=1)
					if stop_signal is not None:
						stop_signal_out = tf.cast(tf.sigmoid(stop_signal)>0.5, tf.float32)
						output_sample = tf.concat([output_sample, stop_signal_out], axis=1)
					local_outputbs.append(output_sample)
				else:
					outputb_sample = tf.identity(outputb)
					if self._hyperparams.get('stop_signal', False):
						stop_signal = tf.expand_dims(outputb_sample[:, -1], axis=1)
						stop_signal_out = tf.cast(tf.sigmoid(stop_signal)>0.5, tf.float32)
					if self._hyperparams.get('gripper_command_signal', False):
						gripper_command = tf.expand_dims(outputb_sample[:, -1], axis=1)
						gripper_command_out = tf.cast(tf.sigmoid(gripper_command)>0.5, tf.float32)
						outputb_sample = tf.concat([outputb_sample, gripper_command_out], axis=1)
					if self._hyperparams.get('stop_signal', False):
						outputb_sample = tf.concat([outputb_sample, stop_signal_out], axis=1)
					local_outputbs.append(outputb_sample)
				if self._hyperparams.get('learn_final_eept', False):
					if self._hyperparams.get('pred_pick_eept', False):
						final_eept_predb, pick_eept_predb = final_eept_predb
						pick_eept_lossb = pick_labels * euclidean_loss_layer(pick_eept_predb, pick_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					else:
						pick_eept_lossb = tf.constant(0.0)
					final_eept_lossb = euclidean_loss_layer(final_eept_predb, final_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
				else:
					final_eept_lossb = tf.constant(0.0)
				discrete_loss = 0.0
				if not self._hyperparams.get('mixture_density', False):
					outputb_ = tf.identity(outputb)
				actionb_ = tf.identity(actionb)
				if self._hyperparams.get('stop_signal', False):
					if self._hyperparams.get('mixture_density', False):
						stop_signal_logitb = stop_signal
					else:
						stop_signal_logitb = tf.expand_dims(outputb[:, -1], axis=1)
					stop_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -1], axis=1), logits=stop_signal_logitb)
					discrete_loss += stop_signal_eps * tf.reduce_mean(stop_lossb)
					if not self._hyperparams.get('mixture_density', False):
						outputb_ = outputb_[:, :-1]
					actionb_ = actionb_[:, :-1]
				if self._hyperparams.get('gripper_command_signal', False):
					if self._hyperparams.get('mixture_density', False):
						gripper_command_logitb = gripper_command
					else:
						gripper_command_logitb = tf.expand_dims(outputb[:, -2], axis=1)
					gripper_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -2], axis=1), logits=gripper_command_logitb)
					discrete_loss += pick_labels * gripper_command_signal_eps * tf.reduce_mean(gripper_lossb)
					if not self._hyperparams.get('mixture_density', False):
						outputb_ = outputb_[:, :-1]
					actionb_ = actionb_[:, :-1]
				if self._hyperparams.get('mixture_density', False):
					local_lossb = act_loss_eps * tf.reduce_mean(-outputb.log_prob(actionb_)) + discrete_loss
				elif self._hyperparams.get('use_discretization', False):
					actionb_ = tf.reshape(actionb_, [-1, self.n_actions, self.n_bins])
					outputb_ = tf.reshape(outputb_, [-1, self.n_actions, self.n_bins])
					local_lossb = act_loss_eps * tf.reduce_mean([tf.losses.softmax_cross_entropy(actionb_[:, i, :], outputb_[:, i, :]) for i in xrange(self.n_actions)]) + discrete_loss
				else:
					local_lossb = act_loss_eps * euclidean_loss_layer(outputb_, actionb_, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False)) + discrete_loss
				control_lossb = tf.identity(local_lossb)
				if self._hyperparams.get('learn_loss_reg', False):
					if self._hyperparams.get('no_action', False):
						actionb_temp = tf.zeros_like(actionb)
					else:
						actionb_temp = actionb
					local_lossb_2_head = euclidean_loss_layer(outputb_2_head, actionb_temp, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					local_lossa_2_head = euclidean_loss_layer(outputa_2_head, actiona, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					hinge_loss = tf.maximum(local_lossb_2_head - local_lossa_2_head, 0.0)
					# hinge_loss = local_lossb_2_head
					local_lossb += hinge_loss_eps * hinge_loss
				else:
					hinge_loss = tf.constant(0.0)

				if self._hyperparams.get('learn_final_eept', False):
					local_lossb += final_eept_loss_eps * final_eept_lossb
					if self._hyperparams.get('pred_pick_eept', False):
						local_lossb += pick_eept_loss_eps * pick_eept_lossb
				acosine_lossb = 0.
				if self._hyperparams.get('use_acosine_loss', False):
					acosine_lossb = acosine_loss(outputb, actionb)
					local_lossb += acosine_loss_eps * acosine_lossb
				if use_whole_traj:
					# assume tbs == 1
					final_eept_lossb = euclidean_loss_layer(final_eept_predb[0], final_eeptb[0], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
				final_eept_lossesb.append(final_eept_lossb)
				control_lossesb.append(control_lossb)
				local_lossesb.append(local_lossb)
				
				# TODO: add img context var to cases where num_updates > 1
				for j in range(num_updates - 1):
					if self._hyperparams.get('use_vision', True):
						state_inputa_new = state_inputas[j+1]
						if self._hyperparams.get('use_img_context', False):
							img_context = tf.zeros_like(img_context)
							img_context += fast_img_context
							inputas[j+1] = tf.concat(axis=3, values=[inputas[j+1], img_context])
						if self._hyperparams.get('use_context', False):
							context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputas[j+1])), range(self._hyperparams.get('context_dim', 10))))
							context += fast_context
							if self._hyperparams.get('no_state'):
								state_inputa_new = context
							else:
								state_inputa_new = tf.concat(axis=1, values=[context, state_inputa_new])
						elif self._hyperparams.get('no_state'):
							state_inputa_new = None
					else:
						if self._hyperparams.get('use_context', False):
							# context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputas[j+1])), range(self._hyperparams.get('context_dim', 10))))
							# context += self.context_var
							inputas[j+1] = tf.concat(axis=1, values=[fast_context, inputas[j+1]])
						state_inputa_new = None
						
					outputa, final_eept_preda = self.forward(inputas[j+1], state_inputa_new, fast_weights, loss_idx=loss_idx, pick_labels=pick_labels, network_config=network_config)
					if self._hyperparams.get('learn_final_eept', False):
						if self._hyperparams.get('pred_pick_eept', False):
							if 'num_learned_loss' in self._hyperparams:
								final_eept_preda, pick_eept_preda, _ = final_eept_preda
							else:
								final_eept_preda, pick_eept_preda = final_eept_preda
							pick_eept_lossa = pick_labels * euclidean_loss_layer(pick_eept_preda, pick_eepta, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
						else:
							pick_eept_lossa = tf.constant(0.0)
						final_eept_lossa = euclidean_loss_layer(final_eept_preda, final_eeptas[j+1], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					else:
						final_eept_lossa = tf.constant(0.0)
					if self._hyperparams.get('learn_loss', False):
						loss = act_loss_eps * self.learn_loss(outputa, actionas[j+1], fast_weights, network_config=network_config)
					elif 'num_learned_loss' in self._hyperparams:
						loss = outputa
					else:
						loss = act_loss_eps * euclidean_loss_layer(outputa, actionas[j+1], None, multiplier=loss_multiplier, behavior_clone=True,
													use_l1=self._hyperparams.get('use_l1', False))# + fast_weights_reg / tf.to_float(self.update_batch_size)
					if self._hyperparams.get('learn_final_eept', False):
						loss += final_eept_loss_eps * final_eept_lossa
						if self._hyperparams.get('pred_pick_eept', False):
							loss += pick_eept_loss_eps * pick_eept_lossa
					if self._hyperparams.get('use_acosine_loss', False) and not self._hyperparams.get('no_action', False):
						acosine_lossa = acosine_loss(outputa, actiona)
						loss += acosine_loss_eps * acosine_lossa
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
							inputbs[j+1] = tf.concat(axis=3, values=[inputbs[j+1], img_contextb])
						if self._hyperparams.get('use_context', False):
							contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(state_inputb)), range(self._hyperparams.get('context_dim', 10))))
							contextb += fast_context
							if self._hyperparams.get('no_state'):
								state_inputb_new = contextb
							else:
								state_inputb_new = tf.concat(axis=1, values=[contextb, state_inputb_new])
						elif self._hyperparams.get('no_state'):
							state_inputb_new = None
					else:
						if self._hyperparams.get('use_context', False):
							contextb = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(inputbs[j+1])), range(self._hyperparams.get('context_dim', 10))))
							contextb += fast_context
							inputb[j+1] = tf.concat(axis=1, values=[contextb, inputb[j+1]])
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
						if self._hyperparams.get('clip_norm', False):
							clip_norm_max = self._hyperparams['clip_norm_max']
						for key in gradients.keys():
							# if 'context' not in key:
							if not self._hyperparams.get('clip_norm', False):
								gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
							else:
								gradients[key] = tf.clip_by_norm(gradients[key], clip_norm_max)
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
						output, final_eept_predb = self.forward(inputbs[j+1], state_inputb_new, fast_weights, pick_labels=pick_labels, meta_testing=True, network_config=network_config)
						# output = self.forward(inputb, state_inputb, fast_weights, update=update, is_training=False, network_config=network_config)
						if self._hyperparams.get('learn_loss_reg', False):
							outputb_2_head, _ = self.forward(inputbs[j+1], state_inputb_new, fast_weights, network_config=network_config)
							outputa_2_head, _ = self.forward(inputas[j+1], state_inputa_new, fast_weights, network_config=network_config)
					else:
						output, final_eept_predb = self.forward(inputbs[j+1], state_inputb_new, fast_weights, pick_labels=pick_labels, meta_testing=True, update=update, is_training=False, testing=testing, network_config=network_config)
						if self._hyperparams.get('learn_loss_reg', False):
							outputb_2_head, _ = self.forward(inputbs[j+1], state_inputb_new, fast_weights, update=update, is_training=False, testing=testing, network_config=network_config)
							outputa_2_head, _ = self.forward(inputas[j+1], state_inputa_new, fast_weights, update=update, is_training=False, testing=testing, network_config=network_config)
					if self._hyperparams.get('mixture_density', False):
						output, gripper_command, stop_signal = output
						if j == num_updates - 2 and testing:
							# samples = tf.stack([output.sample() for _ in range(20)])
							# sample_probs = tf.squeeze(output.prob(samples))
							# output_sample = samples[tf.argmax(sample_probs)]
							# samples = tf.stack([output.sample() for _ in range(50)])
							samples = output.sample(100)
							sample_probs = tf.squeeze(output.prob(samples))
							# this assumes T=1 at test time
							if hasattr(self, 'curr_t'):
								sample_probs = sample_probs[:, curr_t]
							output_sample = samples[tf.argmax(sample_probs, axis=0)]
						else:
							output_sample = output.sample()
						if gripper_command is not None:
							gripper_command_out = tf.cast(tf.sigmoid(gripper_command)>0.5, tf.float32)
							output_sample = tf.concat([output_sample, gripper_command_out], axis=1)
						if stop_signal is not None:
							stop_signal_out = tf.cast(tf.sigmoid(stop_signal)>0.5, tf.float32)
							output_sample = tf.concat([output_sample, stop_signal_out], axis=1)
						local_outputbs.append(output_sample)
					else:
						output_sample = tf.identity(output)
						if self._hyperparams.get('stop_signal', False):
							stop_signal = tf.expand_dims(output_sample[:, -1], axis=1)
							output_sample = output_sample[:, :-1]
							stop_signal_out = tf.cast(tf.sigmoid(stop_signal)>0.5, tf.float32)
						if self._hyperparams.get('gripper_command_signal', False):
							gripper_command = tf.expand_dims(output_sample[:, -1], axis=1)
							output_sample = output_sample[:, :-1]
							gripper_command_out = tf.cast(tf.sigmoid(gripper_command)>0.5, tf.float32)
						if self._hyperparams.get('use_discretization', False):
							output_sample = tf.reshape(output_sample, [-1, self.n_actions, self.n_bins])
							output_sample = tf.nn.softmax(output_sample, dim=2)
							output_sample = tf.reduce_sum(tf.one_hot(tf.argmax(output_sample, axis=2), depth=self.n_bins) * avg_bin_action, axis=2)
						if self._hyperparams.get('gripper_command_signal', False):
							output_sample = tf.concat([output_sample, gripper_command_out], axis=1)
						if self._hyperparams.get('stop_signal', False):
							output_sample = tf.concat([output_sample, stop_signal_out], axis=1)
						local_outputbs.append(output_sample)
					if self._hyperparams.get('learn_final_eept', False):
						if self._hyperparams.get('pred_pick_eept', False):
							final_eept_predb, pick_eept_predb = final_eept_predb
							pick_eept_lossb = pick_labels * euclidean_loss_layer(pick_eept_predb, pick_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
						else:
							pick_eept_lossb = tf.constant(0.0)
						final_eept_lossb = euclidean_loss_layer(final_eept_predb, final_eeptb, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					else:
						final_eept_lossb = tf.constant(0.0)
					# fast_weights_reg = tf.reduce_sum([self.weight_decay*tf.nn.l2_loss(var) for var in fast_weights.values()]) / tf.to_float(self.T)
					discrete_loss = 0.0
					if not self._hyperparams.get('mixture_density', False):
						output_ = tf.identity(output)
					actionb_ = tf.identity(actionb)
					if self._hyperparams.get('stop_signal', False):
						if self._hyperparams.get('mixture_density', False):
							stop_signal_logitb = stop_signal
						else:
							stop_signal_logitb = tf.expand_dims(output[:, -1], axis=1)
						stop_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -1], axis=1), logits=stop_signal_logitb)
						discrete_loss += stop_signal_eps * tf.reduce_mean(stop_lossb)
						if not self._hyperparams.get('mixture_density', False):
							output_ = output_[:, :-1]
						actionb_ = actionb_[:, :-1]
					if self._hyperparams.get('gripper_command_signal', False):
						if self._hyperparams.get('mixture_density', False):
							gripper_command_logitb = gripper_command
						else:
							gripper_command_logitb = tf.expand_dims(output[:, -2], axis=1)
						gripper_lossb = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(actionb[:, -2], axis=1), logits=gripper_command_logitb)
						discrete_loss += pick_labels * gripper_command_signal_eps * tf.reduce_mean(gripper_lossb)
						if not self._hyperparams.get('mixture_density', False):
							output_ = output_[:, :-1]
						actionb_ = actionb_[:, :-1]
					if self._hyperparams.get('mixture_density', False):
						lossb = act_loss_eps * tf.reduce_mean(-output.log_prob(actionb_)) + discrete_loss
					elif self._hyperparams.get('use_discretization', False):
						actionb_ = tf.reshape(actionb_, [-1, self.n_actions, self.n_bins])
						output_ = tf.reshape(output_, [-1, self.n_actions, self.n_bins])
						lossb = act_loss_eps * tf.reduce_mean([tf.losses.softmax_cross_entropy(actionb_[:, i, :], output_[:, i, :]) for i in xrange(self.n_actions)]) + discrete_loss
					else:
						lossb = act_loss_eps * euclidean_loss_layer(output_, actionb_, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False)) + discrete_loss
					control_lossb = tf.identity(lossb)
					if self._hyperparams.get('learn_loss_reg', False):
						if self._hyperparams.get('no_action', False):
							actionb_temp = tf.zeros_like(actionb)
						else:
							actionb_temp = actionb
						lossb_2_head = euclidean_loss_layer(outputb_2_head, actionb_temp, None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
						lossa_2_head = euclidean_loss_layer(outputa_2_head, actionas[j+1], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
						hinge_loss = tf.maximum(lossb_2_head - lossa_2_head, 0.0)
						lossb += hinge_loss_eps * hinge_loss
					else:
						hinge_loss = tf.constant(0.0)
					if self._hyperparams.get('learn_final_eept', False):
						lossb += final_eept_loss_eps * final_eept_lossb
						if self._hyperparams.get('pred_pick_eept', False):
							lossb += pick_eept_loss_eps * pick_eept_lossb
					if self._hyperparams.get('use_acosine_loss', False):
						acosine_lossb = acosine_loss(output, actionb)
						local_lossb += acosine_loss_eps * acosine_lossb
					if use_whole_traj:
						# assume tbs == 1
						final_eept_lossb = euclidean_loss_layer(final_eept_predb[0], final_eeptb[0], None, multiplier=loss_multiplier, behavior_clone=True, use_l1=self._hyperparams.get('use_l1', False))
					final_eept_lossesb.append(final_eept_lossb)
					local_lossesb.append(lossb)
					control_lossesb.append(control_lossb)
				if self._hyperparams.get('use_grad_reg'):
					fast_gradient_reg = 0.0
					for key in gradients.keys():
						fast_gradient_reg += self.grad_reg*tf.reduce_sum(tf.abs(gradients[key]))
					local_lossesb[-1] += self._hyperparams['grad_reg'] *fast_gradient_reg / self.update_batch_size
				if 'num_learned_loss' in self._hyperparams:
					local_lossesb[-1] += task_label_loss_eps * task_label_loss
				# local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, moving_mean, moving_variance, moving_mean_test, moving_variance_test]
				# local_fn_output = [local_outputa, local_outputbs, test_outputa, local_lossa, local_lossesb, flat_img_inputa, fp, conv_layer_2, outputs, test_outputs, mean, variance, moving_mean, moving_variance, moving_mean_new, moving_variance_new]
				fast_weights_values = [fast_weights[key] for key in self.sorted_weight_keys]
				# use post update output
				local_fn_output = [local_outputa, local_outputbs, 
									local_outputbs[-1], local_lossa, local_lossesb, 
									final_eept_lossesb, control_lossesb, flat_img_inputa, 
									flat_img_inputb, final_eept_predb[0], fast_weights_values, 
									gradients_summ, acosine_lossb, pick_eept_lossa, 
									pick_eept_lossb, task_label_loss, loss_idx]
				if self._hyperparams.get('stop_signal', False):
					local_fn_output.append(tf.reduce_mean(stop_lossb))
				if self._hyperparams.get('gripper_command_signal', False):
					local_fn_output.append(tf.reduce_mean(gripper_lossb))
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
				if self._hyperparams.get('aggressive_light_aug', False) and 'Testing' not in prefix:
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0], pick_labels, task_labels, lighting))
						else:
							unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0], pick_labels, lighting))
					else:
						unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0], lighting))
				else:
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0], pick_labels, task_labels))
						else:
							unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0], pick_labels))
					else:
						unused = batch_metalearn((inputa[0], inputb[0], actiona[0], actionb[0]))
		
		# out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, [tf.float32]*num_updates, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
		out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, 
					tf.float32, [tf.float32]*num_updates, 
					[tf.float32]*num_updates, [tf.float32]*num_updates, 
					tf.float32, tf.float32, tf.float32, 
					[tf.float32]*len(self.weights.keys()), 
					[[tf.float32]*len(self.weights.keys())]*num_updates, 
					tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
		if self._hyperparams.get('stop_signal', False):
			out_dtype.append(tf.float32)
		if self._hyperparams.get('gripper_command_signal', False):
			out_dtype.append(tf.float32)
		if self._hyperparams.get('aggressive_light_aug', False) and 'Testing' not in prefix:
			if self._hyperparams.get('pred_pick_eept', False):
				if 'num_learned_loss' in self._hyperparams:
					result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb, pick_labels, task_labels, lighting), dtype=out_dtype)
				else:
					result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb, pick_labels, lighting), dtype=out_dtype)
			else:
				result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb, lighting), dtype=out_dtype)
		else:
			if self._hyperparams.get('pred_pick_eept', False):
				if 'num_learned_loss' in self._hyperparams:
					result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb, pick_labels, task_labels), dtype=out_dtype)
				else:
					result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, actiona, actionb, pick_labels), dtype=out_dtype)
			else:            
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
		if not self._hyperparams.get('use_vision', True) and demos[0]['demoX'].shape[-1] > self._dO:
			for key in demos.keys():
				demos[key]['demoX'] = demos[key]['demoX'][:, :, :-9].copy()
		if demos[0]['demoX'].shape[-1] > len(self.x_idx):
			for key in demos.keys():
				if self._hyperparams.get('sample_traj', False):
					if self._hyperparams.get('eemod', False):
						# demos[key]['demoX'] = demos[key]['demoX'][:, :, :, 7:10].copy()
						demos[key]['demoX'] = demos[key]['demoX'][:, :, :, 7:].copy()
						# demos[key]['demoU'] = demos[key]['demoU'][:, :, :, 7:].copy()
						if self._hyperparams.get('act_xy', False):
							demos[key]['demoU'] = demos[key]['demoU'][:, :, :, 7:9].copy()
						else:
							demos[key]['demoU'] = demos[key]['demoU'][:, :, :, 7:10].copy()
					else:
						demos[key]['demoX'] = demos[key]['demoX'][:, :, :, -len(self.x_idx):].copy()
				else:
					demos[key]['demoX'] = demos[key]['demoX'][:, :, -len(self.x_idx):].copy()
		# import pdb; pdb.set_trace()
		# use dU = 6 for now
		if self._hyperparams.get('use_discretization', False):
			print("Start discretization")
			self.n_actions = self._hyperparams.get('num_actions', 6)
			self.n_bins = self._hyperparams.get('num_bins', 50)
			self.U_min = np.amin(np.array([np.amin(demos[key]['demoU'][:,:,:self.n_actions], axis=(0,1)) for key in demos.keys()]), axis=0)
			self.U_max = np.amax(np.array([np.amax(demos[key]['demoU'][:,:,:self.n_actions], axis=(0,1)) for key in demos.keys()]), axis=0)
			self.bin_size = (self.U_max - self.U_min) / self.n_bins
			self.avg_bin_action = np.array([np.linspace(self.U_min[i], self.U_max[i], self.n_bins + 1) for i in xrange(self.n_actions)])
			self.avg_bin_action = (self.avg_bin_action[:, 1:] + self.avg_bin_action[:, :-1]) / 2
			print('Finish discretization')
		if self._hyperparams.get('push', False):
			for key in demos.keys():
				demos[key]['demoX'] = demos[key]['demoX'][6:-6, :, :].copy()
				demos[key]['demoU'] = demos[key]['demoU'][6:-6, :, :].copy()
		if self._hyperparams.get('sample_traj', False):
			num_samples = self._hyperparams.get('num_samples', 5)
			for key in demos.keys():
				try:
					assert len(demos[key]['demoX'].shape) == 4 and demos[key]['demoX'].shape[0] == num_samples
				except AssertionError:
					import pdb; pdb.set_trace()
				if not self._hyperparams.get('no_sample', False):
					demos[key]['demoX'] = demos[key]['demoX'].transpose(1, 0, 2, 3).reshape(-1, self.T, len(self.x_idx))
					demos[key]['demoU'] = demos[key]['demoU'].transpose(1, 0, 2, 3).reshape(-1, self.T, self._dU-len(self._hyperparams['final_eept_range']))
				else:
					demos[key]['demoX'] = demos[key]['demoX'][0]
					demos[key]['demoU'] = demos[key]['demoU'][0]
		if self._hyperparams.get('sawyer', False):
			final_eept_range = self._hyperparams['final_eept_range_state']
			for key in demos.keys():
				final_eept = np.tile(np.expand_dims(demos[key]['demoX'][:, -1, final_eept_range[0]:final_eept_range[-1]+1].copy(), axis=1), [1, self.T, 1])
				# demos[key]['demoX'] = demos[key]['demoX'][:, :, :final_eept_range[0]].copy()
				demos[key]['demoU'] = np.concatenate((demos[key]['demoU'].copy(), final_eept), axis=2)
		n_folders = len(demos.keys())
		n_val = self._hyperparams['n_val'] # number of demos for testing
		N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.iteritems())
		print "Number of demos: %d" % N_demos
		idx = np.arange(n_folders)
		# shuffle(idx)
		if not hasattr(self, 'train_idx'):
			if n_val != 0:
				if self._hyperparams.get('shuffle_val', False):
					self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
					mask = np.array([(i in self.val_idx) for i in idx])
					self.train_idx = np.sort(idx[~mask])
				elif 'val_list' in self._hyperparams:
					self.val_idx = np.array(self._hyperparams['val_list'])
					assert len(self.val_idx) == n_val
					mask = np.array([(i in self.val_idx) for i in idx])
					self.train_idx = np.sort(idx[~mask])
				else:
					self.val_idx = idx[-n_val:]
					self.train_idx = idx[:-n_val]
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
			# seems problematic when using noisy demos? Should keep both scale and bias for noisy and good demos?
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
		if 'push_train_list' in self._hyperparams:
			self.push_train_list = self._hyperparams['push_train_list']
		if self._hyperparams.get('pred_pick_eept', False):
			self.train_pick_idxes, self.train_other_idxes, self.val_pick_idxes, self.val_other_idxes = [], [], [], []
			for idx in self.train_idx:
				if demos[idx]['is_pick_place'] == 1.:
					if hasattr(self, 'push_train_list'):
						assert idx not in self.push_train_list
					self.train_pick_idxes.append(idx)
				else:
					if hasattr(self, 'push_train_list') and idx in self.push_train_list:
						continue
					self.train_other_idxes.append(idx)
			for idx in self.val_idx:
				if demos[idx]['is_pick_place'] == 1.:
					self.val_pick_idxes.append(idx)
				else:
					self.val_other_idxes.append(idx)
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
					O = np.array(imageio.mimread(self.demo_gif_dir + self.gif_prefix + '%d_noisy/cond%d.samp0.gif' % (idx, selected_cond)))[:, :, :, :3]
				else:
					O = np.array(imageio.mimread(self.demo_gif_dir + self.gif_prefix + '%d/cond%d.samp0.gif' % (idx, selected_cond)))[:, :, :, :3]
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
			train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '%d' % i) for i in self.train_idx}
			# self.val_img_folders = {i: os.path.join(self.demo_gif_dir, 'color_%d' % i) for i in self.val_idx}
			val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '%d' % i) for i in self.val_idx}
			# val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d' % (i+93)) for i in self.val_idx}
			if self._hyperparams.get('use_depth', False):
				train_depth_img_folders = {i: os.path.join(self.demo_gif_depth_dir, self.gif_prefix + '%d' % i) for i in self.train_idx}
				val_depth_img_folders = {i: os.path.join(self.demo_gif_depth_dir, self.gif_prefix + '%d' % i) for i in self.val_idx}
			if noisy:
				# noisy_train_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d_noisy' % i) for i in self.train_idx}
				# noisy_val_img_folders = {i: os.path.join(self.demo_gif_dir, self.gif_prefix + '_%d_noisy' % i) for i in self.val_idx}
				noisy_train_img_folders = {i: os.path.join(self.noisy_demo_gif_dir, self.gif_prefix + '%d' % i) for i in self.train_idx}
				noisy_val_img_folders = {i: os.path.join(self.noisy_demo_gif_dir, self.gif_prefix + '%d' % i) for i in self.val_idx}
				if self._hyperparams.get('use_depth', False):
					noisy_train_depth_img_folders = {i: os.path.join(self.noisy_demo_gif_depth_dir, self.gif_prefix + '%d' % i) for i in self.train_idx}
					noisy_val_depth_img_folders = {i: os.path.join(self.noisy_demo_gif_depth_dir, self.gif_prefix + '%d' % i) for i in self.val_idx}
			TEST_PRINT_INTERVAL = 500
			TOTAL_ITERS = self._hyperparams['iterations']
			is_push = self._hyperparams.get('push', False)
			# VAL_ITERS = int(TOTAL_ITERS / 500)
			self.all_training_filenames = []
			self.all_val_filenames = []
			self.all_training_filenames_depth = []
			self.all_val_filenames_depth = []
			self.training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
			self.val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, int(TOTAL_ITERS/TEST_PRINT_INTERVAL))}
			if noisy:
				self.noisy_training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
				self.noisy_val_batch_idx = {i: OrderedDict() for i in TEST_PRINT_INTERVAL*np.arange(1, TOTAL_ITERS/TEST_PRINT_INTERVAL)}
			if self._hyperparams.get('aggressive_light_aug', False):
				self.lighting_idx = []
			for itr in xrange(TOTAL_ITERS):
				if self._hyperparams.get('aggressive_light_aug', False):
					self.lighting_idx.append(np.random.choice(range(self._hyperparams.get('num_light_samples', 3))))
				if not self._hyperparams.get('pred_pick_eept', False):
					sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
				else:
					sampled_train_pick_idx = random.sample(self.train_pick_idxes, self.train_pick_batch_size)
					if hasattr(self, 'push_train_list'):
						sampled_train_push_idx = random.sample(self.push_train_list, self.train_pick_batch_size)
						sampled_train_other_idx = random.sample(self.train_other_idxes, self.meta_batch_size - 2*self.train_pick_batch_size)
						sampled_train_other_idx = sampled_train_other_idx + sampled_train_push_idx
					else:
						sampled_train_other_idx = random.sample(self.train_other_idxes, self.meta_batch_size-self.train_pick_batch_size)
					sampled_train_idx = sampled_train_pick_idx + sampled_train_other_idx
				for idx in sampled_train_idx:
					if self._hyperparams.get('use_vision', True):
						try:
							sampled_folder = train_img_folders[idx]
						except:
							import pdb; pdb.set_trace()
						if self._hyperparams.get('use_depth', False):
							depth_sampled_folder = train_depth_img_folders[idx]
							depth_image_paths = natsorted(os.listdir(depth_sampled_folder))  
						if self._hyperparams.get('sample_traj', False) and self._hyperparams.get('no_sample', False):
							image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'samp0' in p])
						elif self._hyperparams.get('bird_view', False):
							image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'view0' in p])
						elif self._hyperparams.get('kinect_view', False):
							image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'view1' in p])
						else:    
							image_paths = natsorted(os.listdir(sampled_folder))
						if is_push:
							image_paths = image_paths[6:-6]
						try:
							assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
						except AssertionError:
							import pdb; pdb.set_trace()
						if noisy:
							noisy_sampled_folder = noisy_train_img_folders[idx]
							if self._hyperparams.get('use_depth', False):
								noisy_depth_sampled_folder = noisy_train_depth_img_folders[idx]
								noisy_depth_image_paths = natsorted(os.listdir(noisy_depth_sampled_folder))
							# noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
							if self._hyperparams.get('sample_traj', False) and self._hyperparams.get('no_sample', False):
								noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'samp0' in p])
							elif self._hyperparams.get('bird_view', False):
								noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'view0' in p])
							elif self._hyperparams.get('kinect_view', False):
								noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'view1' in p])
							else:
								noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
							if is_push:
								if idx != 320:
									noisy_image_paths = noisy_image_paths[6:-6]
								else:
									noisy_image_paths = noisy_image_paths[6:-4]
							if not self._hyperparams.get('human_demo', False):
								try:
									assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
								except AssertionError:
									import pdb; pdb.set_trace()
						if not noisy:
							if self._hyperparams.get('sample_traj', False) and self._hyperparams['num_samples'] > 1:
								num_samples = self._hyperparams['num_samples']
								sampled_cond_idx = np.random.choice(range(len(image_paths)//num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
								sampled_sample_idx = np.random.choice(range(num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
								sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx, sampled_sample_idx)]
							else:
								sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
							sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
							# assume no sample traj
							if self._hyperparams.get('use_depth', False):
								depth_sampled_images = [os.path.join(depth_sampled_folder, depth_image_paths[i]) for i in sampled_image_idx]
						else:
							if self._hyperparams.get('sample_traj', False) and self._hyperparams['num_samples'] > 1:
								num_samples = self._hyperparams['num_samples']
								sampled_cond_idx = np.random.choice(range(len(image_paths)//num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
								sampled_sample_idx = np.random.choice(range(num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
								noisy_sampled_image_idx = sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx[:self.update_batch_size], sampled_sample_idx[:self.update_batch_size])]
								sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx[self.update_batch_size:], sampled_sample_idx[self.update_batch_size:])]
							else:
								noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False)
								sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False)
							sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
							sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
							if self._hyperparams.get('use_depth', False):
								depth_sampled_images = [os.path.join(noisy_depth_sampled_folder, noisy_depth_image_paths[i]) for i in noisy_sampled_image_idx]
								depth_sampled_images.extend([os.path.join(depth_sampled_folder, depth_image_paths[i]) for i in sampled_image_idx])
						self.all_training_filenames.extend(sampled_images)
						if self._hyperparams.get('use_depth', False):
							self.all_training_filenames_depth.extend(depth_sampled_images)
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
					if not self._hyperparams.get('pred_pick_eept', False):
						sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
					else:
						sampled_val_pick_idx = random.sample(self.val_pick_idxes, self.val_pick_batch_size)
						sampled_val_other_idx = random.sample(self.val_other_idxes, self.meta_batch_size-self.val_pick_batch_size)
						sampled_val_idx = sampled_val_pick_idx + sampled_val_other_idx
					for idx in sampled_val_idx:
						if self._hyperparams.get('use_vision', True):
							sampled_folder = val_img_folders[idx]
							if self._hyperparams.get('use_depth', False):
								depth_sampled_folder = val_depth_img_folders[idx]
								depth_image_paths = natsorted(os.listdir(depth_sampled_folder)) 
							if self._hyperparams.get('sample_traj', False) and self._hyperparams.get('no_sample', False):
								image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'samp0' in p])
							elif self._hyperparams.get('bird_view', False):
								image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'view0' in p])
							elif self._hyperparams.get('kinect_view', False):
								image_paths = natsorted([p for p in os.listdir(sampled_folder) if 'view1' in p])
							else:    
								image_paths = natsorted(os.listdir(sampled_folder))
							if is_push:
								image_paths = image_paths[6:-6]
							assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
							if noisy:
								noisy_sampled_folder = noisy_val_img_folders[idx]
								if self._hyperparams.get('use_depth', False):
									noisy_depth_sampled_folder = noisy_val_depth_img_folders[idx]
									noisy_depth_image_paths = natsorted(os.listdir(noisy_depth_sampled_folder))
								# noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
								if self._hyperparams.get('sample_traj', False) and self._hyperparams.get('no_sample', False):
									noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'samp0' in p])
								elif self._hyperparams.get('bird_view', False):
									noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'view0' in p])
								elif self._hyperparams.get('kinect_view', False):
									noisy_image_paths = natsorted([p for p in os.listdir(noisy_sampled_folder) if 'view1' in p])
								else:    
									noisy_image_paths = natsorted(os.listdir(noisy_sampled_folder))
								if is_push:
									noisy_image_paths = noisy_image_paths[6:-6]
								if not self._hyperparams.get('human_demo', False):
									assert len(noisy_image_paths) == self.noisy_demos[idx]['demoX'].shape[0]
							if not noisy:
								if self._hyperparams.get('sample_traj', False) and self._hyperparams['num_samples'] > 1:
									num_samples = self._hyperparams['num_samples']
									sampled_cond_idx = np.random.choice(range(len(image_paths)//num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
									sampled_sample_idx = np.random.choice(range(num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
									sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx, sampled_sample_idx)]
								else:
									sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False)
								# sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size+self.test_batch_size, replace=False) # True
								sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
								if self._hyperparams.get('use_depth', False):
									depth_sampled_images = [os.path.join(depth_sampled_folder, depth_image_paths[i]) for i in sampled_image_idx]
							else:
								if self._hyperparams.get('sample_traj', False) and self._hyperparams['num_samples'] > 1:
									num_samples = self._hyperparams['num_samples']
									sampled_cond_idx = np.random.choice(range(len(image_paths)//num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
									sampled_sample_idx = np.random.choice(range(num_samples), size=self.update_batch_size+self.test_batch_size, replace=False)
									noisy_sampled_image_idx = sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx[:self.update_batch_size], sampled_sample_idx[:self.update_batch_size])]
									sampled_image_idx = [cond_idx*num_samples+sample_idx for cond_idx, sample_idx in zip(sampled_cond_idx[self.update_batch_size:], sampled_sample_idx[self.update_batch_size:])]
								else:
									noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False)
									sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False)
								# noisy_sampled_image_idx = np.random.choice(range(len(noisy_image_paths)), size=self.update_batch_size, replace=False) # True
								# sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.test_batch_size, replace=False) # True
								sampled_images = [os.path.join(noisy_sampled_folder, noisy_image_paths[i]) for i in noisy_sampled_image_idx]
								sampled_images.extend([os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx])
								if self._hyperparams.get('use_depth', False):
									depth_sampled_images = [os.path.join(noisy_depth_sampled_folder, noisy_depth_image_paths[i]) for i in noisy_sampled_image_idx]
									depth_sampled_images.extend([os.path.join(depth_sampled_folder, depth_image_paths[i]) for i in sampled_image_idx])
							self.all_val_filenames.extend(sampled_images)
							if self._hyperparams.get('use_depth', False):
								self.all_val_filenames_depth.extend(depth_sampled_images)
							self.val_batch_idx[itr][idx] = sampled_image_idx
							if noisy:
								self.noisy_val_batch_idx[itr][idx] = noisy_sampled_image_idx
						else:
							if noisy:
								self.val_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.test_batch_size, replace=False) # True
								self.noisy_val_batch_idx[itr][idx] = np.random.choice(range(self.noisy_demos[idx]['demoX'].shape[0]), size=self.update_batch_size, replace=False) # True
							else:
								self.val_batch_idx[itr][idx] = np.random.choice(range(self.demos[idx]['demoX'].shape[0]), size=self.update_batch_size+self.test_batch_size, replace=False) # True

	def make_batch_tensor(self, network_config, restore_iter=0, use_depth=False, train=True):
		# TODO: load images using tensorflow fileReader and gif decoder
		TEST_INTERVAL = 500
		batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size
		if train:
			if use_depth:
				all_filenames = self.all_training_filenames_depth
			else:
				all_filenames = self.all_training_filenames
			if restore_iter > 0:
				all_filenames = all_filenames[batch_image_size*(restore_iter+1):]
		else:
			if use_depth:
				all_filenames = self.all_val_filenames_depth
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
		if use_depth:
			# assuming padding the grayscale depth video into rgb
			image = tf.expand_dims(image[:, :, :, 0], axis=3)
		image = tf.cast(image, tf.float32)
		image /= 255.0
		if self._hyperparams.get('use_hsv', False) and not use_depth:
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
			if self._hyperparams.get('rotate_human', False):
				image_human = image[:self.update_batch_size]
				image_robot = image[self.update_batch_size:]
				image_rotated = []
				rotate_angles = np.random.choice(range(1, 4), size=self.update_batch_size, replace=True)
				for j in xrange(self.update_batch_size):
					img = image_human[j]
					img = tf.transpose(tf.reshape(img, [self.T, num_channels, im_width, im_height]), perm=[0, 3, 2, 1])
					if self._hyperparams.get('flip_left_right_human', False):
						img = tf.contrib.image.rotate(img, angles=np.pi)
					else:
						img = tf.contrib.image.rotate(img, angles=np.pi*2.*np.random.random())
					img = tf.transpose(img, perm=[0, 3, 2, 1])
					img = tf.reshape(img, [self.T, -1])
					image_rotated.append(img)
				image = tf.concat(axis=0, values=[tf.stack(image_rotated), image_robot])
			image = tf.reshape(image, [(self.update_batch_size+self.test_batch_size)*self.T, -1])
			all_images.append(image)
		return tf.stack(all_images)
	
	def generate_light_aug(self):
		self.light_aug = {}
		num_samples = self._hyperparams.get('num_light_samples', 3)
		rng = self._hyperparams.get('rng', 0.3)
		if 'num_push_train_1' in self._hyperparams:
			num_place = len(self.train_idx) - self._hyperparams['num_push_train_1'] - self._hyperparams['num_push_train_2']
			for i in xrange(self._hyperparams['num_push_train_1'] // 2):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({2*i: light, 2*i+1: light.copy()})
			for i in xrange(num_place//3):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({self._hyperparams['num_push_train_1']+3*i: light, self._hyperparams['num_push_train_1']+3*i+1: light.copy(), self._hyperparams['num_push_train_1']+3*i+2: light.copy()})
			for i in xrange(self._hyperparams['num_push_train_2'] // 2):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({self._hyperparams['num_push_train_1']+num_place+2*i: light, self._hyperparams['num_push_train_1']+num_place+2*i+1: light.copy()})
			for i in xrange((len(self.val_idx) - self._hyperparams['num_push_val'])//3):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({len(self.train_idx)+3*i: light, len(self.train_idx)+3*i+1: light.copy(), len(self.train_idx)+3*i+2: light.copy()})
			for i in xrange(self._hyperparams['num_push_val'] // 2):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({len(self.demos) - self._hyperparams['num_push_val']+2*i: light, len(self.demos) - self._hyperparams['num_push_val']+2*i+1: light.copy()})
		else:
			for i in xrange(len(self.demos) // 3):
				light = np.random.uniform(low=-rng, high=rng, size=(num_samples, 1))
				self.light_aug.update({3*i: light, 3*i+1: light.copy(), 3*i+2: light.copy()})
		print 'Finished generating light augmentation'
		
	def generate_data_batch(self, itr, train=True):
		if train:
			demos = {key: self.demos[key].copy() for key in self.train_idx}
			idxes = self.training_batch_idx[itr]
			if self._hyperparams.get('use_noisy_demos', False) and not self._hyperparams.get('human_demo', False):
				noisy_demos = {key: self.noisy_demos[key].copy() for key in self.train_idx}
				noisy_idxes = self.noisy_training_batch_idx[itr]
		else:
			demos = {key: self.demos[key].copy() for key in self.val_idx}
			idxes = self.val_batch_idx[itr]
			if self._hyperparams.get('use_noisy_demos', False) and not self._hyperparams.get('human_demo', False):
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
			U = [demos[k]['demoU'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
			X = [demos[k]['demoX'][v].reshape(test_batch_size*self.T, -1) for k, v in idxes.items()]
			if not self._hyperparams.get('human_demo', False):
				noisy_U = [noisy_demos[k]['demoU'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
				noisy_X = [noisy_demos[k]['demoX'][v].reshape(update_batch_size*self.T, -1) for k, v in noisy_idxes.items()]
				U = np.concatenate((np.array(noisy_U), np.array(U)), axis=1)
				X = np.concatenate((np.array(noisy_X), np.array(X)), axis=1)
			else:
				U = np.array(U)
				X = np.array(X)
		if self._hyperparams.get('pred_pick_eept'):
			pick_labels = np.array([demos[k]['is_pick_place'] for k in idxes.keys()]).astype('f')
		if 'num_learned_loss' in self._hyperparams:
			task_labels = np.array([demos[k]['task_labels'] for k in idxes.keys()]).astype('f')
		try:
			assert U.shape[2] == self._dU
		except AssertionError:
			import pdb; pdb.set_trace()
		assert X.shape[2] == len(self.x_idx)
		if not self._hyperparams.get('aggressive_light_aug', False):
			if self._hyperparams.get('pred_pick_eept', False):
				if 'num_learned_loss' in self._hyperparams:
				   return X, U, pick_labels, task_labels 
				return X, U, pick_labels
			return X, U
		else:
			lighting = np.array([self.light_aug[k][self.lighting_idx[itr]] for k in idxes.keys()])
			if self._hyperparams.get('pred_pick_eept', False):
				if 'num_learned_loss' in self._hyperparams:
					return X, U, pick_labels, task_labels, lighting
				return X, U, pick_labels, lighting
			return X, U, lighting
		
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
		train_writer = tf.summary.FileWriter(log_dir, self.graph)
		# actual training.
		with Timer('Training'):
			if self.restore_iter == 0:
				training_range = range(TOTAL_ITERS)
			else:
				training_range = range(self.restore_iter+1, TOTAL_ITERS)
			for itr in training_range:
				# TODO: need to make state and obs compatible
				if self._hyperparams.get('aggressive_light_aug', False):
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							state, tgt_mu, pick_labels, task_labels, lighting = self.generate_data_batch(itr)
						else:
							state, tgt_mu, pick_labels, lighting = self.generate_data_batch(itr)
					else:
						state, tgt_mu, lighting = self.generate_data_batch(itr)
				else:
					if self._hyperparams.get('pred_pick_eept', False):
						if 'num_learned_loss' in self._hyperparams:
							state, tgt_mu, pick_labels, task_labels = self.generate_data_batch(itr)
						else:
							state, tgt_mu, pick_labels = self.generate_data_batch(itr)
					else:
						state, tgt_mu = self.generate_data_batch(itr)
				if self._hyperparams.get('human_demo', False):
					feed_dict = {self.stateb: state,
							self.actionb: tgt_mu}
				else:
					statea = state[:, :self.update_batch_size*self.T, :]
					stateb = state[:, self.update_batch_size*self.T:, :]
					actiona = tgt_mu[:, :self.update_batch_size*self.T, :]
					actionb = tgt_mu[:, self.update_batch_size*self.T:, :]
					feed_dict = {self.statea: statea,
								self.stateb: stateb,
								self.actiona: actiona,
								self.actionb: actionb}
				if self._hyperparams.get('aggressive_light_aug', False):
					feed_dict[self.lighting_tensor] = lighting
				if self._hyperparams.get('pred_pick_eept', False):
					feed_dict[self.pick_labels] = pick_labels
					if 'num_learned_loss' in self._hyperparams:
						feed_dict[self.task_labels] = task_labels
				input_tensors = [self.train_op]
				# if self.use_batchnorm:
				#     feed_dict[self.phase] = 1
				if self.norm_type == 'vbn':
					feed_dict[self.reference_tensor] = self.reference_batch
					input_tensors.append(self.reference_out)
				if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
					input_tensors.append(self.train_act_op)
					# input_tensors.append(self.task_label_pred)
					input_tensors.extend([self.train_summ_op, self.total_loss1, self.total_losses2[self.num_updates-1]])
				result = self.run(input_tensors, feed_dict=feed_dict)
	
				if itr != 0 and itr % SUMMARY_INTERVAL == 0:
					prelosses.append(result[-2])
					train_writer.add_summary(result[-3], itr)
					postlosses.append(result[-1])
	
				if itr != 0 and itr % PRINT_INTERVAL == 0:
					print 'Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(prelosses), np.mean(postlosses))
					# print 'Predicted task label is ' , result[-4]
					# print 'Actual task label is ', actiona[:, 0, -3:]
					# print 'predict eept is ', result[-4][0, 0, :]
					# print 'true eept is ', actionb[0, 0, 6:8]
					prelosses, postlosses = [], []

				if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
					if len(self.val_idx) > 0:
						input_tensors = [self.val_summ_op, self.val_total_loss1, self.val_total_losses2[self.num_updates-1]]
						if self._hyperparams.get('aggressive_light_aug', False):
							if self._hyperparams.get('pred_pick_eept', False):
								if 'num_learned_loss' in self._hyperparams:
									val_state, val_act, pick_labels, task_labels, lighting = self.generate_data_batch(itr, train=False)
								else:
									val_state, val_act, pick_labels, lighting = self.generate_data_batch(itr, train=False)
							else:
								val_state, val_act, lighting = self.generate_data_batch(itr, train=False)
						else:
							if self._hyperparams.get('pred_pick_eept', False):
								if 'num_learned_loss' in self._hyperparams:
									val_state, val_act, pick_labels, task_labels = self.generate_data_batch(itr, train=False)
								else:
									val_state, val_act, pick_labels = self.generate_data_batch(itr, train=False)
							else:
								val_state, val_act = self.generate_data_batch(itr, train=False)
						if self._hyperparams.get('human_demo', False):
							feed_dict = {self.stateb: val_state,
									self.actionb: val_act}
						else:
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
						if self._hyperparams.get('aggressive_light_aug', False):
							feed_dict[self.lighting_tensor] = lighting
						if self._hyperparams.get('pred_pick_eept', False):
							feed_dict[self.pick_labels] = pick_labels
							if 'num_learned_loss' in self._hyperparams:
								feed_dict[self.task_labels] = task_labels
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
                        gif_dir = gif_dir + 'color_%d/' % idx
                        mkdir_p(gif_dir)
                        gif_name = os.path.join(gif_dir,'cond%d.samp%d.gif' % (conditions[i], j))
                    else:
                        gif_name=None
                        gif_fps = None
                else:
                    gif_name=None
                    gif_fps = None
                if self._hyperparams.get('use_vision', True):
                    save_state = False
                else:
                    save_state = True
                samples.append(agent.sample(
                    self.policy, conditions[i],
                    save=False, noisy=False, save_state=save_state,
                    record_gif=gif_name, record_gif_fps=gif_fps, task_idx=idx))
        return SampleList(samples)

    def eval_success_rate(self, test_agent):
        assert type(test_agent) is list
        success_thresh = test_agent[0]['filter_demos'].get('success_upper_bound', 0.05)
        state_idx = np.array(list(test_agent[0]['filter_demos'].get('state_idx', range(4, 7))))
        train_dists = []
        val_dists = []
        # if len(test_agent) > 450:
        #     idx = np.random.choice(np.arange(len(test_agent)-150), replace=False, size=300)
        #     train_agents = {i: test_agent[i] for i in idx}
        #     train_agents.update({i: test_agent[i] for i in range(len(test_agent))[-150:]})
        #     test_agent = train_agents
        for i in xrange(len(test_agent)):
        # for i in test_agent.keys():
            agent = test_agent[i]['type'](test_agent[i])
            conditions = self.demos[i]['demoConditions']
            target_eepts = np.array(test_agent[i]['target_end_effector'])[conditions]
            if len(target_eepts.shape) == 1:
                target_eepts = np.expand_dims(target_eepts, axis=0)
            target_eepts = target_eepts[:, :3]
            if i in self.val_idx:
                # Sample on validation conditions and get the states.
                X_val = self.sample(agent, i, conditions, N=1, testing=True).get_X()
                # min distance of the last 10 timesteps
                val_dists.extend([np.nanmin(np.linalg.norm(X_val[j, :, state_idx].T - target_eepts[j], axis=1)[-10:]) \
                                    for j in xrange(X_val.shape[0])])
            else:
                # Sample on training conditions.
                X_train = self.sample(agent, i, conditions, N=1).get_X()
                # min distance of the last 10 timesteps
                train_dists.extend([np.nanmin(np.linalg.norm(X_train[j, :, state_idx].T - target_eepts[j], axis=1)[-10:]) \
                                    for j in xrange(X_train.shape[0])])

        # import pdb; pdb.set_trace()
        train_success_rate_msg =  "Training success rate is %.5f" % (np.array(train_dists) <= success_thresh).mean()
        val_success_rate_msg = "Validation success rate is %.5f" % (np.array(val_dists) <= success_thresh).mean()
        print train_success_rate_msg
        print val_success_rate_msg
        with open(self._hyperparams['log_filename'], 'a') as f:
            f.write(self._hyperparams['save_dir'] + ':\n')
            f.write(train_success_rate_msg + '\n')
            f.write(val_success_rate_msg + '\n')
