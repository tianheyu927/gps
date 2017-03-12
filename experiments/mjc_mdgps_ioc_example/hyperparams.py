""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY, evall1l2term
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.lin_gauss_init import init_demo, init_lqr
from gps.utility.demo_utils import generate_pos_body_offset, generate_x0, generate_pos_idx
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_mdgps_ioc_example/'
LG_DEMO_DIR = BASE_DIR + '/../experiments/mjc_peg_example/'
CONDITIONS = 1
DEMO_CONDITIONS = 1
# pos_body_offset = [np.array([-0.05, -0.05, -0.05]), np.array([-0.05, 0.05, 0.05]),
#                         np.array([-0.05, -0.05, 0.05]), np.array([0.0,0.0,0.0]),
#                         np.array([-0.05,0.05,-0.05]), np.array([0.05,0.05,-0.05]),
#                         np.array([0.05,-0.05,-0.05]),
#                         np.array([0.05, -0.05, 0.05]), np.array([0.05, 0.05, 0.05])]

# demo_pos_body_offset = [np.array([-0.05, -0.05, -0.05]), np.array([-0.05, 0.05, 0.05]),
#                         np.array([-0.05, -0.05, 0.05]), np.array([0.0,0.0,0.0]),
#                         np.array([-0.05,0.05,-0.05]), np.array([0.05,0.05,-0.05]),
#                         np.array([0.05,-0.05,-0.05]),
#                         np.array([0.05, -0.05, 0.05]), np.array([0.05, 0.05, 0.05])]
pos_body_offset = [np.array([-0.05, -0.05, -0.05])]
demo_pos_body_offset = [np.array([-0.05, -0.05, -0.05])]


SEED = 0
NUM_DEMOS = 20

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    # 'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_06.pkl',
    'demo_exp_dir': LG_DEMO_DIR,
    # 'demo_controller_file': [DEMO_POLICY_DIR[i] + 'algorithm_itr_' + DEMO_POLICY_INDEX[i] + '.pkl' for i in xrange(3)],
    # 'demo_controller_file': LG_DEMO_DIR + 'data_files_9/algorithm_itr_11.pkl',
    'demo_controller_file': LG_DEMO_DIR + 'data_files/algorithm_itr_11.pkl',
    # 'data_files_dir': EXP_DIR + 'data_files_9_LG_local_cost_demo_%d_%d/' % (NUM_DEMOS, SEED),
    'data_files_dir': EXP_DIR + 'data_files_LG_demo_%d_%d/' % (NUM_DEMOS, SEED),
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    # 'LG_demo_file': os.path.join(EXP_DIR, 'data_files_9_LG_local_cost_demo_%d_%d/' % (NUM_DEMOS, SEED), 'demos_LG.pkl'),
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files_LG_demo_%d_%d/' % (NUM_DEMOS, SEED), 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files_demo%d_%d' % (NUM_DEMOS, SEED), 'demos_NN.pkl'),
    'conditions': CONDITIONS,
    'nn_demo': False, # Use neural network demonstrations. For experiment only
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'randomly_sample_bodypos': False,
    'randomly_sample_x0': False,
    'sampling_range_bodypos': [np.array([-0.15,-0.15, -0.15]), np.array([0.15, 0.15, 0.15])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[[None, None, None, None]],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': pos_body_offset,
    # [np.array([-0.1, -0.1, -0.1]), np.array([-0.1, -0.1, 0.1]), np.array([-0.1, 0.1, -0.1]),
    #             np.array([-0.1, 0.1, 0.1]), np.array([0, 0, 0]), np.array([0.1, -0.1, -0.1]),
    #             np.array([0.1, -0.1, 0.1]), np.array([0.1, 0.1, -0.1]), np.array([0.1, 0.1, 0.1])],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
}

demo_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'exp_name': 'peg',
    # 'x0': generate_x0(np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
    #                   np.zeros(7)]), DEMO_CONDITIONS),
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': DEMO_CONDITIONS,
    'pos_body_idx': np.array([1]),
    # 'pos_body_idx': generate_pos_idx(DEMO_CONDITIONS),
    # 'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
    #                     np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    # 'pos_body_offset': generate_pos_body_offset(DEMO_CONDITIONS),
    'pos_body_offset': demo_pos_body_offset,
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'peg_height': 0.1,
    'success_upper_bound': 0.09,
    'failure_lower_bound': 0.15,
}


algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'ioc' : 'ICML',
    # 'ioc_maxent_iter': 30,
    'iterations': 25, #20
    'kl_step': 0.5,
    'min_step_mult': 0.4,
    'max_step_mult': 4.0, #4.0
    # 'min_step_mult': 1.0,
    # 'max_step_mult': 1.0,
    'policy_sample_mode': 'replace',
    'max_ent_traj': 1.0,
    'demo_M': common['conditions'], #1
    'demo_var_mult': 1.0,
    # 'demo_cond': 15,
    'num_demos': NUM_DEMOS,
    'synthetic_cost_samples': 0,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'global_cost': False,
    'num_costs': common['conditions'],
    'sample_on_policy': True,
    'success_upper_bound': 0.10,
}

# Use for nn demos
# algorithm['init_traj_distr'] = {
#     # 'type': init_lqr,
#     'type': init_demo,
#     'init_gains':  1.0 / PR2_GAINS,
#     'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#     'init_var': 5.0,
#     'stiffness': 1.0,
#     'stiffness_vel': 0.5,
#     'final_weight': 50.0,
#     'dt': agent['dt'],
#     'T': agent['T'],
# }

# Use for LG demos.
algorithm['init_traj_distr'] = {
    'type': init_demo,
    # 'type': init_lqr,
    'init_gains':  0.2 / PR2_GAINS, #0.2
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    # 'init_var': 1.0,
    'init_var': 1.5, #1.5
    'stiffness': 0.5, #0.5
    'stiffness_vel': 0.5,
    'final_weight': 10.0, #10.0
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS, #1e-3
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([2, 2, 1, 2, 2, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
}

# Create second cost function for last step only.
final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost['target_end_effector'],
    'wp': fk_cost['wp'],
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
    'evalnorm': evall1l2term,
}

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost, final_cost],
    'weights': [1.0, 1.0, 1.0],
}

algorithm['fk_cost'] = fk_cost

# algorithm['cost'] = {
#     'type': CostIOCNN,
#     'wu': 1000*1e-3 / PR2_GAINS,
#     'T': 100,
#     'dO': 26,
#     'learn_wu': False,
#     'iterations': 5000,
# }

algorithm['cost'] = [{
    'type': CostIOCTF,
    'wu': 1000.0 / PR2_GAINS, # for nn, this is 200
    # 'wu' : 0.0,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data': [],
        'sensor_dims': SENSOR_DIMS,
    },
    'T': agent['T'],
    'dO': 26,
    'iterations': 5000, # TODO - do we need 5k?
    'demo_batch_size': 5, #5
    'sample_batch_size': 5, #5
    'num_hidden': 3, #3
    'dim_hidden': 60,
    'ioc_loss': algorithm['ioc'],
    'mono_reg_weight': 1000.0, #before normalizing, this is 1000
    'smooth_reg_weight': 1.0,
    'batch_norm': False,
    'decay': 0.99,
    'approximate_lxx': False,
    'idx': i,
    'random_seed': SEED, #i
    'global_random_seed': SEED,
    'data_files_dir': common['data_files_dir'],
    'summary_dir': common['data_files_dir'] + 'cost_summary_%d/' % i,
} for i in xrange(algorithm['num_costs'])]

for i in xrange(algorithm['num_costs']):
    if not os.path.exists(algorithm['cost'][i]['summary_dir']):
        os.makedirs(algorithm['cost'][i]['summary_dir'])

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data': [],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'iterations': 1000,  # was 100
    'batch_norm': False,
    'decay': 0.99,
    'weights_file_prefix': common['data_files_dir'] + 'policy',
    'random_seed': SEED,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'record_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'test_gif_dir': os.path.join(common['data_files_dir'], 'test_gifs'),
        'gifs_per_condition': 1,
    },
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': SEED,
}

common['info'] = generate_experiment_info(config)
