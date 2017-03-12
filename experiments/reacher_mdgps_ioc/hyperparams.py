from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_gym import CostGym
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_demo
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
# DEMO_DIR = BASE_DIR + '/../experiments/reacher_mdgps/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher/'

#CONDITIONS = 1
TRAIN_CONDITIONS = 8 #8

np.random.seed(47)
DEMO_CONDITIONS = 8 #8 #20
TEST_CONDITIONS = 0
TOTAL_CONDITIONS = TRAIN_CONDITIONS+TEST_CONDITIONS

demo_pos_body_offset = []
# for _ in range(DEMO_CONDITIONS):
#     demo_pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

pos_body_offset = []
# for _ in range(TOTAL_CONDITIONS):
#     pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

# pos_body_offset.append(np.array([-0.2, 0.1, 0.0]))
# #pos_body_offset.append(np.array([0.05, 0.2, 0.0]))
# demo_pos_body_offset.append(np.array([-0.2, 0.1, 0.0]))
pos_body_offset = [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.2, 0.0]),
                    np.array([-0.1, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0]),
                    np.array([-0.2, 0.1, 0.0]), np.array([-0.2, 0.0, 0.0]),
                    np.array([-0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
demo_pos_body_offset = [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.2, 0.0]),
                    np.array([-0.1, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0]),
                    np.array([-0.2, 0.1, 0.0]), np.array([-0.2, 0.0, 0.0]),
                    np.array([-0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
test_pos_body_offset = [np.array([-0.05, 0.18, 0.0]), np.array([-0.15, 0.18, 0.0]),
                    np.array([-0.15, 0.02, 0.0]), np.array([-0.05, 0.02, 0.0]),
                    np.array([-0.02, 0.15, 0.0]), np.array([-0.18, 0.15, 0.0]),
                    np.array([-0.18, 0.05, 0.0]), np.array([-0.02, 0.05, 0.0]),
                    np.array([0.02, 0.15, 0.0]), np.array([0.02, 0.22, 0.0]),
                    np.array([-0.05, 0.22, 0.0]), np.array([-0.15, 0.22, 0.0]),
                    np.array([-0.22, 0.22, 0.0]), np.array([-0.22, 0.15, 0.0]),
                    np.array([-0.22, 0.05, 0.0]), np.array([-0.22, -0.02, 0.0]),
                    np.array([-0.15, -0.02, 0.0]), np.array([-0.05, -0.02, 0.0]),
                    np.array([0.02, -0.02, 0.0]), np.array([0.02, 0.05, 0.0])]

SEED = 0
NUM_DEMOS = 20
DEMO_CLUSTERS = 2

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    # 'data_files_dir': EXP_DIR + 'data_files_8_demo1_%d/' % SEED,
    # 'data_files_dir': EXP_DIR + 'data_files_LG_demo%d_cost_%d/' % (NUM_DEMOS, SEED),
    'data_files_dir': EXP_DIR + 'data_files_8_LG_demo%d_local_quad_cost_cluster_%d/' % (NUM_DEMOS, SEED),
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    # 'demo_controller_file': DEMO_DIR + 'data_files_8/algorithm_itr_09.pkl',
    'demo_controller_file': DEMO_DIR + 'data_files_8/algorithm_itr_14.pkl', #11 for 1 condition
    'nn_demo': False, # Use neural network demonstrations. For experiment only
    # 'LG_demo_file': os.path.join(EXP_DIR, 'data_files_LG_demo%d_cost_%d' % (NUM_DEMOS, SEED), 'demos_LG.pkl'),
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files_8_LG_demo%d_local_quad_cost_cluster_%d' % (NUM_DEMOS, SEED), 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files_demo%d_%d' % (NUM_DEMOS, SEED), 'demos_NN.pkl'),
    'conditions': TOTAL_CONDITIONS,
    'train_conditions': range(TRAIN_CONDITIONS),
    'test_conditions': range(TRAIN_CONDITIONS, TOTAL_CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(TOTAL_CONDITIONS)],
    'success_upper_bound': 0.05,
    'render': True,
}

unlabeled_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'pos_body_offset': test_pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': len(test_pos_body_offset),
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ test_pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(len(test_pos_body_offset))],
    'success_upper_bound': 0.05,
    'render': True,
} 

demo_agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'exp_name': 'reacher',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': demo_pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': DEMO_CONDITIONS,
    'T': agent['T'],
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
            END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(DEMO_CONDITIONS)],
    'success_upper_bound': 0.05,
    'render': True,
}


algorithm = {
    'type': AlgorithmMDGPS,
    'sample_on_policy': True,
    'ioc' : 'ICML',  # IOC STUFF HERE
    'max_ent_traj': 1.0,
    'num_demos': NUM_DEMOS,
    'synthetic_cost_samples': 0,
    'global_cost': False, #True
    'num_costs': common['conditions'], #1
    'demo_M': common['conditions'],
    'demo_var_mult': 1.0,
    'demo_clusters': DEMO_CLUSTERS,
    'conditions': common['conditions'],  # NON IOC STUFF HERE
    'iterations': 20, #20
    'ioc_maxent_iter': 20,
    'kl_step': 1.0,
    'min_step_mult': 0.2,
    'max_step_mult': 2.0,
    'policy_sample_mode': 'replace',
    'agent_pos_body_idx': agent['pos_body_idx'],
    'agent_pos_body_offset': agent['pos_body_offset'],
    'plot_dir': EXP_DIR,
    'agent_x0': agent['x0'],
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
        for i in range(TOTAL_CONDITIONS)],
}


#algorithm = {
#    'type': AlgorithmTrajOpt,
#    'ioc' : 'ICML',
#    'max_ent_traj': 1.0,
#    'conditions': common['conditions'],
#    'kl_step': 0.5,
#    'min_step_mult': 0.05,
#    'max_step_mult': 2.0,
#    'demo_cond': demo_agent['conditions'],
#    'num_demos': 2,
#    'demo_var_mult': 1.0,
#    'synthetic_cost_samples': 100,
#    'iterations': 15,
#    'plot_dir': EXP_DIR,
#    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
#                            for i in xrange(CONDITIONS)],
#}

PR2_GAINS = np.array([1.0, 1.0])
torque_cost_1 = [{
    'type': CostAction,
    'wu': 1.0 / PR2_GAINS,
} for i in range(common['conditions'])]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
} for i in range(common['conditions'])]

algorithm['fk_cost'] = fk_cost_1

algorithm['gt_cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [2.0, 1.0],
}  for i in range(common['conditions'])]

algorithm['cost'] = [{
    'type': CostIOCTF,
    'wu': 2000.0 / PR2_GAINS, # for nn, this is 200
    # 'wu' : 0.0,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_image_data': [],
        'sensor_dims': SENSOR_DIMS,
    },
    'T': agent['T'],
    'dO': 16,
    'iterations': 3000, # TODO - do we need 5k?
    'demo_batch_size': 5, #5
    'sample_batch_size': 5, #5
    'num_hidden': 0, #3
    'dim_hidden': 40,
    'ioc_loss': algorithm['ioc'],
    'mono_reg_weight': 1000.0, # before normalizing, this is 100
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

# algorithm['cost'] = {
#     'type': CostIOCTF,
#     'wu': 2000.0 / PR2_GAINS, # for nn, this is 200
#     # 'wu' : 0.0,
#     'network_params': {
#         'obs_include': agent['obs_include'],
#         'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
#         'obs_image_data': [],
#         'sensor_dims': SENSOR_DIMS,
#     },
#     'T': agent['T'],
#     'dO': 16,
#     'iterations': 5000, # TODO - do we need 5k?
#     'demo_batch_size': 5,
#     'sample_batch_size': 5,
#     'num_hidden': 5,
#     'dim_hidden': 50,
#     'ioc_loss': algorithm['ioc'],
#     # 'mono_reg_weight': 0.0,
#     'batch_norm': False,
#     'decay': 0.99,
#     'approximate_lxx': False,
#     'random_seed': 0, #SEED
#     'global_random_seed': SEED,
#     'data_files_dir': common['data_files_dir'],
#     'summary_dir': common['data_files_dir'] + 'cost_summary/',
# }

# if not os.path.exists(algorithm['cost']['summary_dir']):
#     os.makedirs(algorithm['cost']['summary_dir'])

#algorithm['init_traj_distr'] = {
#    'type': init_demo,
#    'init_gains':  1.0 / PR2_GAINS,
#    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#    'init_var': 5.0,
#    'stiffness': 1.0,
##    'stiffness_vel': 0.5,
#    'final_weight': 50.0,
#    'dt': agent['dt'],
#    'T': agent['T'],
#}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100.0 * np.ones(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 5.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.5,
    'final_weight': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 30,
        'min_samples_per_cluster': 40,
        'max_samples': 10, #len(common['conditions']),
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'min_eta': 1e-4,
    'max_eta': 1.0
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
    'max_clusters': 40,
    'min_samples_per_cluster': 40,
}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'record_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'test_gif_dir': os.path.join(common['data_files_dir'], 'test_gifs'),
        'gifs_per_condition': 1,
    },
    'agent': agent,
    'unlabeled_agent': unlabeled_agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': SEED,
}

common['info'] = generate_experiment_info(config)
