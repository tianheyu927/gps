from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.behavior_cloning import BehaviorCloning
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.policy_opt.policy_cloning_tf import PolicyCloningTf
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
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
IOC_DIR = BASE_DIR + '/../experiments/reacher_mdgps_ioc/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher/'

#CONDITIONS = 1
TRAIN_CONDITIONS = 8

np.random.seed(47)
DEMO_CONDITIONS = 8 #20
TEST_CONDITIONS = 0
TOTAL_CONDITIONS = TRAIN_CONDITIONS+TEST_CONDITIONS

demo_pos_body_offset = []
# for _ in range(DEMO_CONDITIONS):
#     demo_pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

pos_body_offset = []
# for _ in range(TOTAL_CONDITIONS):
#     pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

# pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
#pos_body_offset.append(np.array([0.05, 0.2, 0.0]))
# demo_pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
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
                    np.array([-0.02, 0.15, 0.0]), np.arra
                    y([-0.18, 0.15, 0.0]),
                    np.array([-0.18, 0.05, 0.0]), np.array([-0.02, 0.05, 0.0]),
                    np.array([0.02, 0.15, 0.0]), np.array([0.02, 0.22, 0.0]),
                    np.array([-0.05, 0.22, 0.0]), np.array([-0.15, 0.22, 0.0]),
                    np.array([-0.22, 0.22, 0.0]), np.array([-0.22, 0.15, 0.0]),
                    np.array([-0.22, 0.05, 0.0]), np.array([-0.22, -0.02, 0.0]),
                    np.array([-0.15, -0.02, 0.0]), np.array([-0.05, -0.02, 0.0]),
                    np.array([0.02, -0.02, 0.0]), np.array([0.02, 0.05, 0.0])]

SEED = 2
NUM_DEMOS = 20

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files_8_LG_demo%d_%d/' % (NUM_DEMOS, SEED),
    # 'data_files_dir': EXP_DIR + 'data_files_LG_demo5_%d/' % SEED,
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': DEMO_DIR + 'data_files_8/algorithm_itr_14.pkl',
    'nn_demo': False, # Use neural network demonstrations. For experiment only
    'LG_demo_file': os.path.join(IOC_DIR, 'data_files_8_LG_demo%d_%d' % (NUM_DEMOS, SEED), 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(IOC_DIR, 'data_files_8_demo20_%d' % SEED, 'demos_NN.pkl'),
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
    'success_upper_bound': 0.01,
    'render': True,
}


algorithm = {
    'type': BehaviorCloning,
    'bc': True,
    'sample_on_policy': True,
    'conditions': common['conditions'],
    'iterations': 1, # must be 1
    'demo_var_mult': 1.0,
    'num_demos': NUM_DEMOS,
    'agent_pos_body_idx': agent['pos_body_idx'],
    'agent_pos_body_offset': agent['pos_body_offset'],
    'plot_dir': EXP_DIR,
    'agent_x0': agent['x0'],
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
        for i in range(TOTAL_CONDITIONS)],
}

PR2_GAINS = np.array([1.0, 1.0])
torque_cost_1 = [{
    'type': CostAction,
    'wu': 1 / PR2_GAINS,
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

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [2.0, 1.0],
}  for i in range(common['conditions'])]


# Cost function
#torque_cost = {
    #'type': CostAction,
    #'wu': np.ones(2),
#}
#

#state_cost = {
    #'type': CostState,
    #'data_types': [END_EFFECTOR_POINTS],
    #'A' : np.c_[np.eye(3), -np.eye(3)],
    #'l1': 1.0,
    #'l2': 0.0,
    #'alpha': 0.0,
    #'evalnorm': evall1l2term,
#}

#algorithm['cost'] = {
    #'type': CostSum,
    #'costs': [torque_cost, state_cost],
    #'weights': [2.0, 1.0],
#}

#algorithm['cost'] = {
#    'type': CostGym,
#}

# algorithm['policy_opt'] = {
#     'type': PolicyOptCaffe,
#     'iterations': 5000,
#     'weights_file_prefix': common['data_files_dir'] + 'policy',
# }

algorithm['policy_opt'] = {
    'type': PolicyCloningTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
        'bc': True,
        'n_layers': 3,
        'dim_hidden': 50,
    },
    'network_model': example_tf_network,
    # 'fc_only_iterations': 5000,
    # 'init_iterations': 1000,
    'batch_norm': False,
    'decay': 0.99,
    'lr': 1e-3,
    'batch_size': 25,
    'iterations': 1000,  # was 100
    'demo_file': common['NN_demo_file'] if common['nn_demo'] else common['LG_demo_file'],
    'agent': demo_agent,
    'weights_file_prefix': common['data_files_dir'] + 'policy',
    'random_seed': SEED,
}


config = {
    'iterations': algorithm['iterations'],
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
