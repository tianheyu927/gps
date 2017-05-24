from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import copy
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, colored_reacher, COLOR_MAP_CONT
from gps.algorithm.behavior_cloning import BehaviorCloning
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.policy_opt.policy_cloning_maml import PolicyCloningMAML
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network, multi_modal_network_fp

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT
from gps.gui.config import generate_experiment_info

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS_NO_TARGET: 3,
    END_EFFECTOR_POINT_VELOCITIES_NO_TARGET: 3,
    ACTION: 2,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
    # IMAGE_FEAT: 30,  # affected by num_filters set below.
}


BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher_mdgps/'
DATA_DIR = BASE_DIR + '/../data/reacher_color_blocks_500_positions_3_colors' #reacher_color_blocks

#CONDITIONS = 1
TRAIN_CONDITIONS = 8
N_VAL = 3 #100
np.random.seed(49) #49
TEST_CONDITIONS = 0
TOTAL_CONDITIONS = TRAIN_CONDITIONS+TEST_CONDITIONS
COLOR_CONDITIONS = 3#999
DEMO_CONDITIONS = 500 #64 #160 #32
VAL_CONDITIONS = 50
N_CUBES = 3
CUBE_SIZE = 0.03
target_color = np.array(['red', 'blue', 'green'])#, 'white', 'yellow', 'purple', 'cyan'])
# demo_pos_body_offset = [[np.array([[-0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]]),
#                         np.array([[-0.1, 0.0, 0.0], [-0.1, 0.2, 0.0]])],
#                         [np.array([[-0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]]),
#                         np.array([[-0.1, 0.0, 0.0], [-0.1, 0.2, 0.0]])]]

# demo_pos_body_offset = [np.array([[-0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]]),
#                         np.array([[-0.1, 0.0, 0.0], [-0.1, 0.2, 0.0]])]

# demo_pos_body_offset = [np.array([[-0.1, 0.2, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
#                         np.array([[-0.1, 0.2, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
#                         np.array([[-0.2, 0.0, 0.0], [-0.1, 0.2, 0.0], [0.0, 0.0, 0.0]]),
#                         np.array([[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, 0.2, 0.0]]),
#                         np.array([[0.0, 0.0, 0.0], [-0.1, 0.2, 0.0], [-0.2, 0.0, 0.0]]),
#                         np.array([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.1, 0.2, 0.0]])]
COLOR_TRIALS = COLOR_CONDITIONS + N_VAL
demo_pos_body_offset = {i: [] for i in xrange(COLOR_TRIALS)}
for j in xrange(DEMO_CONDITIONS):
    cube_pos = np.random.rand(N_CUBES, 2)
    for k in xrange(N_CUBES):
        body_offset = np.zeros((N_CUBES, 3))
        body_offset[0] = np.array([0.4*cube_pos[k, 0]-0.3, 0.4*cube_pos[k, 1]-0.1 ,0])
        body_offset[1:, 0] = 0.4*cube_pos[np.arange(N_CUBES) != k, 0]-0.3
        body_offset[1:, 1] = 0.4*cube_pos[np.arange(N_CUBES) != k, 1]-0.1
        demo_pos_body_offset[k].append(body_offset)
for j in xrange(VAL_CONDITIONS):
    cube_pos = np.random.rand(N_CUBES, 2)
    for k in xrange(N_CUBES, 2*N_CUBES):
        body_offset = np.zeros((N_CUBES, 3))
        body_offset[0] = np.array([0.4*cube_pos[k-N_CUBES, 0]-0.3, 0.4*cube_pos[k-N_CUBES, 1]-0.1 ,0])
        body_offset[1:, 0] = 0.4*cube_pos[np.arange(N_CUBES) != k-N_CUBES, 0]-0.3
        body_offset[1:, 1] = 0.4*cube_pos[np.arange(N_CUBES) != k-N_CUBES, 1]-0.1
        demo_pos_body_offset[k].append(body_offset)
        
# Validation colors and training colors
# VAL_COLORS = np.random.choice(np.arange(COLOR_CONDITIONS), size=N_VAL, replace=False)
# TRAIN_COLORS = np.arange(COLOR_CONDITIONS)[~VAL_COLORS]

# distractor_pos = {i: [] for i in xrange(COLOR_TRIALS)}
# distractor_color_idx = {i: [] for i in xrange(COLOR_TRIALS)}
# target_color = {i:None for i in xrange(COLOR_TRIALS)}

# for i in xrange(COLOR_CONDITIONS - N_VAL):
#     sampled_colors = np.random.choice(TRAIN_COLORS, size=N_CUBES, replace=False)
#     for k in xrange(N_CUBES):
#         target_color[i*N_CUBES + k] = sampled_colors[k]
#         distractor_color_idx[i*N_CUBES + k] = sampled_colors[np.arange(N_CUBES) != k]

# # Let validation color be the last 10*6=60 colors
# for i in xrange(COLOR_CONDITIONS - N_VAL, COLOR_CONDITIONS):
#     sampled_colors = np.random.choice(VAL_COLORS, size=N_CUBES, replace=False)
#     for k in xrange(N_CUBES):
#         target_color[i*N_CUBES + k] = sampled_colors[k]
#         distractor_color_idx[i*N_CUBES + k] = sampled_colors[np.arange(N_CUBES) != k]

pos_body_offset = [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.2, 0.0]),
                   np.array([-0.1, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0]),
                   np.array([-0.2, 0.1, 0.0]), np.array([-0.2, 0.0, 0.0]),
                   np.array([-0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]

SEED = 0 #0
NUM_DEMOS = 1

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files_full_color_demo%d_%d/' % (NUM_DEMOS, SEED),
    # 'data_files_dir': EXP_DIR + 'data_files_LG_demo5_%d/' % SEED,
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': DEMO_DIR + 'data_files_28/algorithm_itr_29.pkl', #28 conditions
    'nn_demo': True, # Use neural network demonstrations. For experiment only
    'NN_demo_file': [os.path.join(DATA_DIR, 'demos_%d.pkl' % i) for i in xrange(COLOR_TRIALS)],
    'LG_demo_file': [os.path.join(DATA_DIR, 'demos_%d.pkl' % i) for i in xrange(COLOR_TRIALS)],
    'conditions': TOTAL_CONDITIONS,
    'train_conditions': range(TRAIN_CONDITIONS),
    'test_conditions': range(TRAIN_CONDITIONS, TOTAL_CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': colored_reacher(target_color='red'),
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'modify_cost_on_sample': False,
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'record_reward': True,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ pos_body_offset[i], np.array([0., 0., 0.])])
        for i in range(TOTAL_CONDITIONS)],
}

pol_agent = [{
    'type': AgentMuJoCo,
    # for testing
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=target_color[target_color != target_color[j]]),
    'exp_name': 'reacher',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'modify_cost_on_sample': False,
    'pos_body_offset': demo_pos_body_offset[j],
    'pos_body_idx': np.arange(4, 4+N_CUBES), #np.array([4]),
    # 'distractor_color': distractor_color[j],
    # 'color_idx': np.arange(5, 4+N_CUBES),
    'conditions': DEMO_CONDITIONS,
    'T': 50,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'record_reward': True,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j][i][0], np.array([0., 0., 0.])])
        for i in range(DEMO_CONDITIONS)],
    'filter_demos': {
        'type': 'last',
        'state_idx': range(4, 7),
        'target_ee_idx': range(7, 10),
        'target': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j][i][0], np.array([0., 0., 0.])])
                            for i in xrange(DEMO_CONDITIONS)],
        'success_upper_bound': 0.05,
    },
} for j in xrange(COLOR_CONDITIONS)]

pol_agent.extend([{
    'type': AgentMuJoCo,
    # for testing
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=target_color[target_color != target_color[j]]),
    'exp_name': 'reacher',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'modify_cost_on_sample': False,
    'pos_body_offset': demo_pos_body_offset[j+COLOR_CONDITIONS],
    'pos_body_idx': np.arange(4, 4+N_CUBES), #np.array([4]),
    # 'distractor_color': distractor_color[j],
    # 'color_idx': np.arange(5, 4+N_CUBES),
    'conditions': VAL_CONDITIONS,
    'T': 50,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'record_reward': True,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j+COLOR_CONDITIONS][i][0], np.array([0., 0., 0.])])
        for i in range(VAL_CONDITIONS)],
    'filter_demos': {
        'type': 'last',
        'state_idx': range(4, 7),
        'target_ee_idx': range(7, 10),
        'target': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j+COLOR_CONDITIONS][i][0], np.array([0., 0., 0.])])
                            for i in xrange(VAL_CONDITIONS)],
        'success_upper_bound': 0.05,
    },
} for j in xrange(N_VAL)])

demo_agent = [{
    'type': AgentMuJoCo,
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=target_color[target_color != target_color[j]]),
    'exp_name': 'reacher',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'modify_cost_on_sample': False,
    'pos_body_offset': demo_pos_body_offset[j],
    'pos_body_idx': np.arange(4, 4+N_CUBES), #np.array([4]),
    # 'distractor_color': distractor_color_idx[j],
    # 'color_idx': np.arange(5, 4+N_CUBES),
    'conditions': DEMO_CONDITIONS,
    'T': 50,
    # 'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
    #                   END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'record_reward': True,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j][i][0], np.array([0., 0., 0.])])
        for i in range(DEMO_CONDITIONS)],
    'filter_demos': {
        'type': 'last',
        'state_idx': range(4, 7),
        'target_ee_idx': range(7, 10),
        'target': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j][i][0], np.array([0., 0., 0.])])
                            for i in xrange(DEMO_CONDITIONS)],
        'success_upper_bound': 0.03,
    },
} for j in xrange(COLOR_CONDITIONS)]

demo_agent.extend([{
    'type': AgentMuJoCo,
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=target_color[target_color != target_color[j]]),
    'exp_name': 'reacher',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'randomly_sample_bodypos': False,
    'sampling_range_bodypos': [np.array([-0.3,-0.1, 0.0]), np.array([0.1, 0.3, 0.0])], # Format is [lower_lim, upper_lim]
    'prohibited_ranges_bodypos':[ [None, None, None, None] ],
    'modify_cost_on_sample': False,
    'pos_body_offset': demo_pos_body_offset[j+COLOR_CONDITIONS],
    'pos_body_idx': np.arange(4, 4+N_CUBES), #np.array([4]),
    # 'distractor_color': distractor_color_idx[j],
    # 'color_idx': np.arange(5, 4+N_CUBES),
    'conditions': VAL_CONDITIONS,
    'T': 50,
    # 'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
    #                   END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, \
                    END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
    'record_reward': True,
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j+COLOR_CONDITIONS][i][0], np.array([0., 0., 0.])])
        for i in range(VAL_CONDITIONS)],
    'filter_demos': {
        'type': 'last',
        'state_idx': range(4, 7),
        'target_ee_idx': range(7, 10),
        'target': [np.concatenate([np.array([.1, -.1, .01])+ demo_pos_body_offset[j+COLOR_CONDITIONS][i][0], np.array([0., 0., 0.])])
                            for i in xrange(VAL_CONDITIONS)],
        'success_upper_bound': 0.03,
    },
} for j in xrange(N_VAL)])

algorithm = {
    'type': BehaviorCloning,
    'bc': True,
    'sample_on_policy': True,
    'conditions': common['conditions'],
    'iterations': 1, # must be 1
    'demo_var_mult': 1.0,
    'demo_M': 1,
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


algorithm['policy_opt'] = {
    'type': PolicyCloningMAML,
    'network_params': {
        'num_filters': [30, 30, 30], #20, 20, 20
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
        'n_layers': 4,
        'layer_size': 50,
        'bc': True,
    },
    'use_gpu': 1,
    'T': agent['T'],
    'demo_file': common['NN_demo_file'] if common['nn_demo'] else common['LG_demo_file'],
    'agent': pol_agent,
    'copy_param_scope': 'model',
    'norm_type': 'layer_norm', # True
    'is_dilated': False,
    'color_hints': False,
    'use_dropout': False,
    'keep_prob': 0.9,
    'decay': 0.9,
    'stop_grad': False,
    'iterations': 40000, #40000 #about 20 epochs
    'restore_iter': 0,
    'random_seed': SEED,
    'n_val': N_VAL, #int(N_VAL*N_CUBES/2), #50
    'step_size': 1e-4, #1e-5 # step size of gradient step
    'num_updates': 3, # take one gradient step
    'meta_batch_size': 3, #10, # number of tasks during training
    'weight_decay': 0.005, #0.005,
    'update_batch_size': 1, # batch size for each task, used to be 1
    'log_dir': '/tmp/data/maml_bc_500_pos_3_colors/4_layer_50_dim_30_3x3_filters_1_step_1e_4_mbs_3_ubs_1_update3_larger_box',
    # 'log_dir': '/tmp/data/maml_bc_three_pos/5_layer_60_dim_30_3x3_filters_1_step_1e_3_mbs_5_ubs_1_update1_larger_box',
    'save_dir': '/tmp/data/maml_bc_500_pos_3_colors_4_layers_50_dim_30_3x3_filters_1e-4_mbs_3_ubs_1_update3_no_hints_larger_box',
    # 'save_dir': '/tmp/data/maml_bc_three_pos_5_layers_60_dim_30_3x3_filters_1e-3_mbs_5_ubs_1_update1_no_hints_larger_box',
    'plot_dir': common['data_files_dir'],
    'uses_vision': True,
    'weights_file_prefix': EXP_DIR + 'policy',
    'record_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs/'),
        'test_gif_dir': os.path.join(common['data_files_dir'], 'test_gifs/'),
        'gifs_per_condition': 1,
    },
}

config = {
    'iterations': algorithm['iterations'],
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'record_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'test_gif_dir': os.path.join(common['data_files_dir'], 'test_gifs'),
        'demo_gif_dir': os.path.join(DATA_DIR, 'demo_gifs/'),
        'gifs_per_condition': 1,
    },
    'agent': agent,
    'demo_agent': demo_agent,
    'pol_agent': pol_agent,
    'gui_on': False,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': SEED,
}

common['info'] = generate_experiment_info(config)
