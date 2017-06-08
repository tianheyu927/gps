from __future__ import division

from datetime import datetime
import os.path
import numpy as np
from scipy.spatial.distance import cdist
import copy
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, colored_reacher, COLOR_MAP_CONT
from gps.algorithm.behavior_cloning import BehaviorCloning
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.policy_opt.policy_cloning_lstm import PolicyCloningLSTM
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
DATA_DIR = BASE_DIR + '/../data/reacher_color_blocks_larger_box_more_1000_images_no_overlap' #reacher_color_blocks

#CONDITIONS = 1
TRAIN_CONDITIONS = 8
N_VAL = 100
np.random.seed(49)
DEMO_CONDITIONS = 10 #10 #6 #12
COLOR_CONDITIONS = 999#511 #100 #80
TEST_CONDITIONS = 0
TOTAL_CONDITIONS = TRAIN_CONDITIONS+TEST_CONDITIONS
N_CUBES = 3
CUBE_SIZE = 0.03

# Validation colors and training colors
VAL_COLORS = np.random.choice(np.arange(COLOR_CONDITIONS), size=N_VAL, replace=False)
TRAIN_COLORS = np.arange(COLOR_CONDITIONS)[~VAL_COLORS]
VAL_TRIALS = 50
TRAIN_TRIALS = 500
COLOR_TRIALS = (TRAIN_TRIALS + VAL_TRIALS) * N_CUBES

demo_pos_body_offset = {i: [] for i in xrange(COLOR_TRIALS)}
distractor_pos = {i: [] for i in xrange(COLOR_TRIALS)}
distractor_color_idx = {i: [] for i in xrange(COLOR_TRIALS)}
target_color = {i:None for i in xrange(COLOR_TRIALS)}
# cube_pos = {i: [] for i in xrange(COLOR_CONDITIONS)}
# cube_color = {i: [] for i in xrange(COLOR_CONDITIONS)}

# for i in xrange(COLOR_CONDITIONS):
#     for _ in range(DEMO_CONDITIONS):
#         demo_pos_body_offset[i].append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))
#     available_colors = range(COLOR_CONDITIONS)
#     available_colors.remove(i)
#     for j in xrange(N_CUBES-1):
#         distractor_pos[i].append(np.random.rand(3))
#         distractor_color[i].append(np.random.choice(available_colors))
# for i in xrange(COLOR_CONDITIONS):
#     sampled_colors = np.random.choice(range(COLOR_CONDITIONS), size=N_CUBES, replace=False)
#     for k in xrange(N_CUBES):
#         target_color[i*N_CUBES + k] = sampled_colors[k]
#         distractor_color_idx[i*N_CUBES + k] = sampled_colors[np.arange(N_CUBES) != k]
#     for j in xrange(DEMO_CONDITIONS):
#         cube_pos = np.random.rand(N_CUBES, 2)
#         for k in xrange(N_CUBES):
#             body_offset = np.zeros((N_CUBES, 3))
#             body_offset[0] = np.array([0.4*cube_pos[k, 0]-0.3, 0.4*cube_pos[k, 1]-0.1 ,0])
#             body_offset[1:, 0] = 0.4*cube_pos[np.arange(N_CUBES) != k, 0]-0.3
#             body_offset[1:, 1] = 0.4*cube_pos[np.arange(N_CUBES) != k, 1]-0.1
#             demo_pos_body_offset[i*N_CUBES + k].append(body_offset)

# TODO: make sure objects not overlapping by checking pairwise distance
for i in xrange(TRAIN_TRIALS):
    sampled_colors = np.random.choice(TRAIN_COLORS, size=N_CUBES, replace=False)
    for k in xrange(N_CUBES):
        target_color[i*N_CUBES + k] = sampled_colors[k]
        distractor_color_idx[i*N_CUBES + k] = sampled_colors[np.arange(N_CUBES) != k]
    for j in xrange(DEMO_CONDITIONS):
        cube_pos = np.random.rand(N_CUBES, 2)
        pair_dist = cdist(cube_pos, cube_pos)
        pair_dist = pair_dist[pair_dist != 0]
        while np.any(pair_dist < CUBE_SIZE*5.0):
            cube_pos = np.random.rand(N_CUBES, 2)
            pair_dist = cdist(cube_pos, cube_pos)
            pair_dist = pair_dist[pair_dist != 0]
        for k in xrange(N_CUBES):
            body_offset = np.zeros((N_CUBES, 3))
            body_offset[0] = np.array([0.4*cube_pos[k, 0]-0.3, 0.4*cube_pos[k, 1]-0.1 ,0])
            body_offset[1:, 0] = 0.4*cube_pos[np.arange(N_CUBES) != k, 0]-0.3
            body_offset[1:, 1] = 0.4*cube_pos[np.arange(N_CUBES) != k, 1]-0.1
            demo_pos_body_offset[i*N_CUBES + k].append(body_offset)
# Let validation color be the last 10*6=60 colors
for i in xrange(TRAIN_TRIALS, TRAIN_TRIALS+VAL_TRIALS):
    sampled_colors = np.random.choice(VAL_COLORS, size=N_CUBES, replace=False)
    for k in xrange(N_CUBES):
        target_color[i*N_CUBES + k] = sampled_colors[k]
        distractor_color_idx[i*N_CUBES + k] = sampled_colors[np.arange(N_CUBES) != k]
    for j in xrange(DEMO_CONDITIONS):
        cube_pos = np.random.rand(N_CUBES, 2)
        pair_dist = cdist(cube_pos, cube_pos)
        pair_dist = pair_dist[pair_dist != 0]
        while np.any(pair_dist < CUBE_SIZE*5.0):
            cube_pos = np.random.rand(N_CUBES, 2)
            pair_dist = cdist(cube_pos, cube_pos)
            pair_dist = pair_dist[pair_dist != 0]
        for k in xrange(N_CUBES):
            body_offset = np.zeros((N_CUBES, 3))
            body_offset[0] = np.array([0.4*cube_pos[k, 0]-0.3, 0.4*cube_pos[k, 1]-0.1 ,0])
            body_offset[1:, 0] = 0.4*cube_pos[np.arange(N_CUBES) != k, 0]-0.3
            body_offset[1:, 1] = 0.4*cube_pos[np.arange(N_CUBES) != k, 1]-0.1
            demo_pos_body_offset[i*N_CUBES + k].append(body_offset)

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
    'data_files_dir': EXP_DIR + 'data_files_demo%d_%d/' % (NUM_DEMOS, SEED),
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
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=distractor_color_idx[j]),
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
    'color_idx': np.arange(5, 4+N_CUBES),
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
} for j in xrange(COLOR_TRIALS)]

demo_agent = [{
    'type': AgentMuJoCo,
    'models': colored_reacher(ncubes=N_CUBES, target_color=target_color[j], cube_size=CUBE_SIZE, distractor_pos=np.zeros((N_CUBES-1, 3)), distractor_color=distractor_color_idx[j]),
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
    'distractor_color': distractor_color_idx[j],
    'color_idx': np.arange(5, 4+N_CUBES),
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
    'save_images': True,
} for j in xrange(COLOR_TRIALS)]

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
    'type': PolicyCloningLSTM,
    'network_params': {
        'num_filters': [40, 40, 40], #20, 20, 20
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
        'n_layers': 4,
        'layer_size': 200,
        'lstm_size': 2048,
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
    'iterations': 100000, #about 20 epochs
    'restore_iter': 0,
    'random_seed': SEED,
    'n_val': VAL_TRIALS*N_CUBES, #50
    'weight_decay': 0.005, #0.005,
    'use_clip': False,
    'clip_min': -10.0,
    'clip_max': 10.0,
    'meta_batch_size': 5,
    'update_batch_size': 1, # batch size for each task, used to be 1
    'eval_batch_size': 1,
    # 'log_dir': '/tmp/data/maml_bc/4_layer_100_dim_40_3x3_filters_1_step_1e_4_mbs_1_ubs_2_update3_hints',
    'log_dir': '/tmp/data/lstm_bc_1000/4_layer_200_dim_40_3x3_filters_2048_lstm_size_mbs_5_ubs_1_ebs_1_10_pos_images_no_dropout_300_trials',
    # 'save_dir': '/tmp/data/maml_bc_model_ln_4_100_40_3x3_filters_fixed_1e-4_cnn_normalized_batch1_noise_mbs_1_ubs_2_update3_hints',
    'save_dir': '/home/kevin/gps/data/models/lstm_bc_1000_model_ln_4_layers_200_dim_40_3x3_filters_2048_lstm_size_mbs_5_ubs_1_ebs_1_10_pos_images_no_dropout_300_trials',
    'plot_dir': common['data_files_dir'],
    'demo_gif_dir': os.path.join(DATA_DIR, 'demo_gifs/'),
    'use_vision': True,
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
