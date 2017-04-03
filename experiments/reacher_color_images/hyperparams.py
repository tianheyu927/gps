from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, colored_reacher
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_gym import CostGym
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
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


np.random.seed(47)
pos_body_offset = []
#pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
#pos_body_offset.append(np.array([0.05, 0.2, 0.0]))
# for _ in range(CONDITIONS):
#     pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

SEED = 0
CONDITIONS = 8
pos_body_offset = [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.2, 0.0]),
                    np.array([-0.1, 0.2, 0.0]), np.array([-0.2, 0.2, 0.0]),
                    np.array([-0.2, 0.1, 0.0]), np.array([-0.2, 0.0, 0.0]),
                    np.array([-0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files_8/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    #'filename': './mjc_models/reacher_img.xml',
    # 'models': [colored_reacher(target_color='red'), colored_reacher(target_color='green'), colored_reacher(target_color='yellow')],
    'models': [colored_reacher(target_color='red') for i in xrange(CONDITIONS)],
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': pos_body_offset,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],  # no IMAGE_FEAT # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'camera_pos': np.array([0., 0., 3., 0., 0., 0.]),
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ pos_body_offset[i], np.array([0., 0., 0.])])
                            for i in xrange(CONDITIONS)],
}


algorithm = {
    'type': AlgorithmMDGPS,
    'max_ent_traj': 0.01,
    'conditions': common['conditions'],
    'iterations': 20,
    'kl_step': 1.0, # TODO was 1.0
    'min_step_mult': 0.5, # TODO was 0.5, maybe try 0.1
    'max_step_mult': 3.0, # TODO was 3.0, maybe try 2.0
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
    'plot_dir': EXP_DIR,
    'agent_pos_body_idx': agent['pos_body_idx'],
    'agent_pos_body_offset': agent['pos_body_offset'],
    'target_end_effector': [np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])])
                            for i in xrange(CONDITIONS)],
}

#algorithm = {
#    'type': AlgorithmMDGPS,
#    'sample_on_policy': True,
#    'conditions': common['conditions'],
#    'train_conditions': common['train_conditions'],
#    'test_conditions': common['test_conditions'],
#    'iterations': 10,
#    'kl_step': 1.0,
#    'min_step_mult': 0.2,
#    'max_step_mult': 2.0,
#    'policy_sample_mode': 'replace',
#}

PR2_GAINS = np.array([1.0, 1.0])

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'num_filters': [15, 15, 15],
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': multi_modal_network_fp,
    'fc_only_iterations': 5000,
    'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'random_seed': SEED,
    'uses_vision': True,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost_1 = [{
    'type': CostAction,
    'wu': 1 / PR2_GAINS,
} for i in range(common['conditions'])]

fk_cost_1 = [{
    'type': CostFK,
    'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
    'wp': np.array([1, 1, 1, 0, 0, 0]),
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 0,
    'evalnorm': evall1l2term,
} for i in range(common['conditions'])]

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [2.0, 1.0],
}  for i in range(common['conditions'])]

algorithm['fk_cost'] = fk_cost_1

# torque_cost_1 = [{
#     'type': CostAction,
#     'wu': 1 / PR2_GAINS,
# } for i in range(common['conditions'])]

# fk_cost_1 = [{
#     'type': CostFK,
#     'target_end_effector': np.concatenate([np.array([.1, -.1, .01])+ agent['pos_body_offset'][i], np.array([0., 0., 0.])]),
#     'wp': np.array([1, 1, 1, 0, 0, 0]),
#     'l1': 0.1,
#     'l2': 10.0,
#     'alpha': 1e-5,
#     'evalnorm': evall1l2term,
# } for i in range(common['conditions'])]

# algorithm['cost'] = [{
#     'type': CostSum,
#     'costs': [torque_cost_1[i], fk_cost_1[i]],
#     'weights': [2.0, 1.0],
#     'evalnorm': evall1l2term,
# }  for i in range(common['conditions'])]


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 30,
        'min_samples_per_cluster': 40,
        'max_samples': 10, #len(common['train_conditions']),
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'min_eta': 1e-4,
    'max_eta': 1.0,
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 40,
    'min_samples_per_cluster': 40,
}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'record_gif': {
        'gif_dir': os.path.join(common['data_files_dir'], 'gifs'),
        'test_gif_dir': os.path.join(common['data_files_dir'], 'test_gifs'),
        'gifs_per_condition': 1,
    },
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': SEED,
}

common['info'] = generate_experiment_info(config)
