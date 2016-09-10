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
from gps.algorithm.cost.cost_ioc_nn import CostIOCNN
from gps.algorithm.cost.cost_fk import CostFK
#from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_gym import CostGym
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, init_demo
from gps.utility.demo_utils import generate_pos_body_offset, generate_x0, generate_pos_idx
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network, multi_modal_network_fp

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET
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
}

PR2_GAINS = np.array([1.0, 1.0])

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
DEMO_DIR = BASE_DIR + '/../experiments/reacher/'

CONDITIONS = 1
DEMO_CONDITIONS = 20

np.random.seed(13)
pos_body_offset = []
pos_body_offset.append(np.array([-0.1, 0.2, 0.0]))
#for _ in range(CONDITIONS):
#    pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1, 0]))

demo_pos_body_offset = []
for _ in range(DEMO_CONDITIONS):
    demo_pos_body_offset.append(np.array([0.4*np.random.rand()-0.3, 0.4*np.random.rand()-0.1 ,0]))

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_14.pkl',
    'conditions': CONDITIONS,
    'nn_demo': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/reacher_img.xml',
    'x0': np.zeros(4),
    'dt': 0.05,
    'substeps': 5,
    'pos_body_offset': None,
    'pos_body_idx': np.array([4]),
    'conditions': common['conditions'],
    'T': 50,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
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
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'camera_pos': np.array([0., 0., 1.5, 0., 0., 0.]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'ioc' : 'ICML',
    'max_ent_traj': 1.0,
    'conditions': common['conditions'],
    'kl_step': 0.5,
    'min_step_mult': 0.05,
    'max_step_mult': 2.0,
    'num_demos': 20,
    'demo_var_mult': 1.0,
    'synthetic_cost_samples': 100,
    'policy_sample_mode': 'replace',
    'iterations': 25,
    'plot_dir': EXP_DIR,
}


# algorithm['policy_opt'] = {
#     'type': PolicyOptCaffe,
#     'iterations': 5000,
#     'weights_file_prefix': common['data_files_dir'] + 'policy',
# }

#algorithm['policy_opt'] = {
#    'type': PolicyOptTf,
#    'network_params': {
#        'num_filters': [15, 15, 15],
#        'obs_include': agent['obs_include'],
#        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
#        'obs_image_data': [RGB_IMAGE],
#        'image_width': IMAGE_WIDTH,
#        'image_height': IMAGE_HEIGHT,
#        'image_channels': IMAGE_CHANNELS,
#        'sensor_dims': SENSOR_DIMS,
#    },
#    'network_model': multi_modal_network_fp,
#    'fc_only_iterations': 5000,
#    'init_iterations': 1000,
#    'iterations': 100,
#    'weights_file_prefix': EXP_DIR + 'policy',
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

#algorithm['init_traj_distr'] = {
##    'type': init_demo,
#    'init_gains':  1.0 / PR2_GAINS,
#    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#    'init_var': 5.0,
#    'stiffness': 1.0,
#    'stiffness_vel': 0.5,
#    'final_weight': 50.0,
#    'dt': agent['dt'],
#    'T': agent['T'],
#}

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

algorithm['gt_cost'] = [{
    'type': CostSum,
    'costs': [torque_cost_1[i], fk_cost_1[i]],
    'weights': [200.0, 100.0],
}  for i in range(common['conditions'])][0]

algorithm['cost'] = {  # TODO - make vision cost and emp. est derivatives
    'type': CostIOCNN,
    'wu': 200 / PR2_GAINS,
    'T': agent['T'],
    'dO': 16,
    'iterations': 5000,
    'demo_batch_size': 5,
    'sample_batch_size': 5,
    'ioc_loss': algorithm['ioc'],
    'smooth_reg_weight': 1.0,
    'mono_reg_weight': 0.0,
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 10,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
    'min_eta': 1e-4,
    'max_eta': 1.0,
}



NUM_SAMPLES = 5
config = {
    'iterations': algorithm['iterations'],
    'num_samples': NUM_SAMPLES,
    'verbose_trials': NUM_SAMPLES,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
    'random_seed': 1,
}

common['info'] = generate_experiment_info(config)
