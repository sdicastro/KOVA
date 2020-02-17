import sys
import os
import os.path as osp
import random

import gym
from collections import defaultdict
import tensorflow as tf
import datetime
import shutil
import json
import multiprocessing

from stable_baselines import SAC
from stable_baselines import logger as logger2
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from baselines.common.tf_util import get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common import atari_wrappers, retro_wrappers

from kalman_algorithms.kalman_sac import KalmanSAC

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # env_type = env._entry_point.split(':')[0].split('.')[-1]
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type):
    env_type, env_id = get_env_type(args.env)
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    if args.alg == 'sac' and args.vf_alg == 'sac':
        sac_class = SAC(policy="MlpPolicy", env=args.env, verbose=1)
        learn = sac_class.learn
    elif args.alg == 'sac' and args.vf_alg == 'kalman':
        kalman_kwargs = get_kalman_learn_function_defaults(args.alg)
        kalman_kwargs["kalman_lr"] = kalman_lr
        kalman_kwargs["kalman_eta"] = kalman_eta
        kalman_kwargs["kalman_onv_coeff"] = kalman_onv_coeff
        kalman_kwargs["kalman_onv_type"] = kalman_onv_type
        alg_kwargs.update(kalman_kwargs)
        print("kalman_kwargs", kalman_kwargs)
        sac_class = KalmanSAC(policy="MlpPolicy", env=args.env, verbose=1, **kalman_kwargs)
        learn = sac_class.learn

    elif args.vf_alg == 'kalman':
        learn = get_kalman_learn_function(args.alg)
        kalman_kwargs = get_kalman_learn_function_defaults(args.alg)
        kalman_kwargs["kalman_lr"] = kalman_lr
        kalman_kwargs["kalman_eta"] = kalman_eta
        kalman_kwargs["kalman_onv_coeff"] = kalman_onv_coeff
        kalman_kwargs["kalman_onv_type"] = kalman_onv_type
        alg_kwargs.update(kalman_kwargs)
    else:
        learn = get_learn_function(args.alg)

    if args.alg == 'trpo_mpi':
        alg_kwargs['vf_with_fisher'] = False  # vf in trpo is optimized with adam

    env = build_env(args)
    if args.eval and env_type == 'mujoco':
        eval_env = build_env(args)
        model_in_eval_env = eval_env.venv.envs[0].env.env.model
        original_body_mass = model_in_eval_env.body_mass[1]
        new_mean_mass = args.body_mass_percentage * original_body_mass / 100
        model_in_eval_env.body_mass[1] = new_mean_mass

        alg_kwargs['eval_env'] = eval_env

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.alg == 'sac':
        model = learn(total_timesteps=total_timesteps)
    else:
        model = learn(env=env, seed=seed, total_timesteps=total_timesteps, **alg_kwargs)

    return model, env, alg_kwargs


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type == 'atari':
        if alg == 'acer':
            env = make_vec_env(env_id, env_type, nenv, seed)
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            env.seed(seed)
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_vec_env(env_id, env_type, nenv, seed), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000,
                                        use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)

    else:
        if alg == 'trpo_mpi':
            get_session(tf.ConfigProto(allow_soft_placement=True,
                                       intra_op_parallelism_threads=1,
                                       inter_op_parallelism_threads=1,
                                       gpu_options=tf.GPUOptions(allow_growth=True)))
        else:
            get_session(tf.ConfigProto(allow_soft_placement=True,
                                       intra_op_parallelism_threads=1,
                                       inter_op_parallelism_threads=1,
                                       gpu_options=tf.GPUOptions(allow_growth=True)))

        if alg == 'sac':
            from stable_baselines.common.vec_env import DummyVecEnv
            from stable_baselines.common.vec_env import VecNormalize as VecNormalize2
            from stable_baselines import bench as bench2

            env = gym.make(env_id)
            env = bench2.Monitor(env, logger2.get_dir())
            env.seed(seed)
            env = DummyVecEnv([lambda: env])
            env = VecNormalize2(env)
        else:
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)

        if env_type == 'mujoco' and alg is not 'sac':
            env = VecNormalize(env)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    print("env_type", env_type)
    return env_type, env_id


def get_default_network(env_type):
    if env_type == 'atari':
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        if alg == 'sac':
            alg_module = import_module('.'.join(['stable_baselines', alg, submodule]))
            print("alg_module", alg_module)
        else:
            alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_kalman_alg_module(alg, submodule=None):
    submodule = submodule or alg
    kalman_alg_module = import_module('.'.join(['kalman_algorithms', 'kalman_' + submodule]))
    return kalman_alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_kalman_learn_function(alg):
    return get_kalman_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def get_kalman_learn_function_defaults(alg):
    try:
        alg_defaults = get_kalman_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, 'kalman')()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    """
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    """
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def main_run(env, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type, comb_num):
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
        logger2.configure()
    else:
        logger.configure(format_strs=[])
        logger2.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    seed = random.randint(0, 100)
    arg_parser = common_arg_parser()
    arg_parser.set_defaults(env=env)
    arg_parser.set_defaults(alg='trpo_mpi')  # 'trpo_mpi', 'ppo2', 'acktr', 'sac'
    arg_parser.set_defaults(num_timesteps=2e6)
    arg_parser.set_defaults(seed=seed)
    arg_parser.set_defaults(save_path=os.path.join(logger.get_dir(), 'policy'))
    arg_parser.set_defaults(save_path2=os.path.join(logger2.get_dir(), 'policy'))
    arg_parser.add_argument('--vf_alg', default="kalman", type=str)  # 'kalman', 'ppo2', 'trpo_mpi', 'sac', 'acktr'
    arg_parser.add_argument('--body_mass_percentage', default=100, type=int)
    arg_parser.add_argument('--eval', default=False)

    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    env_type, env_id = get_env_type(args.env)
    print("env_type", env_type)
    print("env_id", env_id)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    kalman_kwargs = get_kalman_learn_function_defaults(args.alg)

    alg_kwargs.update(kalman_kwargs)
    alg_kwargs.update(extra_args)
    if args.vf_alg == "kalman":
        alg_kwargs['kalman_lr'] = kalman_lr
        alg_kwargs['kalman_eta'] = kalman_eta
        alg_kwargs['kalman_onv_coeff'] = kalman_onv_coeff
        alg_kwargs['kalman_onv_type'] = kalman_onv_type

    if args.alg == 'trpo_mpi':
        alg_kwargs['vf_with_fisher'] = False  # vf in trpo is optimized with adam

    alg_kwargs_for_json = {key: value for (key, value) in alg_kwargs.items() if not callable(value)}

    if args.alg == 'sac':
        curdir = logger2.get_dir()
    else:
        curdir = logger.get_dir()
    print("curdir", curdir)

    with open(curdir + '/info.txt', 'w') as infofile:
        infofile.write(json.dumps(vars(args)))
        infofile.write('\n')
        infofile.write(json.dumps(alg_kwargs_for_json))

    exp_dir = os.getcwd()
    if args.alg == 'ppo2':
        exp_name = "/results/exp_{}_{}_pol-{}_vf-{}_{}_{}_{}_{}_{}_{}_{}_{}_last-{}_sepVars-{}_eval-{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
            args.env, 'ppo',
            'ppo' if args.vf_alg == 'ppo2' else args.vf_alg,
            alg_kwargs['kalman_lr'],
            alg_kwargs['kalman_onv_coeff'],
            alg_kwargs['kalman_eta'],
            alg_kwargs['kalman_onv_type'],
            args.body_mass_percentage,
            alg_kwargs['nsteps']/alg_kwargs['nminibatches'],
            args.seed,
            comb_num,
            str(alg_kwargs['kalman_only_last_layer'])[0],
            str(alg_kwargs['kalman_separate_vars'])[0],
            str(args.eval)[0]
        )
    else:
        exp_name = "/results/exp_{}_{}_pol-{}_vf-{}_{}_{}_{}_{}_{}_{}_{}_last-{}_sepVars-{}_eval-{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
            args.env, 'trpo' if args.alg == 'trpo_mpi' else args.alg,
            'trpo' if args.vf_alg == 'trpo_mpi' else args.vf_alg,
            alg_kwargs['kalman_lr'],
            alg_kwargs['kalman_onv_coeff'],
            alg_kwargs['kalman_eta'],
            alg_kwargs['kalman_onv_type'],
            args.body_mass_percentage,
            args.seed,
            comb_num,
            str(alg_kwargs['kalman_only_last_layer'])[0],
            str(alg_kwargs['kalman_separate_vars'])[0],
            'F'
            )
    print("exp_dir", exp_dir)
    print("exp_name", exp_name)

    model, env, alg_kwargs = train(args, extra_args, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type)

    if args.save_path is not None and rank == 0:
        if args.alg == 'sac':
            save_path = osp.expanduser(args.save_path2)
        else:
            save_path = osp.expanduser(args.save_path)

    shutil.copytree(curdir, exp_dir + exp_name)
    print("We copied to: ", exp_dir, exp_name)

    env.close()
    sess = tf.get_default_session()
    sess.close()
    tf.reset_default_graph()


if __name__ == '__main__':
    env = 'Walker2d-v2'  # 'Swimmer-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2'
    kalman_lr = 1.
    kalman_eta = 0.01
    kalman_onv_coeff = 1.
    kalman_onv_type = 'batch-size'  # 'max-ratio', 'batch_size'
    comb_num = 80
    main_run(env, kalman_lr, kalman_eta, kalman_onv_coeff, kalman_onv_type, comb_num)
