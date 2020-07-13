import torch
import json
import os
import yaml
import gym_minigrid  # noqa
import numpy as np
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.samplers.sampler import make_env
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
from tensorboardX import SummaryWriter


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    # Set tb_writer
    log_name = "env-name::%s_num-steps::%s_log" % (config["env-name"], config["num-steps"])
    tb_writer = SummaryWriter("./{0}/{1}_logs".format(args.output_folder, log_name))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = make_env(config["env-name"])()
    env.close()

    # Policy
    policy = get_policy_for_env(
        env,
        hidden_sizes=config['hidden-sizes'],
        nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(
        config['env-name'],
        env_kwargs=config.get('env-kwargs', {}),
        batch_size=config['fast-batch-size'],
        policy=policy,
        baseline=baseline,
        env=env,
        seed=args.seed,
        num_workers=args.num_workers)

    metalearner = MAMLTRPO(
        policy,
        fast_lr=config['fast-lr'],
        first_order=config['first-order'],
        device=args.device)

    for batch in range(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])

        futures = sampler.sample_async(
            tasks,
            num_steps=config['num-steps'],
            fast_lr=config['fast-lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae-lambda'],
            device=args.device)

        metalearner.step(
            *futures,
            max_kl=config['max-kl'],
            cg_iters=config['cg-iters'],
            cg_damping=config['cg-damping'],
            ls_max_steps=config['ls-max-steps'],
            ls_backtrack_ratio=config['ls-backtrack-ratio'])

        # For logging
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        tb_writer.add_scalars("reward/", {"train": np.mean(get_returns(train_episodes[0]))}, batch)
        tb_writer.add_scalars("reward/", {"val": np.mean(get_returns(valid_episodes))}, batch)
        print(batch, np.mean(get_returns(train_episodes[0])), np.mean(get_returns(valid_episodes)))

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description='Reinforcement learning with Model-Agnostic Meta-Learning (MAML) - Train')
    parser.add_argument(
        '--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument(
        '--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument(
        '--seed', type=int, default=None,
        help='random seed')
    misc.add_argument(
        '--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument(
        '--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')

    main(args)
