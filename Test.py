# Training the Convolutional Neural Network to play Breakout Atari game
# The code is based on the demo https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Atari--VmlldzoxMTExNTI and
# https://wandb.ai/cleanrl/cleanrl.benchmark/runs/lqyi4g2g/files/code/cleanrl/ppo_atari_visual.py

# Usage
# Run: python3 ai_breakout_game.py --run=1 --model-filename="ai_breakout_game.mdl"
# Train: python3 ai_breakout_game.py --num-envs=8 --total-timesteps=3000000 --multithreading=True
# To see statistics in real-time, run a tensorboard server:
# tensorboard --logdir=runs


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import numpy as np
from datetime import datetime
import time
import random
import os
import cv2
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv, MaxAndSkipEnv, ClipRewardEnv, WarpFrame
from distutils.util import strtobool
import logging
import argparse


gym_id = "BreakoutNoFrameskip-v4"

# Gym wrappers imported from:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class RGBSaveEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.rgb_frame = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.rgb_frame = obs
        return obs, reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]



class ImageToPyTorch(gym.ObservationWrapper):
    """
    Imported from https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/atari_wrappers.py
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


class VecPyTorch(VecEnvWrapper):
    """
    Helper class to run multiple game environments
    Imported from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_env(gym_id, seed, keep_rgb=False):
    def make_func():
        env = gym.make(gym_id)
        env = NoopResetEnv(env, noop_max=30)
        if keep_rgb:
            env = RGBSaveEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = RecordEpisodeStatistics(env)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)
        env = ImageToPyTorch(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return make_func


class Agent(nn.Module):
    def __init__(self, n_actions, frames=4):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            Agent.layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            Agent.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            Agent.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            Agent.layer_init(nn.Linear(3136, 512)),
            nn.ReLU())
        self.actor = Agent.layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = Agent.layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.network(x/255.0)

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def save(self, max_steps):
        fname = f"ai_breakout-{int(max_steps)}.mdl"
        torch.save(self.state_dict(), fname)
        logging.debug("Model saved to %s" % fname)

    def load(self, model_filename):
        self.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))


def run(g_id: str, model_filename: str):
    env = make_env(g_id, seed=int(time.time()), keep_rgb=True)()

    agent = Agent(n_actions=env.action_space.n)
    agent.load(model_filename)
    agent.eval()
    agent = agent.to(device)
    gif_writer, gif_filename = None, ""  # Recording to .gif, optional
    with torch.no_grad():
        state = env.reset()
        step_num, total_reward = 0, 0
        while True:
            s_ = torch.from_numpy(state).reshape((1, 4, 84, 84)).to(device)
            action, _, _ = agent.get_action(s_)
            state, reward, done, info = env.step(action)
            total_reward += reward
            logging.debug(f"S{step_num}: Action={action}\tstate={state.shape}, reward={reward}, total_reward={total_reward}, done={done}")

            # Show
            frame_data = env.rgb_frame  # state[0]//4 + state[1]//2 + state[2]//1 - to see data 'as is'
            img = cv2.resize(frame_data, (480, 480), interpolation=cv2.INTER_CUBIC)
            if gif_writer is not None:
                gif_writer.append_data(img)
            cv2.imshow('Frame', img)
            # Process keys
            key_code = cv2.waitKey(1)
            if key_code & 0xFF == 27:
                break
            if key_code & 0xFF == ord('g'):
                # Save frames stream as GIF
                if gif_writer is None:
                    gif_filename = datetime.now().strftime("ai_breakout_%Y%m%d%H%M%S.gif")
                    gif_writer = imageio.get_writer(gif_filename, mode='I')
                    logging.info("File %s will be saved" % gif_filename)
                else:
                    gif_writer.close()
                    gif_writer = None
                    logging.info("Done")

            if done:
                state = env.reset()
                if 'lives' in info and info['lives'] == 0:
                    total_reward = 0

            time.sleep(0.1)
            step_num += 1

    env.close()
    if gif_writer:
        gif_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=False, type=lambda x: bool(strtobool(x)), help='Run the pre-trained network')
    parser.add_argument('--model-filename', default="", type=str)
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total timesteps of the train experiment')
    parser.add_argument('--multithreading', default=False, type=lambda x: bool(strtobool(x)), help='Enable multithreading for training')
    parser.add_argument('--num-envs', type=int, default=8, help='The number of parallel game environment')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.run else 'cpu')

    # Run a pre-trained model
    if args.run:
        run(gym_id, model_filename=args.model_filename)
        exit(0)

    # Training parameters
    update_epochs = 4  # The K epochs to update the policy
    n_minibatch = 4  # The number of mini batch
    num_steps = 128  # The number of steps per game environment
    batch_size = int(args.num_envs*num_steps)
    minibatch_size = int(batch_size//n_minibatch)

    # Setup the environment
    exp_name = os.path.basename(__file__).rstrip(".py")
    learning_rate = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95  # Lambda for the general advantage estimation
    max_grad_norm = 0.5  # The maximum norm for the gradient clipping
    clip_coef = 0.1  # The surrogate clipping coefficient
    vf_coef = 0.5  # Coefficient of the value function
    ent_coef = 0.01  # Coefficient of the entropy
    multithreading = True
    seed = int(time.time())
    experiment_name = f"{gym_id}__{exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.multithreading:
        envs = VecPyTorch(SubprocVecEnv([make_env(gym_id, seed + i) for i in range(args.num_envs)], "fork"), device)
    else:
        envs = VecPyTorch(DummyVecEnv([make_env(gym_id, seed + i) for i in range(args.num_envs)]), device)
    assert isinstance(envs.action_space, Discrete), "Only discrete action space is supported"

    agent = Agent(n_actions=envs.action_space.n).to(device)
    if len(args.model_filename) > 0:
        logging.debug("Loading the model from %s" % args.model_filename)
        agent.load(args.model_filename)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    lr = lambda f: f * learning_rate

    # Storage for epoch data
    obs = torch.zeros((num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((num_steps, args.num_envs)).to(device)
    dones = torch.zeros((num_steps, args.num_envs)).to(device)
    values = torch.zeros((num_steps, args.num_envs)).to(device)

    # Start the game
    global_step, global_game = 0, 0
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // batch_size
    t_start = time.monotonic()
    logging.debug(f"Train: total_timesteps={args.total_timesteps}, batch_size={batch_size}, num_steps={num_steps}, num_updates={num_updates}")
    for update in range(1, num_updates+1):
        # Update the learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

        # Prepare the execution of the game.
        for step in range(num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Put action logic here
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                action, logproba, _ = agent.get_action(obs[step])

            actions[step] = action
            logprobs[step] = logproba

            # Execute the game and log data.
            next_obs, rs, ds, infos = envs.step(action)
            rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
            for info in infos:
                if 'lives' in info.keys() and info['lives'] == 0:
                    global_game += 1
                if 'episode' in info.keys():
                    logging.debug(f"global_games={global_game}, global_step={global_step}, episode_reward={info['episode']['r']}")
                    writer.add_scalar("charts/episode_reward", info['episode']['r'], global_game)
                    break

        # Bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + gamma*nextvalues*nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma*gae_lambda*nextnonterminal*lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        target_agent = Agent(n_actions=envs.action_space.n).to(device)
        inds = np.arange(batch_size,)
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            target_agent.load_state_dict(agent.state_dict())
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - ent_coef*entropy_loss + v_loss*vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


    agent.save(max_steps=args.total_timesteps)
    envs.close()
    writer.close()

    logging.debug(f"Training complete, T={time.monotonic() - t_start}s")