from typing import Union, Optional
from logging import Logger, getLogger
from uuid import uuid4
import pathlib
from datetime import datetime
import os.path
import numpy as np
from pert import PERT
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.yawning_titan_run import YawningTitanRun
from yawning_titan import AGENTS_DIR, PPO_TENSORBOARD_LOGS_DIR
from yawning_titan.game_modes.game_mode_db import default_game_mode, GameModeDB
from yawning_titan.networks.network import Network
from yawning_titan.networks.network_db import default_18_node_network, NetworkDB

from adaptive_red import AdaptiveRed, BayesHurwiczRed
from nsa_red import NSARed
from multiagent_env import MultiAgentEnv

_LOGGER = getLogger(__name__)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
SUPPORTED_TYPES = ["Ransomware", "APT"]

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        
    def __len__(self):
        return len(self.rewards)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(DEVICE)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(DEVICE)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(DEVICE)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, attacker_type="Ransomware", k_epochs=10, lr_actor=0.0002, lr_critic=0.0004,
                 gamma=0.99, eps_clip=0.2, has_continuous_action_space=False, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        assert attacker_type in SUPPORTED_TYPES, f"Specified attacker type is not supported. Choose from {SUPPORTED_TYPES}"
        self.type = attacker_type
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(DEVICE)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class BayesHurwiczAgent(PPO):
    def __init__(self, state_dim, action_dim, p_hat_path, p_bar_path, p_ubar_path, attacker_type,
                 k_epochs=5, lr_actor=0.0002, lr_critic=0.0005, gamma=0.99, eps_clip=0.2,
                 has_continuous_action_space=False, action_std_init=0.6):
        super().__init__(state_dim, action_dim, attacker_type, k_epochs, lr_actor, lr_critic, gamma, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.load(p_hat_path)
        self.p_bar = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)
        self.p_bar.load_state_dict(torch.load(p_bar_path, map_location=lambda storage, loc: storage))
        self.p_ubar = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(DEVICE)
        self.p_ubar.load_state_dict(torch.load(p_ubar_path, map_location=lambda storage, loc: storage))
        self.k = 0
        self.gamma = 1
        self.mu_hat = PERT(0, 0.5, 1).rvs().item()
        self.phi_hat = PERT(0.2, 0.5, 0.8).rvs().item()
        self.al_hats = list()

    def select_action(self, state):
        if self.has_continuous_action_space:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                hat_action_probs = self.policy_old.actor(state)
                bar_action_probs = self.p_bar.actor(state)
                ubar_action_probs = self.p_ubar.actor(state)
                hat_state_val = self.policy_old.critic(state)
                bar_state_val = self.p_bar.critic(state)
                ubar_state_val = self.p_ubar.critic(state)

                action_probs = self.bayes_hurwicz(hat_action_probs, bar_action_probs, ubar_action_probs)
                state_val = self.get_state_val(hat_state_val, bar_state_val, ubar_state_val)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def bayes_hurwicz(self, hat_action_probs, bar_action_probs, ubar_action_probs):
        bhc = self.gamma * hat_action_probs + \
              (1 - self.gamma) * ((1 - self.mu_hat) * bar_action_probs +
                                  self.mu_hat * ubar_action_probs)
        return bhc

    def get_state_val(self, hat_state_val, bar_state_val, ubar_state_val):
        state_val = self.gamma * hat_state_val + \
                    (1 - self.gamma) * ((1 - self.mu_hat) * bar_state_val + self.mu_hat * ubar_state_val)
        return state_val

    def update_mu_gamma(self, blue_action, network_interface):
        blue_action_target_node = blue_action[1]
        if blue_action_target_node is None:
            return
        else:
            blue_target_uuid = blue_action_target_node.uuid()
            compromised_states = network_interface.get_all_node_compromised_states()
            if compromised_states[blue_target_uuid] == 0:
                return
            else:
                self.k += 1
                num_alerts = self.mu_hat * self.phi_hat
                self.al_hats.append(num_alerts)
                al_hat = np.average(num_alerts)
                var_al_hat = np.var(num_alerts)
                alpha = (al_hat ** 2) * (((1 - al_hat) / var_al_hat) - (1 / al_hat))
                beta = alpha * ((1 / al_hat) - 1)
                self.gamma = self.gamma / self.k
                self.mu_hat = np.random.beta(alpha, beta)


class MultiAgentYTRun(YawningTitanRun):

    def __init__(self, network: Optional[Network] = None, game_mode: Optional[GameMode] = None,
                 red_agent_class: object = AdaptiveRed, blue_agent_class: object = BlueInterface,
                 print_metrics: bool = False, show_metrics_every: int = 1, collect_additional_per_ts_data: bool = False,
                 eval_freq: int = 10, total_timesteps: int = 5000, training_runs: int = 1000,
                 n_eval_episodes: int = 1, deterministic: bool = False, warn: bool = True, render: bool = False,
                 verbose: int = 1, logger: Optional[Logger] = None, output_dir: Optional[str] = None, auto: bool = True,
                 **kwargs: object) -> object:

        super().__init__(network, game_mode, red_agent_class, blue_agent_class, print_metrics, show_metrics_every,
                         collect_additional_per_ts_data, eval_freq, total_timesteps, training_runs, n_eval_episodes,
                         deterministic, warn, render, verbose, logger, output_dir, auto, **kwargs)
        self.uuid = str(uuid4())

        self.network_interface: Optional[NetworkInterface] = None
        self.red: Optional[AdaptiveRed] = None
        self.blue: Optional[BlueInterface] = None
        self.env: Optional[MultiAgentEnv] = None
        self.blue_agent: Optional[PPO] = None
        self.red_agent: Optional[PPO] = None

        # Set the network using the network arg if one was passed,
        # otherwise use the default 18 node network.
        if network:
            self.network: Network = network
        else:
            self.network = default_18_node_network()

        # Set the game_mode using the game_mode arg if one was passed,
        # otherwise use the game mode
        if game_mode:
            self.game_mode: GameMode = game_mode
        else:
            self.game_mode = default_game_mode()

        self._red_agent_class = red_agent_class
        self._blue_agent_class = blue_agent_class

        self.print_metrics = print_metrics
        self.show_metrics_every = show_metrics_every
        self.collect_additional_per_ts_data = collect_additional_per_ts_data
        self.eval_freq = eval_freq
        self.total_timesteps = total_timesteps
        self.training_runs = training_runs
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.warn = warn
        self.render = render
        self.verbose = verbose
        self.auto = auto

        self.logger = _LOGGER if logger is None else logger
        self.logger.debug(f"YT run  {self.uuid}: Run initialised")

        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=f"{PPO_TENSORBOARD_LOGS_DIR}/{self.uuid}")

        # Automatically setup, train, and evaluate the agent if auto is True.
        if self.auto:
            self.setup()
            self.train()
            self.evaluate()
            self.save()

    def _get_new_ppo(self, agent_type: str, attacker_type: str = "Ransomware") -> PPO:
        obs_space = self.env.observation_space.shape[0]
        if agent_type == "red":
            action_space = len(self.red.action_dict)
        else:
            action_space = self.env.action_space.n
        agent = PPO(obs_space, action_space, attacker_type)
        return agent

    def setup(self, new=True, blue_ppo_path=None, red_ppo_path=None, red_agent_type="Ransomware",
              p_hat_path=None, p_bar_path=None, p_ubar_path=None):
        if not new and not blue_ppo_path:
            msg = "Performing setup when new=False requires saved PPO .zip files"
            try:
                raise AttributeError(msg)
            except AttributeError as e:
                _LOGGER.critical(e)
                raise e
        if not new and isinstance(self.red, AdaptiveRed) and not red_ppo_path:
            msg = "Performing setup when new=False requires saved PPO .zip files"
            try:
                raise AttributeError(msg)
            except AttributeError as e:
                _LOGGER.critical(e)
                raise e

        if self.output_dir:
            if isinstance(self.output_dir, str):
                self.output_dir = pathlib.Path(self.output_dir)
        else:
            self.output_dir = pathlib.Path(
                os.path.join(
                    AGENTS_DIR, "trained", str(datetime.now().date()), f"{self.uuid}"
                )
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.network_interface = NetworkInterface(game_mode=self.game_mode, network=self.network)
        self.logger.debug(f"YT run  {self.uuid}: Network interface created")

        self.red = self._red_agent_class(self.network_interface)
        self.logger.debug(f"YT run  {self.uuid}: Red agent created")

        self.blue = self._blue_agent_class(self.network_interface)
        self.logger.debug(f"YT run  {self.uuid}: Blue agent created")

        self.env = MultiAgentEnv(red_agent=self.red,
                                 blue_agent=self.blue,
                                 network_interface=self.network_interface,
                                 print_metrics=self.print_metrics,
                                 show_metrics_every=self.show_metrics_every,
                                 collect_additional_per_ts_data=self.collect_additional_per_ts_data)

        self.logger.debug(f"YT run  {self.uuid}: MultiAgentEnv created")

        self.logger.debug(f"YT run  {self.uuid}: Env checking complete")

        self.env.reset()
        self.logger.debug(f"YT run  {self.uuid}: GenericNetworkEnv reset")
        self.logger.debug(f"YT run  {self.uuid}: Instantiating agent")
        if new:
            self.blue_agent = self._get_new_ppo("blue")
            if isinstance(self.red, BayesHurwiczRed):
                if p_hat_path and p_bar_path and p_ubar_path:
                    obs_space = self.env.observation_space.shape[0]
                    action_space = len(self.red.action_dict)
                    self.red_agent = BayesHurwiczAgent(state_dim=obs_space,
                                                       action_dim=action_space,
                                                       p_hat_path=p_hat_path,
                                                       p_bar_path=p_bar_path,
                                                       p_ubar_path=p_ubar_path
                                                       )
                else:
                    msg = "Bayes-Hurwicz agent requires paths to all three pretrained networks."
                    raise Exception(msg)
            elif isinstance(self.red, NSARed):
                self.red_agent = self.red
            else:
                self.red_agent = self._get_new_ppo("red")
        else:
            if isinstance(self.red, BayesHurwiczRed):
                msg = "Bayes-Hurwicz agents require pretrained networks to operate."
                raise Exception(msg)

            self.blue_agent = self._load_existing_ppo(blue_ppo_path)
            if isinstance(self.red, NSARed):
                self.red_agent = self.red
            else:
                self.red_agent = self._load_existing_ppo(red_ppo_path)
        self.logger.debug(f"YT run  {self.uuid}: Agent instantiated")

    def train(self) -> Union[PPO, PPO, None]:
        if self.env and self.blue_agent and self.red_agent:
            self.logger.debug(f"YT run  {self.uuid}: Performing agent training")
            red_running_reward = 0
            blue_running_reward = 0
            episode_lengths = list()
            for i in range(self.training_runs):
                state = self.env.reset()
                self.logger.debug(f"YT run  {self.uuid}: MultiAgentEnv reset")
                red_ep_reward = 0
                blue_ep_reward = 0
                global ep_length
                ep_length = 0
                for t in range(1, self.total_timesteps):
                    red_action = self.red_agent.select_action(state)
                    blue_action = self.blue_agent.select_action(state)
                    if isinstance(self.red_agent, BayesHurwiczAgent):
                        self.red_agent.update_mu_gamma(blue_action, self.network_interface)
                    state, red_reward, blue_reward, done, notes = self.env.step(red_action, blue_action, self.red_agent.type)
                    self.red_agent.buffer.rewards.append(red_reward)
                    self.red_agent.buffer.is_terminals.append(done)
                    self.blue_agent.buffer.rewards.append(blue_reward)
                    self.blue_agent.buffer.is_terminals.append(done)
                    red_ep_reward += red_reward
                    blue_ep_reward += blue_reward
                    if done:
                        self.writer.add_scalar("Red Episode Reward", red_ep_reward, i)
                        self.writer.add_scalar("Blue Episode Reward", blue_ep_reward, i)
                        self.writer.add_scalar("Episode Length", t, i)
                        ep_length = t
                        episode_lengths.append(ep_length)
                        break

                red_running_reward += red_ep_reward
                blue_running_reward += blue_ep_reward
                if i % 64 == 0:
                    self.red_agent.update()
                    self.blue_agent.update()

                print(f'Episode: {i+1} \t Episode Length:{ep_length} \t'
                      f'Red Reward: {red_ep_reward:.2f} \t Blue Reward {blue_ep_reward:.2f}\t')
                if i != 0 and i % 100 == 0:
                    red_avg_reward = red_running_reward / i
                    blue_avg_reward = blue_running_reward / i
                    avg_ep_length = np.average(episode_lengths)
                    print(f"Average stats after {i} episodes:\n"
                          f"Average episode length: {avg_ep_length:.0f} \t "
                          f"Average red reward: {red_avg_reward:.2f} \t Average blue reward: {blue_avg_reward:.2f}")
                self.logger.debug(f"YT run  {self.uuid}: Episode {i + 1} complete")

            red_avg_reward = red_running_reward / self.training_runs
            blue_avg_reward = blue_running_reward / self.training_runs
            print(f"Training complete!\n"
                  f"Average red reward: {red_avg_reward:.2f} \t\t Average blue reward: {blue_avg_reward:.2f}")
            self.logger.debug(f"YT run  {self.uuid}: Agent training complete")
            return self.red_agent, self.blue_agent
        else:
            self.logger.error(
                f"Cannot train the agent for YT run  {self.uuid} as the run has not been setup. "
                f"Call .setup() on the instance of {self.__class__.__name__} to setup the run."
            )

    def multi_type_setup(self, new=True):
        if self.output_dir:
            if isinstance(self.output_dir, str):
                self.output_dir = pathlib.Path(self.output_dir)
        else:
            self.output_dir = pathlib.Path(
                os.path.join(
                    AGENTS_DIR, "trained", str(datetime.now().date()), f"{self.uuid}"
                )
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.network_interface = NetworkInterface(game_mode=self.game_mode, network=self.network)
        self.logger.debug(f"YT run  {self.uuid}: Network interface created")

        self.red = self._red_agent_class(self.network_interface)
        self.logger.debug(f"YT run  {self.uuid}: Red agent created")

        self.blue = self._blue_agent_class(self.network_interface)
        self.logger.debug(f"YT run  {self.uuid}: Blue agent created")

        self.env = MultiAgentEnv(red_agent=self.red,
                                 blue_agent=self.blue,
                                 network_interface=self.network_interface,
                                 print_metrics=self.print_metrics,
                                 show_metrics_every=self.show_metrics_every,
                                 collect_additional_per_ts_data=self.collect_additional_per_ts_data)

        self.logger.debug(f"YT run  {self.uuid}: MultiAgentEnv created")

        self.logger.debug(f"YT run  {self.uuid}: Env checking complete")

        self.env.reset()
        self.logger.debug(f"YT run  {self.uuid}: GenericNetworkEnv reset")
        self.logger.debug(f"YT run  {self.uuid}: Instantiating agent")
        self.blue_agent = self._get_new_ppo("blue")
        self.red_rw_agent = self._get_new_ppo("red", "Ransomware")
        self.red_apt_agent = self._get_new_ppo("red", "APT")
        self.logger.debug(f"YT run  {self.uuid}: Agent instantiated")

    def multi_type_train(self) -> Union[PPO, PPO, PPO, None]:
        if self.env and self.blue_agent and self.red_rw_agent and self.red_apt_agent:
            self.logger.debug(f"YT run  {self.uuid}: Performing agent training")
            blue_rewards = list()
            apt_rewards = list()
            rw_rewards = list()
            episode_lengths = list()
            for i in range(self.training_runs):
                active_attacker = np.random.choice([self.red_rw_agent, self.red_apt_agent])
                state = self.env.reset()
                self.logger.debug(f"YT run  {self.uuid}: MultiAgentEnv reset")
                red_ep_reward = 0
                blue_ep_reward = 0
                global ep_length
                ep_length = 0
                for t in range(1, self.total_timesteps):
                    red_action = active_attacker.select_action(state)
                    blue_action = self.blue_agent.select_action(state)
                    state, red_reward, blue_reward, done, notes = self.env.step(red_action, blue_action, active_attacker.type)
                    active_attacker.buffer.rewards.append(red_reward)
                    active_attacker.buffer.is_terminals.append(done)
                    self.blue_agent.buffer.rewards.append(blue_reward)
                    self.blue_agent.buffer.is_terminals.append(done)
                    red_ep_reward += red_reward
                    blue_ep_reward += blue_reward
                    if done:
                        blue_rewards.append(blue_ep_reward)
                        if active_attacker.type == "Ransomware":
                            rw_rewards.append(red_ep_reward)
                            self.writer.add_scalar("Ransomware Reward", red_ep_reward, i)
                        else:
                            self.writer.add_scalar("APT Reward", red_ep_reward, i)
                            apt_rewards.append(red_ep_reward)
                        self.writer.add_scalar("Blue Episode Reward", blue_ep_reward, i)
                        self.writer.add_scalar("Episode Length", t, i)
                        ep_length = t
                        episode_lengths.append(ep_length)
                        break
                if i > 0 and i % 64 == 0:
                    if len(self.red_rw_agent.buffer) >= 16:
                        self.red_rw_agent.update()
                    if len(self.red_apt_agent.buffer) >= 16:
                        self.red_apt_agent.update()
                    self.blue_agent.update()

#                 print(f'Episode: {i} \t Episode Length:{ep_length} \t'
#                       f'Red Reward: {red_ep_reward:.2f} \t Blue Reward {blue_ep_reward:.2f}\t')
                if i > 0 and i % 50 == 0:
                    rw_avg_reward = np.average(rw_rewards)
                    rw_wins = np.sum([1 if x > 0 else 0 for x in rw_rewards])
                    apt_avg_reward = np.average(apt_rewards)
                    apt_wins = np.sum([1 if x > 0 else 0 for x in apt_rewards])
                    blue_avg_reward = np.average(blue_rewards)
                    print(f"Stats as of episode {i}\n"
                          f"Average ransomware reward: {rw_avg_reward:.2f} \t Ransomware episodes: {len(rw_rewards)} \t Ransomware wins: {rw_wins}\n"
                          f"Average APT reward: {apt_avg_reward:.2f} \t APT episodes: {len(apt_rewards)} \t APT wins: {apt_wins}\n"
                          f"Average blue reward: {blue_avg_reward:.2f}")
                    self.logger.debug(f"YT run  {self.uuid}: Episode {i + 1} complete")

            rw_avg_reward = np.average(rw_rewards)
            apt_avg_reward = np.average(apt_rewards)
            blue_avg_reward = np.average(blue_rewards)
            print(f"Training complete!\n"
                  f"Average ransomware reward: {rw_avg_reward:.2f} \t Average APT reward: {apt_avg_reward:.2f} \t "
                  f"Average blue reward: {blue_avg_reward:.2f}")
            self.logger.debug(f"YT run  {self.uuid}: Agent training complete")
            return self.red_rw_agent, self.red_apt_agent, self.blue_agent
        else:
            self.logger.error(
                f"Cannot train the agent for YT run  {self.uuid} as the run has not been setup. "
                f"Call .setup() on the instance of {self.__class__.__name__} to setup the run."
            )

    def save(self) -> Union[str, None]:
        save_path = "./saved_models"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = f"{save_path}/{self.uuid}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        blue_path = f"{model_path}/blue.pth"
        red_path = f"{model_path}/red.pth"
        self.blue_agent.save(blue_path)
        self.red_agent.save(red_path)

    def multi_type_save(self) -> Union[str, None]:
        save_path = "./saved_models"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = f"{save_path}/{self.uuid}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        blue_path = f"{model_path}/blue.pth"
        apt_path = f"{model_path}/apt.pth"
        rw_path = f"{model_path}/ransomware.pth"
        self.blue_agent.save(blue_path)
        self.red_apt_agent.save(apt_path)
        self.red_rw_agent.save(rw_path)
        print(f"Saved models to {model_path}")


if __name__ == "__main__":
    gdb = GameModeDB()
    ndb = NetworkDB()
    game_mode = gdb.get("26edf1f7-c71d-4564-89d8-0eeee1659afc")
    network = ndb.get("3b921390-cd7b-41c5-8120-5e9ac587d2f2")
    network_interface = NetworkInterface(game_mode, network)
    red = AdaptiveRed
    blue = BlueInterface
    yt_run = MultiAgentYTRun(network=network, game_mode=game_mode, red_agent_class=red, blue_agent_class=blue)
    yt_run.setup()
    yt_run.train()
    yt_run.save()
