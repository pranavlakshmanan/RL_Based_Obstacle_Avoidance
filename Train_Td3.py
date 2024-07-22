# TD3 (Twin Delayed Deep Deterministic Policy Gradient) Implementation for Robot Training in Gazebo
#
# This script implements the TD3 algorithm to train a robot in a Gazebo simulation environment.
# TD3 is an actor-critic method that addresses the overestimation bias problem in DDPG by using two critics.
#
# Key components:
# - Actor network: Determines the best action for a given state
# - Two Critic networks: Estimate Q-values (expected future rewards) for state-action pairs
# - Target networks: Slowly updated versions of actor and critic networks for stability
# - Replay buffer: Stores experiences for off-policy learning
#
# Important hyperparameters:
# - tau: Controls the speed of target network updates
# - policy_noise: Magnitude of noise added for target policy smoothing
# - noise_clip: Limits the magnitude of the added noise
# - policy_freq: Frequency of actor network updates relative to critic updates

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayBuffer
from Gazebo_vel import GazeboEnv # This Gazebo_vel file is the place where the Td3 algorithm tranining actually
#gets visualised in Gazebo and Rviz using ROS topics

# Evaluation function to assess the performance of the network
def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward}, {avg_col}")
    print("..............................................")
    return avg_reward

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Q-value estimator
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)
        # Second Q-value estimator
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # First Q-value
        s1 = F.relu(self.layer_1(s))
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)
        # Second Q-value
        s2 = F.relu(self.layer_4(s))
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# TD3 Algorithm
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99999,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # Sample a batch from the replay buffer
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Get next action and add noise for smoothing
            next_action = self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Compute current Q-value estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss and optimize
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state))[0].mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += critic_loss

        self.iter_count += 1
        # Log training statistics
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, filename)))
        self.critic.load_state_dict(torch.load("%s/%s_critic.pth" % (directory, filename)))

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0  # Random seed for reproducibility
eval_freq = 5e3  # Frequency of evaluation
max_ep = 2000  # Maximum number of steps per episode
eval_ep = 10  # Number of episodes for evaluation
max_timesteps = 1e6  # Maximum number of training timesteps
expl_noise = 0.8  # Initial exploration noise
expl_decay_steps = 2000000  # Steps over which exploration noise decays
expl_min = 0.3  # Minimum exploration noise
batch_size = 256  # Size of each training batch
discount = 0.98  # Discount factor for future rewards
tau = 0.01  # Soft target update rate
policy_noise = 0.4  # Noise added to target policy during critic update
noise_clip = 0.8  # Range to clip target policy noise
policy_freq = 2  # Frequency of delayed policy updates
buffer_size = 1e6  # Size of the replay buffer
file_name = "TD3_velodyne"  # Name for saving the model
save_model = True
load_model = False
random_near_obstacle = True  # Whether to take random actions near obstacles

# Create necessary directories
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Initialize the environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

# Set random seeds
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize TD3 agent, replay buffer, and related variables
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1
network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")

evaluations = []
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Main training loop
while timestep < max_timesteps:
    if done:
        # Train the network if it's not the first timestep
        if timestep != 0:
            network.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

        # Evaluate the network periodically
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        # Reset the environment for a new episode
        state = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Decrease exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # Select action with exploration noise
    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

    # Optionally force random actions near obstacles
    if random_near_obstacle:
        if np.random.uniform(0, 1) > 0.5 and min(state[4:-8]) < 1.0 and count_rand_actions < 1:
            count_rand_actions = np.random.randint(15, 30)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = max(0, action[0])

    # Execute action in the environment
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Store experience in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update state and counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# Final evaluation and model saving
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)
