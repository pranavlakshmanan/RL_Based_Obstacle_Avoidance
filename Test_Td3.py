#This code is used to test the TD3 network after it has been trained.
#Notice how there is no use of teh critic network here




import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Gazebo_vel import GazeboEnv

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Define the layers of the neural network
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, s):
        # Forward pass through the network
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# TD3 (Twin Delayed Deep Deterministic Policy Gradient) network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
    
    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available, otherwise CPU
seed = 0  # Random seed number for reproducibility
max_ep = 500  # Maximum number of steps per episode
file_name = "TD3_velodyne"  # Name of the file to load the policy from

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)  # Wait for the environment to initialize

# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Set up the state and action dimensions
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the TD3 network
network = TD3(state_dim, action_dim)

# Load the pre-trained model
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

# Initialize episode variables
done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop
while True:
    # Get action from the network
    action = network.get_action(np.array(state))
    
    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    
    # Take a step in the environment
    next_state, reward, done, target = env.step(a_in)
    print(next_state, reward, done, target)
    
    # Check if episode is done
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    
    # On termination of episode
    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
