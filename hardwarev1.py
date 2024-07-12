import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math

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

class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
    
    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()
    
    def load(self, filename, directory):
         self.actor.load_state_dict(
             torch.load("%s/%s_actor.pth" % (directory, filename), map_location=torch.device('cpu'))
         )

class TurtleBotController:
    def __init__(self):
        rospy.init_node('turtlebot_controller', anonymous=True)
        
        self.state_dim = 24  # 20 laser readings + 4 odometry values
        self.action_dim = 2
        self.network = TD3(self.state_dim, self.action_dim)
        self.network.load("TD3_velodyne", "./pytorch_models")
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        self.odom_data = None
        self.laser_data = None
        
        self.rate = rospy.Rate(10)  # 10 Hz
        
        # Set target 20 units ahead in a straight line
        self.target = Point()
        self.target.x = 5.0
        self.target.y = 5.0
        self.target.z = 0.0
        
        self.min_obstacle_distance = 0.5  # Minimum allowed distance to obstacles
    
    def odom_callback(self, data):
        self.odom_data = data
    
    def laser_callback(self, data):
        self.laser_data = data
    
    def get_state(self):
        if self.odom_data is None or self.laser_data is None:
            return None
        
        # Extract relevant information from odometry
        pos_x = self.odom_data.pose.pose.position.x
        pos_y = self.odom_data.pose.pose.position.y
        orient_z = self.odom_data.pose.pose.orientation.z
        orient_w = self.odom_data.pose.pose.orientation.w
        
        # Sample 20 equally spaced laser readings
        laser_readings = np.array(self.laser_data.ranges)
        sampled_readings = laser_readings[::len(laser_readings)//20][:20]
        
        # Clip laser readings to ensure minimum obstacle distance
        sampled_readings = np.clip(sampled_readings, self.min_obstacle_distance, np.inf)
        
        # Neural network state (odometry + laser readings)
        nn_state = np.concatenate(([pos_x, pos_y, orient_z, orient_w], sampled_readings))
        
        # Calculate target information (not used in NN state, but used for control)
        dx = self.target.x - pos_x
        dy = self.target.y - pos_y
        distance_to_target = math.sqrt(dx*dx + dy*dy)
        angle_to_target = math.atan2(dy, dx) - 2 * math.atan2(orient_z, orient_w)
        
        return nn_state, distance_to_target, angle_to_target
    
    def run(self):
        while not rospy.is_shutdown():
            state_info = self.get_state()
            if state_info is not None:
                nn_state, distance_to_target, angle_to_target = state_info
                action = self.network.get_action(nn_state)
                
                # Convert action to Twist message
                twist_msg = Twist()
                twist_msg.linear.x = (action[0] + 1) / 2  # Scale to [0, 1]
                twist_msg.angular.z = action[1]  # Already in [-1, 1]
                
                # Adjust action based on target information
                if distance_to_target < 0.1:  # If close to target
                    twist_msg.linear.x = 0
                    twist_msg.angular.z = 0
                    self.cmd_vel_pub.publish(twist_msg)
                    rospy.loginfo("Target reached!")
                    break
                elif distance_to_target < 1.0:  # If approaching target
                    twist_msg.linear.x *= 0.5  # Slow down
                
                # Adjust orientation towards target
                if abs(angle_to_target) > 0.1:
                    twist_msg.angular.z += np.clip(angle_to_target, -0.5, 0.5)
                
                # Check for potential collisions
                if np.min(nn_state[-20:]) <= self.min_obstacle_distance:
                    twist_msg.linear.x = 0
                    twist_msg.angular.z = 0.5  # Turn in place to avoid obstacle
                
                self.cmd_vel_pub.publish(twist_msg)
            
            self.rate.sleep()

if __name__ == '__main__':
    controller = TurtleBotController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass