#This code is used to make the TD3 network training with the Gazebo environment and Rviz.
#It uses ros nodes and subscribers to do the same.
#At the end is the reward function for the actor.

# Import necessary libraries and modules
import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Constants for goal-reaching and collision detection
GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.35
TIME_DELTA = 0.1

# Function to check if the random goal position is located on an obstacle
def check_pos(x, y):
    goal_ok = True

    # Define obstacle regions where goal positions are not allowed
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok

# GazeboEnv class that sets up the Gazebo simulation environment and manages the robot's interactions with it
class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim  # Dimension of the environment
        self.odom_x = 0  # Robot's x position
        self.odom_y = 0  # Robot's y position

        self.goal_x = 1  # Goal x position
        self.goal_y = 0.0  # Goal y position

        self.upper = 5.0  # Upper bound for random goal generation
        self.lower = -5.0  # Lower bound for random goal generation
        self.velodyne_data = np.ones(self.environment_dim) * 10  # Placeholder for velodyne data
        self.last_odom = None  # Placeholder for the last odometry data

        self.set_self_state = ModelState()  # Initializing model state for the robot
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Define gaps for dividing the 360-degree field of view of the lidar
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim])
        self.gaps[-1][-1] += 0.03

        # Start roscore
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        # Initialize ROS node
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        # Launch the Gazebo simulation
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # ROS Publishers and Subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)

    # Read velodyne point cloud and convert it to distance data
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:  # Filter points based on height
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                # Assign distances to the respective gap ranges
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    # Callback function to update the last known odometry data
    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Unpause Gazebo to let the robot move
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # Propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        # Pause Gazebo after the action is performed
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        heading = math.atan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x)

        # Calculate state observation
        state = [distance, heading]
        state += laser_state[0].tolist()
        state += [vel_cmd.linear.x, vel_cmd.angular.z]

        # Check if the goal is reached
        if distance < GOAL_REACHED_DIST:
            done = True
            target = True

        return np.asarray(state), done, collision, target, vel_cmd

    # Function to observe collisions based on laser data
    def observe_collision(self, data):
        collision = False
        done = False
        min_laser = min(data)
        if min_laser < COLLISION_DIST:
            collision = True
            done = True
        return done, collision, min_laser

    # Function to reset the environment and robot's state
    def reset(self):
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.goal_x, self.goal_y = self.set_new_goal()
        self.publish_goal_marker()
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")

        time.sleep(1.0)
        self.set_self_state.pose.position.x = 0
        self.set_self_state.pose.position.y = 0
        self.set_self_state.pose.position.z = 0
        self.set_self_state.pose.orientation.x = 0
        self.set_self_state.pose.orientation.y = 0
        self.set_self_state.pose.orientation.z = 0
        self.set_self_state.pose.orientation.w = 1
        self.set_state.publish(self.set_self_state)
        time.sleep(1.0)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(1.0)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        while self.last_odom is None:
            time.sleep(0.1)

        return self.step([0, 0])[0]

    # Function to set a new random goal within the environment
    def set_new_goal(self):
        while True:
            x = random.uniform(self.lower, self.upper)
            y = random.uniform(self.lower, self.upper)
            if check_pos(x, y):
                return x, y

    # Function to publish goal markers for visualization
    def publish_goal_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

    # Function to publish action markers for visualization
    def publish_markers(self, action):
        # Linear velocity marker
        marker2 = Marker()
        marker2.header.frame_id = "base_link"
        marker2.type = marker2.ARROW
        marker2.action = marker2.ADD
        marker2.scale.x = action[0]
        marker2.scale.y = 0.1
        marker2.scale.z = 0.1
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 0
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0
        marker_array2 = MarkerArray()
        marker_array2.markers.append(marker2)
        self.publisher2.publish(marker_array2)

        # Angular velocity marker
        marker3 = Marker()
        marker3.header.frame_id = "base_link"
        marker3.type = marker3.ARROW
        marker3.action = marker3.ADD
        marker3.scale.x = 0.1
        marker3.scale.y = 0.1
        marker3.scale.z = 0.1
        marker3.color.a = 1.0
        marker3.color.r = 0.0
        marker3.color.g = 0.0
        marker3.color.b = 1.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 0
        marker3.pose.position.y = 0
        marker3.pose.position.z = 0
        marker_array3 = MarkerArray()
        marker_array3.markers.append(marker3)
        self.publisher3.publish(marker_array3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            # return 100.0
            return 150.0
        elif collision:
            # return -100.0
            return -150.0
        else:
            forward_reward = action[0]
        
        # Penalize turning, but less harshly
            turning_penalty = -abs(action[1])/2   
        
        # More nuanced distance-based reward
            distance_reward = 0
            if min_laser < 0.5:
               distance_reward = -2 * (0.5 - min_laser)  # Stronger penalty when very close
            elif min_laser < 1.0:
                distance_reward = min_laser - 0.5  # Positive reward for maintaining safe distance
        
        # Small constant reward for survival
            survival_reward = 0.1

            rotation_penalty = -abs(action[1]) * 1.5

            return forward_reward + turning_penalty + distance_reward + survival_reward + rotation_penalty
            # r3 = lambda x: 1 - x if x < 1 else 0.0
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            # return 100.0
            return 150.0
        elif collision:
            # return -100.0
            return -150.0
        else:
            forward_reward = action[0]
        
        # Penalize turning, but less harshly
            turning_penalty = -abs(action[1])/2   
        
        # More nuanced distance-based reward
            distance_reward = 0
            if min_laser < 0.5:
               distance_reward = -2 * (0.5 - min_laser)  # Stronger penalty when very close
            elif min_laser < 1.0:
                distance_reward = min_laser - 0.5  # Positive reward for maintaining safe distance
        
        # Small constant reward for survival
            survival_reward = 0.1

            rotation_penalty = -abs(action[1]) * 1.5

            return forward_reward + turning_penalty + distance_reward + survival_reward + rotation_penalty
            # r3 = lambda x: 1 - x if x < 1 else 0.0
            # return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
            # return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
