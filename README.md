# RL_Based_Obstacle_Avoidance

Reinforcement learning based obstacle avoidance for Pesticidrones with limited Agility

|Project Explaination Video|
|:------------:|
|[Youtube Link](https://youtu.be/ZfBmw0Nrzvg)|

## **Abstract -**

This project was undertaken during my internship at Technology and Innovation Hub (TiH-IOT)
at IIT Bombay and Eyantra , aimed to research, develop, implement, and test the feasi-
bility of a high-level velocity controller for a pesticide-spraying drone. The
primary challenge was to create an efficient control system for a drone with
limited maneuverability, designed for economic viability and extended flight
time.
The project faced several key constraints:
1. The drone’s movement was restricted to a 2D plane due to the lack of one
degree of freedom.
2. The algorithm needed to be computationally light.
3. It had to function as a high-level controller, providing only linear and
angular velocity values for obstacle avoidance.
4. The system needed to integrate with the existing low-level controller
(Ardupilot autopilot).
5. The drone was equipped with a pseudo-LIDAR system with an 86-degree
field of view.

Since the drone moves on a 2D plane on the advice of our mentors we, chose to test the fesability of t
these methods a differential drive based ground
robot called the Turtle Bot 3 (Burger Model)

We have used custom reward functions , observation space and action space in order to better mimic the movement of the 
drone where flight time was the major goal along with coverage of the agriculture field while tracing the contours of the obstacles.

## Challenges Faced- 

• Simulator selection:
  - Chose Gazebo Classic for environment resetting
  - Advanced simulators like Flightmare or Isaac Sim could yield better results

• Reward function tuning:
  - Robot exploited loopholes (e.g., moving in circles)
  - Undermined training objectives

• Neural network creation and hyperparameter tuning:
  - Complex and time-consuming
  - Balancing exploration vs. exploitation rates
  - Determining appropriate discount factor, batch size, and timesteps per episode

• Hardware limitations:
  - Lack of high-performance GPU
  - Training sessions exceeded 6 hours
  - Hindered convergence of reward function and loss function stabilization



