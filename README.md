[//]: # (Image References)

[image1]: images/agent.gif "Trained Agent"

# Banana Collector Navigation using DQN 

### Introduction

An Implementation of Deep Q-Network Reinforcement Learning algorithm in Unity's Banana Collector environment.

![Trained Agent][image1]

### Environment

The environment is a simulation of a large square world with blue and yellow bananas placed randomly.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

| Environment Properties         |  |
|-------------------|------------------------|
| Environment       | Unity Banana Collector |
| Type              | Episodic Task          |
| Trials            | 100                    |
| Observation Space | Box(37,)               |
| Action Space      | Discrete(4,)           |
| Reward            | (-inf, inf)            |
| Required Score    | 13                     |
| Trials            | 100                    |
| Solved in         | 776 episodes           |

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  

Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

To consider the environment as solved, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Clone/Download this repository 
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Place the file in the root of this repository folder, and unzip (or decompress) the file. 

4. Navigate to python folder and execute the following command to install the dependencies 
`pip install .`
5. You can now open the `Navigation.ipynb` Notebook and execute the cells.

### Run Instructions

Once the dependencies mentioned in the previous section are fullfilled, start an instance of `jupyter notebook` and open `Navigation.ipynb` notebook and run all code blocks to train and watch the agent perform!  

