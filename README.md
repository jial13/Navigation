# Project : Navigation

##Description

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world. 

##Project environment details

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions.

Four discrete actions are available, corresponding to:

`0` - move forward

`1` - move backward

`2` - turn left

`3` - turn right

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

##Files

- `Navigation.ipynb`: Notebook used to control and train the agent 
- `dqn_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `model.py`: Q-network class used to map state to action values 
- `report.pdf`: Technical report


##Dependencies

Dependencies are listed in `requirements.txt` file

install using the following command: 
```
pip install requirements.txt
``` 

You will also need to download the Unity environment, which can be downloaded at:

Linux: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

Mac OSX: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Windows (32-bit): [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

Windows (64-bit): [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

##Running
Run all of the cells in the notebook `Navigation.ipynb`