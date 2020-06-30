# Safe Model-Based Reinforcement Learning via Ensembles and Model Predictive Control

![Imgur](https://i.imgur.com/6F47oFY.gif)

This is an implementation of a model-based reinforcement learning agent that predicts the safety and scores of actions sequences by learning the environment's dynamics using an ensemble of neural networks. This algorithm achieves higher sample efficiency and comparable results to those reported in [Safety Gym](https://openai.com/blog/safety-gym/) in the PointGoal-v1 task.


## To Reproduce results
Install Safety Gym then clone and ```pip install .```, preferably within a conda or virtualenv python environment.

To run all of the experiments use
```
cd scripts
bash run_experiments.sh
```
Note that this command runs all 12 experiments at once, accordingly, please consider running each experiment seperately.

To visualize experiments results please run
```
python plot_results.py --data_path <your_data_path>
```

Hyper parameters and configuration can be found in the config directory.
