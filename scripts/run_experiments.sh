#!/bin/bash
echo Starting experiments run...

echo

echo Running four seeds of Safe-CEM-MPC policy...
python train.py --name 1 --config_dir ../config/ --config_basename experiment.yaml \
 --log_level DEBUG --log_dir ../data/aware/ --seed 1 &
python train.py --name 123 --config_dir ../config/ --config_basename experiment.yaml \
 --log_level DEBUG --log_dir ../data/aware/ --seed 123 &
python train.py --name 456 --config_dir ../config/ --config_basename experiment.yaml \
 --log_level DEBUG --log_dir ../data/aware/ --seed 456 &
python train.py --name 789 --config_dir ../config/ --config_basename experiment.yaml \
 --log_level DEBUG --log_dir ../data/aware/ --seed 789 &

echo

echo Running four seeds of CEM-MPC policy...
python train.py --name 1 --config_dir ../config/ --config_basename experiment_unaware.yaml \
 --log_level DEBUG --log_dir ../data/unaware/ --seed 1 &
python train.py --name 123 --config_dir ../config/ --config_basename experiment_unaware.yaml \
 --log_level DEBUG --log_dir ../data/unaware/ --seed 123 &
python train.py --name 456 --config_dir ../config/ --config_basename experiment_unaware.yaml \
 --log_level DEBUG --log_dir ../data/unaware/ --seed 456 &
python train.py --name 789 --config_dir ../config/ --config_basename experiment_unaware.yaml \
 --log_level DEBUG --log_dir ../data/unaware/ --seed 789 &

echo Running four seeds of Safe-CEM-MPC policy without sampling...
python train.py --name 1 --config_dir ../config/ --config_basename experiment_no_sample.yaml \
 --log_level DEBUG --log_dir ../data/no_sample/ --seed 1 &
python train.py --name 123 --config_dir ../config/ --config_basename experiment_no_sample.yaml \
 --log_level DEBUG --log_dir ../data/no_sample/ --seed 123 &
python train.py --name 456 --config_dir ../config/ --config_basename experiment_no_sample.yaml \
 --log_level DEBUG --log_dir ../data/no_sample/ --seed 456 &
python train.py --name 789 --config_dir ../config/ --config_basename experiment_no_sample.yaml \
 --log_level DEBUG --log_dir ../data/no_sample/ --seed 789 &

wait

echo Experiments script finished.



