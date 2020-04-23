#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "SIMBA relies on Safety Gym which is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='simba',
    packages=['simba'],
    install_requires=[
        'safety-gym==0.0.0',
        'gym~=0.15.3',
        'joblib==0.14.0',
        'matplotlib==3.1.1',
        'mpi4py==3.0.2',
        'mujoco_py==2.0.2.7',
        'numpy~=1.17.4',
        'seaborn==0.8.1',
        'tensorflow>=1.13.1',
        'pyyaml~=5.3.1',
        'tensorboardx==1.8',
        'scipy~=1.4.1'
    ]
)
