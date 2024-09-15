#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup


d = generate_distutils_setup(
    packages=['pose_estimator'],
    package_dir={'': 'src'},
    scripts=['scripts/pose_estimator_node']
)

setup(**d)
