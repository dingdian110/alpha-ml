import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'scikit-learn',
    'emcee',
    'scipy >= 0.12',
    'numpy >= 1.7'
    ]

setup(name='alpha-ml',
      version='0.1.0',
      description='AutoML toolkit',
      author='Thomas Young',
      author_email='liyang.cs@pku.edu.cn',
      url='https://github.com/thomas-young-2013/',
      keywords='AutoML',
      packages=find_packages(),
      license='LICENSE.txt',
      test_suite='nose.collector',
      install_requires=requires)
