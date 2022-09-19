from setuptools import find_packages, setup

setup(
    name='efa_utils',
    packages=find_packages(include=['efa_utils']),
    version='0.1',
    description='Custom utility functions for exploratory factor analysis',
    author='Dr. Marcel Wiechmann',
    license='GPLv3',
    install_requires=['numpy', 'pandas', 'factor_analyzer'],
)