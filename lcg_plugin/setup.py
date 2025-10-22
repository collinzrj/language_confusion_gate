# setup.py

from setuptools import setup

setup(
    name='lcg_plugin',
    version='0.1',
    packages=['lcg_plugin'],
    entry_points={
        'vllm.general_plugins': [
            'register_lcg = lcg_plugin:register_plugin'
        ]
    },
    install_requires=[
        'vllm',  # Ensure compatibility with installed vLLM version
    ],
)