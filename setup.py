#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='IsoNet',
    version=0.3,
    description='IsoNet isotropic reconstruction',
    url='https://github.com/Heng-Z/IsoNet',
    license='MIT',
    packages=['IsoNet'],
    package_dir={
        'IsoNet': '.',
    },
    entry_points={
        "console_scripts": [
            "isonet.py = IsoNet.bin.isonet:main",
        ],
    },
    include_package_data = True,
    install_requires=[
        'torch==1.12.1',
        'pytorch_lightning',
        'mrcfile',
        'matplotlib',
        'fire',
        'scikit-image==0.17.2',
        'tqdm',
    ]
)
