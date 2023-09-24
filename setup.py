#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='spIsoNet',
    version=1.0,
    description='spIsoNet isotropic reconstruction',
    url='https://github.com/spIsoNet-cryoET/spIsoNet',
    license='MIT',
    packages=['spIsoNet'],
    package_dir={
        'spIsoNet': '.',
    },
    entry_points={
        "console_scripts": [
            "spisonet.py = spIsoNet.bin.isonet:main",
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
