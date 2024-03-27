#!/usr/bin/env python

from setuptools import setup, find_packages
print(find_packages())
setup(name='spIsoNet',
    version=1.0,
    description='spIsoNet isotropic reconstruction',
    url='https://github.com/spIsoNet-cryoET/spIsoNet',
    license='MIT',
    packages=find_packages(),#['spIsoNet'],
    #package_dir={
    #    'spIsoNet': '..',
    #},
    entry_points={
        "console_scripts": [
            "spisonet.py = spIsoNet.bin.spisonet:main",
        ],
    },
    include_package_data = True,
    install_requires=[
        #'torch==2.2.1',
        'mrcfile',
        'matplotlib',
        'fire',
        'scikit-image',
        'tqdm',
    ],
    #dependency_links=[
    #    'https://download.pytorch.org/whl/cu118'
    #]
)
