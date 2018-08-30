#
"""
Setup file.
"""

import setuptools

setuptools.setup(
    name="texar",
    version="0.1",
    url="https://github.com/asyml/texar",

    description="Toolkit for Text Generation and Beyond",

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'numpy',
        'pyyaml',
        'requests',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.7.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.7.0']
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
