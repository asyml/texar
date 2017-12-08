#
"""
Setup file.
"""

import setuptools

setuptools.setup(
    name="texar",
    version="0.0.1",
    url="https://github.com/ZhitingHu/txtgen",

    description="An open and flexible framework for text generation.",

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'numpy'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4.0'],
        'tensorflow with gpu': ['tensorflow-gpu>=1.4.0']
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
