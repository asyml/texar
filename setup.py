#
"""
Setup file.
"""

import setuptools

setuptools.setup(
    name="txtgen",
    version="0.0.1",
    url="https://github.com/ZhitingHu/txtgen",

    description="An open and flexible framework for text generation.",

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=['tensorflow==1.1.0'],

    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
