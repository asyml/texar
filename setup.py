import sys
import setuptools


long_description = '''
Texar is an open-source toolkit based on TensorFlow,
aiming to support a broad set of machine learning especially text generation
tasks, such as machine translation, dialog, summarization, content manipulation,
language modeling, and so on.

Texar is designed for both researchers and practitioners for fast prototyping
and experimentation. Checkout https://github.com/asyml/texar-pytorch for the
PyTorch version which has the same functionalities and (mostly) the same
interfaces.
'''

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required by Texar.')

setuptools.setup(
    name="texar",
    version="0.2.4",
    url="https://github.com/asyml/texar",

    description="Toolkit for Machine Learning and Text Generation",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'regex>=2018.01.10',
        'numpy<1.17.0',
        'pathlib>=1.0',
        'pyyaml',
        'requests',
        'funcsigs>=1.0.2',
        'sentencepiece>=0.1.8',
        'packaging'
    ],
    extras_require={
        'tensorflow-cpu': [
            'tensorflow>=1.10.0,<2.0',
            'tensorflow-probability>=0.3.0,<0.8.0'
        ],
        'tensorflow-gpu': [
            'tensorflow-gpu>=1.10.0,<2.0',
            'tensorflow-probability>=0.3.0,<0.8.0'
        ]
    },
    package_data={
        "texar": [
            "../bin/utils/multi-bleu.perl",
        ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
