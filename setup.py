import setuptools


long_description = '''
Texar is an open-source toolkit based on Tensorflow,
aiming to support a broad set of machine learning especially text generation tasks,
such as machine translation, dialog, summarization, content manipulation, language modeling, and so on.

Texar is designed for both researchers and practitioners for fast prototyping and experimentation.
'''

setuptools.setup(
    name="texar",
    version="0.1",
    url="https://github.com/asyml/texar",

    description="Toolkit for Text Generation and Beyond",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'numpy',
        'pyyaml',
        'requests',
        'funcsigs',
    ],
    extras_require={
        'tensorflow-cpu': ['tensorflow>=1.7.0', 'tensorflow-probability >= 0.3.0'],
        'tensorflow-gpu': ['tensorflow-gpu>=1.7.0', 'tensorflow-probability-gpu >= 0.3.0']
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
