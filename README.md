# Open library of text generation #

# Install

  * Run the following: 
    ```bash
    pip install [--user] -e .    
    ```

  * After installation, the library is ready to use. E.g., 
    - Run the example code in `./examples`:
      ```bash
      python ./examples/language_model.py
      ```
    - Run unit tests, e.g.,
      ```bash
      python ./txtgen/modules/encoders/rnn_encoders_test.py
      ```

# Code style

  * Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
    
    - E.g., Maximum line length is *80 characters*.

  * Use [Pylint](https://www.pylint.org) for automatic code style check

    - [Pylint intro (in Chinese)](https://www.ibm.com/developerworks/cn/linux/l-cn-pylint/index.html)

    - Setup Pylint in your IDE, or use the cmd like this: 
      ``` bash
      pylint --reports=n --include-ids=y --rcfile=.pylintrc foo.py
      ```
  * Follow [Google Python DocString style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 
  for code comments. 

    - [Pytorch documentation](http://pytorch.org/docs/master/nn.html#parameters) 
    gives a great example of writing docstrings

## View Documentation
TxtGen uses [Sphinx](http://www.sphinx-doc.org/en/stable/index.html)
to automatically generate beautiful documentations. 

Install Sphinx:

    apt-get install python-sphinx

Generate HTML:

    cd ./doc
    make html

View HTML

    open ./doc/_build/index.html

# Reference libraries

  * [seq2seq](https://github.com/google/seq2seq)

  * [tensorforce](https://github.com/reinforceio/tensorforce)

  * [sonnet](https://github.com/deepmind/sonnet)

  * [tensorflow](https://github.com/tensorflow/tensorflow)


# TODO
    
  * New name of the project ?
