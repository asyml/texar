# Open library of text generation #

# Code style

  * [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

  * [Pylint: code analysis for Python](https://www.pylint.org)

    - [Pylint intro (in Chinese)](https://www.ibm.com/developerworks/cn/linux/l-cn-pylint/index.html)

    - example cmd: 
      ``` bash
      pylint --reports=n --include-ids=y --rcfile=.pylintrc foo.py
      ```

# Reference libraries

  * [seq2seq](https://github.com/google/seq2seq)

  * [tensorforce](https://github.com/reinforceio/tensorforce)

  * [sonnet](https://github.com/deepmind/sonnet)

  * [tensorflow](https://github.com/tensorflow/tensorflow)

## View Documentation
TxtGen uses [Sphinx](http://www.sphinx-doc.org/en/stable/index.html)
to automatically generate beautiful documentations. TxtGen assumes the
code comments are in [Google Python DocString style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Install Sphinx:

    apt-get install python-sphinx

Generate HTML:

    cd ./doc
    make html

View HTML

    open ./doc/_build/index.html

# TODO
    
  * New name of the project ?
