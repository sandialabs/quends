========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |python-tests| |deployment| |coveralls|
    * - package
      - |commits-since|

.. |docs| image:: https://readthedocs.org/projects/quends/badge/?style=flat
    :target: https://readthedocs.org/projects/quends/
    :alt: Documentation Status

.. |python-tests| image:: https://github.com/sandialabs/quends/actions/workflows/python-tests.yml/badge.svg
    :alt: Python Tests Build Status
    :target: https://github.com/sandialabs/quends/actions/workflows/python-tests.yml

.. |deployment| image:: https://github.com/sandialabs/quends/actions/workflows/deployment.yml/badge.svg
    :alt: Deployment Status
    :target: https://github.com/sandialabs/quends/actions/workflows/deployment.yml

.. |coveralls| image:: https://coveralls.io/repos/github/sandialabs/quends/badge.svg?branch=main
    :target: https://coveralls.io/github/sandialabs/quends?branch=main
    :alt: Coverage Status

.. |commits-since| image:: https://img.shields.io/github/commits-since/sandialabs/quends/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/sandialabs/quends/compare/v0.0.0...main

.. end-badges

Quantification of Uncertainties in ENsembles of Data Streams

* Free software: BSD 3-Clause License

Installation
============

To install the package, you can choose one of the following methods:

1. **Clone the repository**:

   Using SSH:

       git clone git@github.com:sandialabs/quends.git
       cd quends

   Using HTTPS:

       git clone https://github.com/sandialabs/quends.git
       cd quends

2. **Install the package and its dependencies**:

   Once you have cloned the repository, you can install the package using pip:

       pip install .

3. **Verify the installation**:

   To ensure that the installation was successful, you can run a simple test:

       python -c "import quends"

Documentation
=============

https://quends.readthedocs.io/

Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

Coverage
========

You can view the coverage report `here`_.

.. _here: https://sandialabs.github.io/quends/coverage/index.html
