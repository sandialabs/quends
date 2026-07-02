========
Overview
========

.. start-badges

|maintained| |python-tests| |deployment| |python| |license| |black| |coveralls|

.. |maintained| image:: https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg
    :target: https://github.com/sandialabs/quends/graphs/commit-activity
    :alt: Maintained

.. |python-tests| image:: https://github.com/sandialabs/quends/actions/workflows/python-tests.yml/badge.svg
    :target: https://github.com/sandialabs/quends/actions/workflows/python-tests.yml
    :alt: Run Tests

.. |deployment| image:: https://github.com/sandialabs/quends/actions/workflows/deployment.yml/badge.svg
    :target: https://sandialabs.github.io/quends/
    :alt: Documentation

.. |python| image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
    :target: https://www.python.org/
    :alt: Python 3.9+

.. |license| image:: https://img.shields.io/badge/license-BSD--3--Clause-green.svg
    :target: https://github.com/sandialabs/quends/blob/main/LICENSE
    :alt: License: BSD-3-Clause

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. |coveralls| image:: https://coveralls.io/repos/github/sandialabs/quends/badge.svg?branch=main
    :target: https://coveralls.io/github/sandialabs/quends?branch=main
    :alt: Coverage Status

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

https://sandialabs.github.io/quends/

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

You can view the coverage report |coveralls|.
