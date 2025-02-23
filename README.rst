========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |commits-since|
.. |docs| image:: https://readthedocs.org/projects/quends/badge/?style=flat
    :target: https://readthedocs.org/projects/quends/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/sandialabs/quends/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/sandialabs/quends/actions

.. |codecov| image:: https://codecov.io/gh/sandialabs/quends/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/sandialabs/quends

.. |commits-since| image:: https://img.shields.io/github/commits-since/sandialabs/quends/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/sandialabs/quends/compare/v0.0.0...main



.. end-badges

Quantification of Uncertainties in ENsembles of Data Streams

* Free software: MIT license

Installation
============

::

    pip install quends

You can also install the in-development version with::

    pip install https://github.com/sandialabs/quends/archive/main.zip


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
