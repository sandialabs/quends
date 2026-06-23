======
QUENDS
======

|maintained| |python-tests| |deployment| |python| |license| |black| |coveralls|

**QUENDS** (Quantifying Uncertainty in ENsemble Data Streams) turns raw
time-series simulation output into trustworthy statistics: it trims transients,
detects steady state, and quantifies uncertainty for single signals and for
ensembles of runs.

Quickstart
==========

.. code-block:: bash

   pip install quends

.. code-block:: python

   import quends as qnds

   # load one signal (+ its time column), drop the warm-up, quantify
   ds = qnds.from_csv("output.csv", "Q_D/Q_GBD")
   trimmed = ds.trim(method="threshold", window_size=100, threshold=0.1)
   print(trimmed.compute_statistics())   # mean + uncertainty (effective sample size)

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started
      :link-type: doc

      Install QUENDS and run your first **load → trim → quantify** in minutes.

   .. grid-item-card:: 📖 User Guide
      :link: user_guide/index
      :link-type: doc

      The concepts and the recommended workflow for each part of the library.

   .. grid-item-card:: 🖼️ Gallery of examples
      :link: auto_tutorials/index
      :link-type: doc

      Runnable, end-to-end tutorials.

   .. grid-item-card:: 🧩 API Reference
      :link: autoapi/index
      :link-type: doc

      The full, auto-generated API.

.. toctree::
   :hidden:
   :caption: Home

   self

.. toctree::
   :hidden:
   :caption: Getting Started

   readme
   installation
   getting_started

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/index

.. toctree::
   :hidden:
   :caption: Gallery of examples

   auto_tutorials/index

.. toctree::
   :hidden:
   :caption: API Reference

   autoapi/index

.. toctree::
   :hidden:
   :caption: About

   usage
   contributing
   authors
   changelog


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
   :alt: Coverage
