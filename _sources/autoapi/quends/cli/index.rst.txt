quends.cli
==========

.. py:module:: quends.cli

.. autoapi-nested-parse::

   Minimal command-line interface for QUENDS.

   Currently supports reporting the version and loading + summarizing a single
   variable from a data file. Extend the subcommands as the package grows.

   Usage
   -----
       python -m quends --version
       python -m quends summary <file> <variable>



Functions
---------

.. autoapisummary::

   quends.cli.build_parser
   quends.cli.main


Module Contents
---------------

.. py:function:: build_parser()

.. py:function:: main(argv=None)

