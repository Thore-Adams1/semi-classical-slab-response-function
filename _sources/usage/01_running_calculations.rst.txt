.. Super handy cheat-sheet for writing RST: 
..   https://learnxinyminutes.com/docs/rst

Running Calculations
====================

To run calculations use the ``bin/scsr-calc`` script.

.. The below directive will auto-generate some docs from the --help of the 
.. script

.. argparse::
   :module: scsr.cli.calc
   :func: get_parser
   :prog: scsr-calc