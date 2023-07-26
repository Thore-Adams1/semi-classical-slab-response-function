Running Tests
=============

To run some tests to compare against the original matlab logic, first set 
the following environment variables:

* ``$MATLAB_EXECUTABLE``: The path to the matlab executable on your machine.
  Examples: ``'/usr/local/bin/matlab'`` or ``'C:/Program Files/MATLAB/R2022a/bin/matlab.exe'``
* ``$MATLAB_SCRIPTS_PATH``: The path to directory containing the original logic for
  this code. This directory must include ``get_matelement.m`` and dependent functions.
  
Then, to run the tests:

.. code::

    $ test/test_thesis_code_against_matlab.sh

