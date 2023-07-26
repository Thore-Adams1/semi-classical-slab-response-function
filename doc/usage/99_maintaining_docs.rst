Writing Documentation
=====================

This documentation is generated using `sphinx <https://www.sphinx-doc.org/en/master/>`_.
The docs are automatically built and published to GitHub Pages on a merge into the 
`main <https://www.sphinx-doc.org/en/master/>`_ branch, using GitHub actions.

These pages are written using ReStructured Text (RST). Here's a 
`quick primer <https://learnxinyminutes.com/docs/rst/>`_ on the syntax, and 
a `more thorough page <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ 
from the sphinx documentation.

API Documentation
-----------------

The API documentation is generated using a ``sphinx-apidoc``, an extension which
parsers the ``scsr`` python package and builds api docs from the functions, pulling
details from type annotations and docstrings. We use ``sphinx.ext.napolean`` 
(`docs <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_) to
interpret the docstrings, which expects them to use
`Google-style formatting <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

All modules within ``scsr`` (excluding ``scsr.cli``) will have documentation built
automatically.

CLI Documentation
-----------------

We use ``sphinx-argparse`` to create the documentation for the executables. We've 
configured it such that it expects markdown-formatted parser documentation 
(as RST-formatted text would make the ``--help`` text look less readable).
Here's a quick primer for 
`writing markdown <https://learnxinyminutes.com/docs/markdown/>`_.

All exectutable files will be documented if they have a submodule within 
``scsr/cli`` (and each must have a ``get_parser()`` method which returns an 
``argparse.ArgumentParser``)

Building the Documentation (Locally)
------------------------------------

When updating the docs, it's important to build them locally to make the building 
process still works, and to be sure any changes are formatted correctly.

The first time you do this, you'll need to install the documentation-specific
dependencies, which can be done using:

.. code-block:: console
    
    $ pip install -r requirements-docs.txt

Then you can build by running:

.. code-block:: console

    $ python doc/build.py && sphinx-build doc doc/_build

``doc/build.py`` creates the template ``RST`` files for the API & CLI docs within
``./doc``, and then ``sphix-build`` generates the html pages.

To preview the documenation, open ``./doc/_build/index.html`` in a web browser.

