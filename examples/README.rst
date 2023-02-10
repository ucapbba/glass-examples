
Examples for *GLASS*
====================

These examples show how `GLASS`__, the Generator for Large Scale Structure, can
be used in practice.  They are often a good starting point for more complicated
and realistic simulations.

__ https://glass.readthedocs.io

To run the examples yourself, you need to have GLASS installed.  The latest
release is available via pip::

    $ pip install glass

The examples currently require `CAMB`__ to produce angular matter power spectra
and for the cosmological background.  Make sure you have CAMB installed::

    $ python -c 'import camb'  # should not give an error

If you want to compute the angular matter power spectra in the examples, you
need the `glass-camb` package::

    $ pip install glass-camb

__ https://camb.readthedocs.io/
