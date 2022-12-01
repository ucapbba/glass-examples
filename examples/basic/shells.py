'''
Defining the simulation
=======================

This example demonstrates how matter shell definitions can be computed, saved,
and loaded.

The computational part is common to all simulations.  The saving and loading is
useful to accelerate repeated computations which use the same input parameters,
because the angular matter power spectrum for a fixed cosmology and matter
weight function can be saved and then loaded again.  This prevents re-running a
costly computation many times over when different models are being compared
further down the simulation pipeline.

Under the hood, the saving and loading is done very plainly using
:func:`numpy.savez` and :func:`numpy.load` for the given arrays.

'''


# %%
# Compute
# -------
# Here we define the shells for these examples, and use CAMB to compute the
# angular matter power spectra for the shell definitions.  We also compute a
# set of lensing weights, which need the cosmology object (``cosmo``).
#
# Afterwards, we can save the shell definition.  Most examples (and real
# simulations) can then be run from the saved definitions without needing to
# know about the cosmology again.

import os.path
import camb
from cosmology import Cosmology
import glass.matter
import glass.camb
import glass.lensing
import glass.user


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
lmax = 1000

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# get the cosmology from CAMB
cosmo = Cosmology.from_camb(pars)

# shells in redshift spacing
shells = glass.matter.redshift_shells(0., 1., dz=0.1)

# redshift weight function for matter
# CAMB requires linear ramp for low redshifts
mweights = glass.matter.redshift_weights(shells, zlin=0.1)

# compute angular matter power spectra with CAMB
cls = glass.camb.matter_cls(pars, lmax, mweights)

# compute lensing weights
lweights = glass.lensing.midpoint_weights(shells, mweights, cosmo)


# %%
# Saving
# ------
# We can save shell definitions to file.  The full set of saved entries are the
# shells (``shells``), matter weights (``mweights``), angular matter power
# spectra (``cls``), and lensing weights (``lweights``).
#
# Internally, the function uses numpy's ``savez``, so the output file will be
# given an ``.npz`` extension if it does not already have one.

# save the matter shell definition to file
# not all arguments need to be given
glass.user.save_shells('shells.npz', shells, mweights, cls, lweights)


# %%
# Loading
# -------
# Loading the shell definitions works in the same way.  Because not all entries
# have to be saved, some of the return values can be ``None``.  The number and
# order of returned values is of course always the same.

# load previously saved shell definitions (normally in another file)
shells, mweights, cls, lweights = glass.user.load_shells('shells.npz')
