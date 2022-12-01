'''
Weak lensing
============

This example computes weak lensing maps (convergence and shear) for a redshift
distribution of sources.  The lensing is simulated by a line of sight
integration of the matter fields.

'''

# %%
# Setup
# -----
#
# Simulate the matter fields, and compute the lensing fields from them.
#
# To obtain the effective integrated lensing maps of a distribution of sources,
# the :func:`glass.lensing.lensing_dist` generator will iteratively collect and
# integrate the contributions from each shell.
#
# We need the cosmology object (``cosmo``) for the lensing interpolation, so we
# use CAMB to get it back (even though we load the precomputed shell definitions
# from the :doc:`/basic/shells` example).  We also use CAMB below to compute the
# theory lensing spectra.

import os.path
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and everything in the glass namespace
from cosmology import Cosmology
import glass.all
import glass

# also needs camb itself to get the parameter object, and the expectation
import camb


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
nside = 512
lmax = nside

# localised redshift distribution
z = np.linspace(0, 1, 101)
nz = np.exp(-(z - 0.5)**2/(0.1)**2)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# use CAMB cosmology in GLASS
cosmo = Cosmology.from_camb(pars)

# load precomputed shell definition
shells, mweights, cls, lweights = glass.user.load_shells('shells.npz')

# generators for a lensing-only simulation with three correlated matter shells
generators = [
    glass.matter.gen_lognormal_matter(cls, nside, ncorr=3),
    glass.lensing.gen_convergence(lweights),
    glass.lensing.gen_shear(),
    glass.lensing.gen_lensing_dist(z, nz, cosmo),
]

# variables we will use for plotting
yields = [
    glass.lensing.KAPPA_BAR,
    glass.lensing.GAMMA_BAR,
]


# %%
# Simulation
# ----------
# The simulation is then straightforward:  Only the integrated lensing maps are
# stored here.  While the simulation returns the result after every redshift
# interval in the light cone, only the last result will be show below, so the
# previous values are not kept.

# simulate and store the integrated lensing maps
# note that we merely keep kappa, gamma1, gamma2 for after the loop
for kappa, (gamma1, gamma2) in glass.core.generate(generators, yields):
    pass


# %%
# Analysis
# --------
# To make sure the simulation works, compute the angular power spectrum of the
# simulated convergence field, and compare with the expectation (from CAMB) for
# the given redshift distribution of sources.
#
# We are not doing the modelling very carefully here, so a bit of discrepancy is
# to be expected.

# get the angular power spectra of the lensing maps
sim_cls = hp.anafast([kappa, gamma1, gamma2], pol=True, lmax=lmax, use_pixel_weights=True)

# get the expected cls from CAMB
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [camb.sources.SplinedSourceWindow(z=z, W=nz, source_type='lensing')]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# plot the realised and expected cls
l = np.arange(lmax+1)
plt.plot(l, (2*l+1)*sim_cls[0], '-k', lw=2, label='simulation')
plt.plot(l, (2*l+1)*theory_cls['W1xW1'], '-r', lw=2, label='expectation')
plt.xscale('symlog', linthresh=10, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.yscale('symlog', linthresh=1e-7, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel(r'angular mode number $l$')
plt.ylabel(r'angular power spectrum $(2l+1) \, C_l^{\kappa\kappa}$')
plt.legend()
plt.show()
