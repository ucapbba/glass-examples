'''
Galaxy shear
============

This example simulates a galaxy catalogue with shears affected by weak lensing,
combining the :doc:`/basic/plot_density` and :doc:`/basic/plot_lensing` examples
with generators for the intrinsic galaxy ellipticity and the resulting shear.

'''

# %%
# Setup
# -----
# The setup of galaxies and weak lensing fields is the same as in the basic
# examples.  We reuse the shell definitions from the :doc:`/basic/shells`
# example, but we also set up a matching CAMB cosmology to obtain the ``cosmo``
# object.
#
# In addition, there is a generator for intrinsic galaxy ellipticities,
# following a normal distribution.  The standard deviation is much too small to
# be realistic, but enables the example to get away with fewer total galaxies.
#
# Finally, there is a generator that applies the reduced shear from the lensing
# maps to the intrinsic ellipticities, producing the galaxy shears.

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

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# use CAMB cosmology in GLASS
cosmo = Cosmology.from_camb(pars)

# a redshift grid used below in a number of places
z = np.linspace(0, 1, 101)

# load matter shell definition
shells, mweights, cls, lweights = glass.user.load_shells('../basic/shells.npz')

# the intrinsic galaxy ellipticity
# this is very small so that the galaxy density can be small, too
sigma_e = 0.01

# galaxy density
n_arcmin2 = 0.01

# localised redshift distribution with the given density
dndz = np.exp(-(z - 0.5)**2/(0.1)**2)
dndz *= n_arcmin2/np.trapz(dndz, z)

# galaxy density in each shell
ngal = glass.galaxies.densities_from_dndz(z, dndz, shells)

# generators for lensing and galaxies
generators = [
    glass.matter.gen_lognormal_matter(cls, nside, ncorr=2),
    glass.lensing.gen_convergence(lweights),
    glass.lensing.gen_shear(),
    glass.galaxies.gen_uniform_positions(ngal),
    glass.galaxies.gen_redshifts_from_nz(z, dndz, shells),
    glass.galaxies.gen_ellip_gaussian(sigma_e),
    glass.galaxies.gen_shear_interp(cosmo),
]

# values we want from the simulation
yields = [
    glass.galaxies.GAL_LON,
    glass.galaxies.GAL_LAT,
    glass.galaxies.GAL_SHE,
]


# %%
# Simulation
# ----------
# Simulate the galaxies with shears.  In each iteration, get the shears and map
# them to a HEALPix map for later analysis.

# map for sum of shears
she = np.zeros(hp.nside2npix(nside), dtype=complex)

# keep count of total number of galaxies
num = np.zeros_like(she, dtype=int)

# iterate and map the galaxy shears to a HEALPix map
for gal_lon, gal_lat, gal_she in glass.core.generate(generators, yields):
    gal_pix = hp.ang2pix(nside, gal_lon, gal_lat, lonlat=True)
    s = np.argsort(gal_pix)
    pix, start, count = np.unique(gal_pix[s], return_index=True, return_counts=True)
    she[pix] += list(map(np.sum, np.split(gal_she[s], start[1:])))
    num[pix] += count


# %%
# Analysis
# --------
# Compute the angular power spectrum of the observed galaxy shears.  To compare
# with the expectation, take into account the expected noise level due to shape
# noise, and the expected mixing matrix for a uniform distribution of points.

# get the angular power spectra from the galaxy shears
cls = hp.anafast([num, she.real, she.imag], pol=True, lmax=lmax, use_pixel_weights=True)

# get the theory cls from CAMB
pars.NonLinear = 'NonLinear_both'
pars.Want_CMB = False
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# factor transforming convergence to shear
l = np.arange(lmax+1)
fl = (l+2)*(l+1)*l*(l-1)/np.clip(l**2*(l+1)**2, 1, None)

# number of arcmin2 in sphere
ARCMIN2_SPHERE = 60**6//100/np.pi

# will need number of pixels in map for the expectation
npix = len(she)

# compute the mean number of shears per pixel
nbar = ARCMIN2_SPHERE/npix*n_arcmin2

# the noise level from discrete observations with shape noise
nl = 4*np.pi*nbar/npix*sigma_e**2 * (l >= 2)

# mixing matrix for uniform distribution of points
mm = (nbar**2 - nbar/(npix-1))*np.eye(lmax+1, lmax+1) + (2*l+1)*nbar/(npix-1)/2
mm[:2, :] = mm[:, :2] = 0

# plot the realised and expected cls
plt.plot(l, cls[1] - nl, '-k', lw=2, label='simulation')
plt.plot(l, mm@(fl*theory_cls['W1xW1']), '-r', lw=2, label='expectation')
plt.xscale('symlog', linthresh=10, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.yscale('symlog', linthresh=1e-9, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel('angular mode number $l$')
plt.ylabel('angular power spectrum $C_l^{EE}$')
plt.legend()
plt.tight_layout()
plt.show()
