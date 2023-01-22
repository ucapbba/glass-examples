'''
Stage IV Galaxy Survey
======================

This example simulates a galaxy catalogue from a Stage IV Space Satellite Galaxy
Survey such as *Euclid* and *Roman* combining the :doc:`/basic/plot_density` and
:doc:`/basic/plot_lensing` examples with galaxy ellipticities and galaxy shears,
as well as using some auxiliary functions.

The focus in this example is mock catalogue generation using auxiliary functions
built for simulating Stage IV galaxy surveys.

'''

# %%
# Setup
# -----
# The setup is essentially the same as in the :doc:`/advanced/plot_shears`
# example.
#
# In addition to a generator for intrinsic galaxy ellipticities,
# following a normal distribution, we also show how to use auxiliary functions
# to generate tomographic redshift distributions and visibility masks.
#
# Finally, there is a generator that applies the reduced shear from the lensing
# maps to the intrinsic ellipticities, producing the galaxy shears.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology import Cosmology

# GLASS modules: cosmology and everything in the glass namespace
import glass.fields
import glass.points
import glass.shapes
import glass.matter
import glass.lensing
import glass.galaxies
import glass.observations


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
nside = lmax = 256

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# get the cosmology from CAMB
cosmo = Cosmology.from_camb(pars)

# %%
# Set up the matter sector.

# shells of 200 Mpc in comoving distance spacing
shells = glass.matter.distance_shells(cosmo, 0., 3., dx=200.)

# uniform matter weight function
# CAMB requires linear ramp for low redshifts
weights = glass.matter.uniform_weights(shells, zlin=0.1)

# compute the angular matter power spectra of the shells with CAMB
cls = glass.camb.matter_cls(pars, lmax, weights)

# compute Gaussian cls for lognormal fields for 3 correlated shells
# putting nside here means that the HEALPix pixel window function is applied
gls = glass.fields.lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=3)

# generator for lognormal matter fields
matter = glass.fields.generate_lognormal(gls, nside, ncorr=3)

# %%
# Set up the lensing sector.

# compute the effective redshifts of the matter shells
# these will be the source redshifts of the lensing planes
zlens = glass.matter.effective_redshifts(weights)

# compute the multi-plane lensing weights for these redshifts
wlens = glass.lensing.multi_plane_weights(zlens, weights)

# this will compute the convergence field iteratively
convergence = glass.lensing.MultiPlaneConvergence(cosmo)

# %%
# Set up the galaxies sector.

# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# true redshift distribution following a Smail distribution
z = np.arange(0., 3., 0.01)
dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
dndz *= n_arcmin2

# compute the galaxy number density in each shell
ngal = glass.galaxies.density_from_dndz(z, dndz, bins=shells)

# compute bin edges with equal density
nbins = 10
zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)

# photometric redshift error
sigma_z0 = 0.03

# split distribution by tomographic bin, assuming photometric redshift errors
tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# constant bias parameter for all shells
bias = 1.2

# ellipticity standard deviation as expected for a Stage-IV survey
sigma_e = 0.27

# %%
# Plotting the overall redshift distribution and the
# distribution for each of the equal density tomographic bins

plt.figure()
plt.title('redshift distributions')
sum_nz = np.zeros_like(tomo_nz[0])
for nz in tomo_nz:
    plt.fill_between(z, nz, alpha=0.5)
    sum_nz = sum_nz + nz
plt.fill_between(z, dndz, alpha=0.2, label='dn/dz')
plt.plot(z, sum_nz, ls='--', label='sum of the bins')
plt.ylabel('dN/dz - gal/arcmin2')
plt.xlabel('z')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Make a visibility map typical of a space telescope survey, seeing both
# hemispheres, and low visibility in the galactic and ecliptic bands.
vis = glass.observations.vmap_galactic_ecliptic(nside)

# checking the mask:
hp.mollview(vis, title='Stage IV Space Survey-like Mask', unit='Visibility')
plt.show()


# %%
# Simulation
# ----------
# Simulate the galaxies with shears.  In each iteration, get the quantities of
# interest to build our mock catalogue.

# we will store the catalogue as a structured numpy array, initially empty
catalogue = np.empty(0, dtype=[('RA', float), ('DEC', float), ('TRUE_Z', float),
                               ('G1', float), ('G2', float), ('TOMO_ID', int)])

# simulate the matter fields in the main loop, and build up the catalogue
for i, delta_i in enumerate(matter):

    # boundary redshifts for this shell
    zmin, zmax = shells[i], shells[i+1]

    # compute the lensing maps for this shell
    convergence.add_plane(delta_i, zlens[i], wlens[i])
    kappa_i = convergence.kappa
    gamm1_i, gamm2_i = glass.lensing.shear_from_convergence(kappa_i)

    # generate galaxy positions from the matter density contrast
    gal_lon, gal_lat = glass.points.positions_from_delta(ngal[i], delta_i, bias, vis)

    # number of galaxies in this shell
    gal_siz = len(gal_lon)

    # generate random redshifts from the provided nz
    gal_z, gal_pop = glass.galaxies.redshifts_from_nz(gal_siz, z, tomo_nz,
                                                      zmin=zmin, zmax=zmax)

    # generate galaxy ellipticities from the chosen distribution
    gal_eps = glass.shapes.ellipticity_intnorm(gal_siz, sigma_e)

    # apply the shear fields to the ellipticities
    gal_she = glass.galaxies.galaxy_shear(gal_lon, gal_lat, gal_eps,
                                          kappa_i, gamm1_i, gamm2_i)

    # make a mini-catalogue for the new rows
    rows = np.empty(gal_siz, dtype=catalogue.dtype)
    rows['RA'] = gal_lon
    rows['DEC'] = gal_lat
    rows['TRUE_Z'] = gal_z
    rows['G1'] = gal_she.real
    rows['G2'] = gal_she.imag
    rows['TOMO_ID'] = gal_pop

    # add the new rows to the catalogue
    catalogue = np.append(catalogue, rows)

print(f'Total Number of galaxies sampled: {len(catalogue["TRUE_Z"]):,}')

# %%
# Catalogue checks
# ----------------
# Here we can perform some simple checks at the catalogue level to
# see how our simulation performed.

# redshift distribution of tomographic bins & input distributions
plt.figure()
plt.title('redshifts in catalogue')
plt.ylabel('dN/dz - normalised')
plt.xlabel('z')
for i in range(0, 10):
    plt.hist(catalogue['TRUE_Z'][catalogue['TOMO_ID'] == i], histtype='stepfilled', edgecolor='none', alpha=0.5, bins=50, density=1, label=f'cat. bin {i}')
for i in range(0, 10):
    plt.plot(z, (tomo_nz[i]/n_arcmin2)*nbins, alpha=0.5, label=f'inp. bin {i}')
plt.plot(z, dndz/n_arcmin2*nbins, ls='--', c='k')
plt.legend(ncol=2)
plt.show()
