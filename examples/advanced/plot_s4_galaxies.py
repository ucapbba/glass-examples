'''
Stage IV Galaxy Survey
======================

This example simulates a galaxy catalogue from a Stage IV Space Satellite Galaxy
Survey such as *Euclid* and *Roman* combining the :doc:`/basic/plot_density` and
:doc:`/basic/plot_lensing` examples with generators for the intrinsic galaxy
ellipticity and the resulting shear with some auxiliary functions.

The focus in this example is mock catalogue generation using auxiliary functions
built for simulating Stage IV galaxy surveys.
'''

# %%
# Setup
# -----
# The basic setup of galaxies and weak lensing fields is the same as in the
# previous examples.
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

# these are the GLASS imports: cosmology and everything in the glass namespace
from cosmology import Cosmology
import glass.all
import glass

# also needs camb itself to get the parameter object
import camb


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
nside = 512
lmax = nside

# set up the random number generator
rng = np.random.default_rng(seed=42)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# use CAMB cosmology
cosmo = Cosmology.from_camb(pars)

# redshift grid that will be used for a number of continuous functions
z = np.linspace(0, 3, 1000)


# %%
# Set up the matter sector.

# use matter shells with 200 Mpc thickness in comoving distance
shells = glass.matter.distance_shells(cosmo, 0., 3., dx=200.)

# use a matter weight function that is uniform in redshift
mweights = glass.matter.redshift_weights(shells, zlin=0.1)

# compute the angular matter power spectra
cls = glass.camb.matter_cls(pars, lmax, mweights)


# %%
# Set up the lensing sector.

# compute the midpoint lensing weights
lweights = glass.lensing.midpoint_weights(shells, mweights, cosmo)


# %%
# Set up the galaxies sector.

# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# true redshift distribution following a Smail distribution
dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
dndz *= n_arcmin2

# compute the galaxy number density in each shell
ngal = glass.galaxies.densities_from_dndz(z, dndz, shells)

# compute bin edges with equal density
nbins = 10
zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)

# photometric redshift error
sigma_z0 = 0.03

# split distribution by tomographic bin, assuming photometric redshift errors
tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# constant bias parameter for all shells
b = 1.2

# sigma_ellipticity as expected for a Stage-IV survey
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

# generators for clustering and lensing
generators = [
    glass.matter.gen_lognormal_matter(cls, nside, ncorr=2, rng=rng),
    glass.lensing.gen_convergence(lweights),
    glass.lensing.gen_shear(),
    glass.observations.gen_constant_visibility(vis),
    glass.galaxies.gen_positions_from_matter(ngal, b, rng=rng),
    glass.galaxies.gen_redshifts_from_nz(z, tomo_nz, shells, rng=rng),
    glass.galaxies.gen_ellip_intnorm(sigma_e, rng=rng),
    glass.galaxies.gen_shear_interp(cosmo),
]

# values we want from the simulation
yields = [
    glass.galaxies.GAL_LON,
    glass.galaxies.GAL_LAT,
    glass.galaxies.GAL_Z,
    glass.galaxies.GAL_SHE,
    glass.galaxies.GAL_POP,
]

# we will store the catalogue as a dictionary
catalogue = {'RA': np.array([]), 'DEC': np.array([]), 'TRUE_Z': np.array([]),
             'G1': np.array([]), 'G2': np.array([]), 'TOMO_ID': np.array([])}

# iterate and store the quantities of interest for our mock catalogue
for gal_lon, gal_lat, gal_z, gal_she, gal_pop in glass.core.generate(generators, yields):
    # let's assume here that lon lat here are RA and DEC:
    catalogue['RA'] = np.append(catalogue['RA'], gal_lon)
    catalogue['DEC'] = np.append(catalogue['DEC'], gal_lat)
    catalogue['TRUE_Z'] = np.append(catalogue['TRUE_Z'], gal_z)
    catalogue['G1'] = np.append(catalogue['G1'], gal_she.real)
    catalogue['G2'] = np.append(catalogue['G2'], gal_she.imag)
    catalogue['TOMO_ID'] = np.append(catalogue['TOMO_ID'], gal_pop)

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
