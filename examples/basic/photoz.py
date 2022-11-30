'''
Photometric redshifts
=====================

This example simulates galaxies with a simple photometric redshift model.

'''

# %%
# Setup
# -----
# The simplest galaxies-only GLASS simulation, sampling galaxies using some
# redshift distribution.  Then add a model for photometric redshifts with
# Gaussian errors.

import numpy as np
import matplotlib.pyplot as plt

# import everything in the glass namespace
import glass.all
import glass


# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 1e-4

# photometric redshift error at redshift 0
phz_sigma_0 = 0.05

# matter shell definition, uniform thickness in redshift
shells = glass.matter.redshift_shells(0., 3., dz=0.25)

# parametric galaxy redshift distribution
z = np.linspace(0, 3, 301)
dndz = n_arcmin2*glass.observations.smail_nz(z, 1.0, 2.2, 1.5)

# compute the galaxy density in each shell
ngal = glass.galaxies.densities_from_dndz(z, dndz, shells)

# generators for a uniform galaxies simulation
generators = [
    glass.galaxies.gen_uniform_positions(ngal),
    glass.galaxies.gen_redshifts_from_nz(z, dndz, shells),
    glass.galaxies.gen_phz_gausserr(phz_sigma_0),
]


# %%
# Simulation
# ----------
# Keep all simulated true and photometric redshifts.

# arrays for true (ztrue) and photmetric (zphot) redshifts
ztrue = np.empty(0)
zphot = np.empty(0)

# simulate and add galaxies in each matter shell to arrays
for shell in glass.core.generate(generators):
    ztrue = np.append(ztrue, shell[glass.galaxies.GAL_Z])
    zphot = np.append(zphot, shell[glass.galaxies.GAL_PHZ])


# %%
# Plots
# -----
# Make a couple of typical photometric redshift plots.
#
# First the :math:`z`-vs-:math:`z` plot across the entire sample.  The simple
# Gaussian error model only has the diagonal but no catastrophic outliers.

plt.figure(figsize=(5, 5))
plt.plot(ztrue, zphot, '+k', ms=3, alpha=0.1)
plt.xlabel(r'$z_{\rm true}$', size=12)
plt.ylabel(r'$z_{\rm phot}$', size=12)
plt.show()

# %%
# Now define a number of photometric redshift bins.  They are chosen by the
# :func:`~glass.observations.equal_dens_zbins` function to produce the same
# number of galaxies in each bin.

nbins = 5
zbins = glass.observations.equal_dens_zbins(z, dndz, nbins)

# %%
# After the photometric bins are defined, make histograms of the *true* redshift
# distribution :math:`n(z)` using the *photometric* redshifts for binning.  Use
# the :func:`~glass.observations.tomo_nz_gausserr()` function to also plot the
# expected tomographic redshift distributions with the same model.

tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, phz_sigma_0, zbins)
tomo_nz *= glass.util.ARCMIN2_SPHERE*(z[-1] - z[0])/40

for (z1, z2), nz in zip(zbins, tomo_nz):
    plt.hist(ztrue[(z1 <= zphot) & (zphot < z2)], bins=40, range=(z[0], z[-1]),
             histtype='stepfilled', alpha=0.5)
    plt.plot(z, nz, '-k', lw=1, alpha=0.5)
plt.xlabel('true redshift $z$')
plt.ylabel('number of galaxies')
plt.show()
