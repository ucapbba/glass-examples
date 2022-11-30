'''
Generator groups
================

This example demonstrates how generators can be grouped.  Groups provide a
layered namespace for outputs.  This is useful if similar generators are run
for different populations of objects, since it removes the need to manually
change input and output names.

'''

# %%
# Setup
# -----
# The simplest galaxies-only GLASS simulation, sampling galaxies uniformly over
# the sphere using some redshift distribution.  Galaxies are sampled in two
# groups: low and high redshifts.

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

# set up the matter shell boundaries
shells = glass.matter.redshift_shells(0., 3., dz=0.25)

# parametric galaxy redshift distribution: one for low-z, one for high-z
z = np.linspace(0, 3, 301)
dndz_low = n_arcmin2*glass.observations.smail_nz(z, 0.5, 1.0, 2.5)
dndz_high = n_arcmin2*glass.observations.smail_nz(z, 2.0, 4.0, 2.5)

# compute galaxy densities from dndz
dens_low = glass.galaxies.densities_from_dndz(z, dndz_low, shells)
dens_high = glass.galaxies.densities_from_dndz(z, dndz_high, shells)

# generators for a uniform galaxies simulation in two groups
# we need to sample positions because that Poisson-samples the galaxy number
generators = [
    glass.core.group('low-z', [
        glass.galaxies.gen_uniform_positions(dens_low),
        glass.galaxies.gen_redshifts_from_nz(z, dndz_low, shells),
    ]),
    glass.core.group('high-z', [
        glass.galaxies.gen_uniform_positions(dens_high),
        glass.galaxies.gen_redshifts_from_nz(z, dndz_high, shells),
    ]),
]


# %%
# Simulation
# ----------
# Keep the simulated redshifts of both populations.  Note how the groups provide
# a nested namespace for the data.

# arrays for true (ztrue) and photmetric (zphot) redshifts
low_z = np.empty(0)
high_z = np.empty(0)

# simulate and add galaxies in each matter shell to arrays
for shell in glass.core.generate(generators):
    low_z = np.append(low_z, shell['low-z'][glass.galaxies.GAL_Z])
    high_z = np.append(high_z, shell['high-z'][glass.galaxies.GAL_Z])


# %%
# Plots
# -----
# Plot the two distributions together with the expected inputs.

norm = glass.util.ARCMIN2_SPHERE*(z[-1] - z[0])/40

for zz, nz, label in (low_z, dndz_low, 'low-z'), (high_z, dndz_high, 'high-z'):
    plt.hist(zz, bins=40, range=(z[0], z[-1]), histtype='stepfilled', alpha=0.5, label=label)
    plt.plot(z, norm*nz, '-k', lw=1, alpha=0.5)
plt.xlabel('redshift $z$')
plt.ylabel('number of galaxies')
plt.legend()
plt.show()
