'''
FITS file creation
=======================

This example demonstrates how to use the cls geenrated from shells.py an
create a FITS file with 

'''

from itertools import count

import numpy as np
import camb
from cosmology import Cosmology
from glass.fields import generate_lognormal, lognormal_gls
from glass.galaxies import galaxy_shear
from glass.lensing import MultiPlaneConvergence, from_convergence
from glass.points import positions_from_delta
from glass.shapes import ellipticity_intnorm
from glass.shells import distance_grid, partition, tophat_windows
from glass.observations import gaussian_nz
from glass.user import awrite


from pprint import pprint

def info(itemName, item):
    
    if(isinstance(item, int)==False):
        try:
            print(itemName)
            print("type = " + str(type(item))  + ' ' + str(type(item[0]))) 
            print('shape, max, min = ' + str(item.shape) + ", "+ str(item.max()) + ", " + str(item.min()))
        except:
            print(itemName + 'len = ' + str(len(item)))
        finally:
            print("-------------------------") 
    else:
        print(itemName + ' = ' + str(item))

lmax = 500
nside = 512
ncorr = 3

# photometric redshift distribution
gal_mean_z = [0.5, 1.0]
gal_sigma_z = 0.125

# total number of galaxies per arcmin2 in each bin
gal_dens = 2.0

# galaxy number density in units of galaxies/arcmin2/dz
z = np.linspace(0., 2., 1000) #what's going on here? 
info('z', z)
dndz = gaussian_nz(z, gal_mean_z, gal_sigma_z, norm=gal_dens)
info('dndz', dndz)
# ellipticity distribution standard deviation
sigma_e = 0.26

# galaxy bias < 1 so that we can compute expectation from theory
beff = 0.9


nbin = len(dndz)
info('nbin', nbin)

# labels for tomographic bins
binlabels = np.arange(nbin)
info('binlabels', binlabels)

#name of the catalog we will create
catalog="myCatalog.FITS"

# random number generator, fix seed to make reproducible
rng = np.random.default_rng()

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2,
                       NonLinear=camb.model.NonLinear_both)

# get the cosmology from CAMB
cosmo = Cosmology.from_camb(pars)

# shells of 200 Mpc in comoving distance spacing
zgrid = distance_grid(cosmo, 0., 1. , dx=200.)

# shells from windows
shells = tophat_windows(zgrid)

# partition the galaxy distribution over shells
ngal = partition(z, dndz, shells)

cls = np.load('cls.npy')
gls = lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=ncorr)
assert len(gls) == len(shells) * (len(shells) + 1) // 2

# footprint
fp = None #load_fp(nside)

print("simulating fields using NSIDE=%d", nside)

# generator for lognormal fields
matter = generate_lognormal(gls, nside, ncorr=ncorr, rng=rng)

# iterative computation of convergence
convergence = MultiPlaneConvergence(cosmo)

# keep track of number of galaxies sampled
sum_n = np.zeros(dndz.shape[:-1], dtype=int)

# open FITS file for asynchronous writing
with awrite(catalog, ext="CATALOG") as out:

    # go through the shells and simulate
    for i, delta, shell in zip(count(), matter, shells):
        
        print("shell ", i)
        print("galaxy density: ", ngal[i])

        convergence.add_window(delta, shell)
        kappa = convergence.kappa

        gamma, = from_convergence(kappa, lmax, shear=True)

        galaxies = positions_from_delta(ngal[i], delta, beff, fp, rng=rng)

        for gal_lon, gal_lat, gal_count in galaxies:

            print("galaxies sampled: ", gal_count)

            # tomo bin id is predetermined by the population
            gal_b = np.repeat(binlabels, gal_count)

            gal_eps = ellipticity_intnorm(gal_count, sigma_e, rng=rng)

            gal_she = galaxy_shear(gal_lon, gal_lat, gal_eps,
                                   kappa, gamma.real, gamma.imag,
                                   reduced_shear=False)

            # some variance in shear weights
            gal_w = 10**rng.uniform(-2, 2, size=gal_she.shape)

            sum_n += gal_count

            out.write(
                RA=gal_lon,
                DEC=gal_lat,
                E1=gal_she.real,
                E2=gal_she.imag,
                W=gal_w,
                BIN=gal_b,
            )

print("FINISHED, total number of galaxies: ", sum_n)
