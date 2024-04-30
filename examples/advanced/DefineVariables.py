'''
Define Variables
=======================

To avoid large example files filled with variable definitions we define all relevant variables
for the advanced exampels here

'''


import numpy as np
import camb
import glass.ext.camb
from cosmology import Cosmology
from glass.fields import generate_lognormal, lognormal_gls
from glass.lensing import MultiPlaneConvergence
from glass.shells import distance_grid, tophat_windows
from glass.observations import gaussian_nz, smail_nz, equal_dens_zbins


def get_common_data():
    global lmax, nside, ncorr, rng, n_arcmin2
    lmax = nside = 256
    ncorr = 3
    # photometric redshift distribution
    gal_mean_z = [0.5, 1.0]
    gal_sigma_z = 0.125
    # total number of galaxies per arcmin2 in each bin
    gal_dens = 2.0
    # galaxy number density in units of galaxies/arcmin2/dz
    z = np.linspace(0., 2., 100)
    dndz = gaussian_nz(z, gal_mean_z, gal_sigma_z, norm=gal_dens)
    # ellipticity distribution standard deviation
    sigma_e = 0.26
    # photometric redshift error
    sigma_z0 = 0.03
    # galaxy bias < 1 so that we can compute expectation from theory
    beff = 0.9
    nbin = len(dndz)
    # labels for tomographic bins
    binlabels = np.arange(nbin)
    # name of the catalog we will create
    catalogName = "myCatalog.FITS"
    # random number generator, fix seed to make reproducible
    rng = np.random.default_rng()
    # galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
    n_arcmin2 = 0.3
    return lmax, nside, sigma_e, sigma_z0, beff, binlabels, catalogName, rng, n_arcmin2


def get_glass_data(*, loadCls: bool = False):
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
    zgrid = distance_grid(cosmo, 0., 3., dx=200.)
    # shells from windows
    shells = tophat_windows(zgrid)

    ws = glass.shells.tophat_windows(zgrid, weight=glass.ext.camb.camb_tophat_weight)
    if loadCls is True:
        print("Loading Cls")
        cls = np.load('../basic/cls.npy')
    else:
        print("Calculating Cls (Consider saving and loading them)")
        cls = glass.ext.camb.matter_cls(pars, lmax, ws)

    gls = lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=ncorr)
    # assert len(gls) == len(shells) * (len(shells) + 1) // 2
    # footprint

    print("simulating fields using NSIDE = ", str(nside))
    # generator for lognormal fields
    matter = generate_lognormal(gls, nside, ncorr=ncorr, rng=rng)
    # iterative computation of convergence
    convergence = MultiPlaneConvergence(cosmo)
    # true redshift distribution following a Smail distribution
    z = np.arange(0., 3., 0.01)
    dndz = smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
    dndz *= n_arcmin2
    # compute tomographic redshift bin edges with equal density
    nbins = 10
    zbins = equal_dens_zbins(z, dndz, nbins=nbins)
    return pars, matter, shells, convergence, ws, z, dndz, zbins
