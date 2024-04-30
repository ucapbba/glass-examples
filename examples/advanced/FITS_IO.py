'''
FITS I/O example
================

This example creates a galaxy catalogue from a Stage IV Space Satellite Galaxy
Survey such as *Euclid* and *Roman* combining the :doc:`/basic/plot_density` and
:doc:`/basic/plot_lensing` examples with galaxy ellipticities and galaxy shears,
as well as using some auxiliary functions.

The focus in this example is mock catalogue generation using auxiliary functions
built for simulating Stage IV galaxy surveys.

The catalog is then read and the redshifts plotted

'''
from itertools import count
import numpy as np
from glass.galaxies import galaxy_shear, redshifts_from_nz, gaussian_phz
from glass.points import positions_from_delta
from glass.shapes import ellipticity_intnorm
from glass.lensing import from_convergence
from glass.user import write_context
from glass.shells import restrict
from glass.observations import vmap_galactic_ecliptic, tomo_nz_gausserr
from DefineVariables import get_common_data, get_glass_data
import healpy as hp
import matplotlib.pyplot as plt
import fitsio


# Import common variables from create_FITS_helper module
lmax, nside, sigma_e, sigma_z0, beff, binlabels, catalogName, rng, n_arcmin2 = get_common_data()
pars, matter, shells, convergence, ws, z, dndz, zbins = get_glass_data(loadCls=False)

# Make a visibility map typical of a space telescope survey, seeing both
# hemispheres, and low visibility in the galactic and ecliptic bands.
vis = vmap_galactic_ecliptic(nside)
# checking the mask:
hp.mollview(vis, title='Stage IV Space Survey-like Mask', unit='Visibility')
plt.show()

print("Creating the catalog " + catalogName)
# open a glass defined write context with a HDU extension name 'CATALOG'
with write_context(catalogName, ext="CATALOG") as out:
    # go through the shells and simulate
    for i, delta, shell in zip(count(), matter, shells):
        z_i, dndz_i = restrict(z, dndz, ws[i])
        ngal = np.trapz(dndz_i, z_i)
        print("shell ", i)
        print("galaxy density: ", ngal)
        convergence.add_window(delta, shell)
        kappa = convergence.kappa
        gamma, = from_convergence(kappa, lmax, shear=True)

        for gal_lon, gal_lat, gal_count in positions_from_delta(ngal, delta, beff, vis):
            print("galaxies sampled: ", gal_count)
            gal_eps = ellipticity_intnorm(gal_count, sigma_e, rng=rng)
            gal_she = galaxy_shear(gal_lon, gal_lat, gal_eps,
                                   kappa, gamma.real, gamma.imag,
                                   reduced_shear=False)
            # some variance in shear weights
            gal_w = 10**rng.uniform(-2, 2, size=gal_she.shape)
            gal_z = redshifts_from_nz(gal_count, z_i, dndz_i)
            # generator photometric redshifts using a Gaussian model
            gal_phz = gaussian_phz(gal_z, sigma_z0)
            # attach tomographic bin IDs to galaxies, based on photometric redshifts
            gal_zbin = np.digitize(gal_phz, np.unique(zbins)) - 1

            out.write(
                RA=gal_lon,
                DEC=gal_lat,
                E1=gal_she.real,
                E2=gal_she.imag,
                W=gal_w,
                Z_TRUE=gal_z,
                PHZ=gal_phz,
                ZBIN=gal_zbin,
            )

# split dndz using the same Gaussian error model assumed in the sampling
tomo_nz = tomo_nz_gausserr(z, dndz, sigma_z0, zbins)
nbins = 10
print("Reading the catalog " + catalogName)
with fitsio.FITS(catalogName, "r") as fits:
    data = fits[1].read()
    plt.figure()
    plt.title('redshifts in catalogue')
    plt.ylabel('dN/dz - normalised')
    plt.xlabel('z')
    for i in range(nbins):
        in_bin = (data['ZBIN'] == i)
        plt.hist(data['Z_TRUE'][in_bin], histtype='stepfilled', edgecolor='none', alpha=0.5, bins=50, density=1, label=f'cat. bin {i}')
    for i in range(nbins):
        plt.plot(z, (tomo_nz[i]/n_arcmin2)*nbins, alpha=0.5, label=f'inp. bin {i}')
    plt.plot(z, dndz/n_arcmin2*nbins, ls='--', c='k')
    plt.legend(ncol=2)
    plt.show()
