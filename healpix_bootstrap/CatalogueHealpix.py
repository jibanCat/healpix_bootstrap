'''
CatalogueHealpix.py : Transforming a full-sky catalogue (with a single feature)
                      to a HEALPix array
'''
import numpy as np
import healpy as hp 

class CatalogueHealpix(object):
    '''
    Feed in coordinates (longitudes, latitudes) and build a HEALPix map with bootstrap errors
    for each HEALPix pixels.

    Note: No special treatments with RA, DEC. Keep in mind the coordinates you choose.

    Parameters:
    ----
    lons (np.ndarray) : 1d array represents the longitudes of each rows in the catalogue.
    lats (np.ndarray) : 1d array represents the latitudes  of each rows in the catalogue.
    observed_features (np.ndarray) : 1d array with the same length of lons and lats, 
        represents the columns in your catalogue you want to transform to a HEALPix array.
    '''
    def __init__(self, lons, lats, observed_features):
        self.lons = lons
        self.lats = lats
        self.observed_features = observed_features


    def get_healpix(self, nside):
        '''
        return a HEALPix array (medians, bootstrap errors) based on a given nside
        '''
        # get the HEALPix index with given longitudes and latitudes
        # nest : whether you want your HEALPix ordering is hierarchical structured (NEST) or
        #   counting down from the top of the sphere (RING). You may want to check the healpy
        #   document, it's quite interesting.
        # lonlat : if you turn on this arg, the function will treat `theta` as longitude and `phi` as latitude.
        # ipix : an array of index of a HEALPix grid (with a given nside) while each number 
        #   in the ipix corresponds to the index on the HEALPix array the catalogue coordinate, 
        #   (lons, lats), should be mapped to.
        ipix = hp.ang2pix(nside, self.lons, self.lats, nest=False, lonlat=True)

        # initialize a HEALPix grid
        healpix = np.empty(hp.nside2npix(nside))
        errors  = np.empty(hp.nside2npix(nside))

        healpix[:] = np.nan
        errors[:]  = np.nan

        for i in range(ipix.max() + 1):
            inds = (ipix == i)

            # only calcuate the HEALPix grids with more than on row of data falling in
            if inds.sum() > 0:
                this_feature_bin = self.observed_features[inds]

                healpix[i] = np.nanmedian(this_feature_bin)
                errors[i]  = self.bootstrap_error(this_feature_bin)

        self.healpix = healpix
        self.errors  = errors

        return healpix, errors

    def healpix_realization(self, num_realizations=1000):
        '''
        return a HEALPix realization based on given healpix medians and 1-sigma errors
        '''
        return np.random.normal(
            loc=self.healpix, scale=self.errors, size=(num_realizations, self.healpix.shape[0]))

    @staticmethod
    def bootstrap_error(array, iterations=1000):
        '''
        bootstrap error for a given 1-D array
        '''
        randinds = np.random.randint(0, high=array.shape[0], size=(iterations, array.shape[0]))

        # resampling
        resampled_vals = array[randinds]
        
        resampled_error = np.sum( 
            (np.nanmedian(resampled_vals, axis=1) - np.nanmedian(resampled_vals))**2.
            ) / iterations
        resampled_error = np.sqrt(resampled_error)

        return resampled_error
