import numpy as np
import healpy as hp

def healpix_hist2d(pinpoint, nside, weight):
    '''
    Input pinpoint class and nside with weights, 
    return a 2d histogram on Healpix.
    
    Parameters:
    ----------
    pinpoint : class, 
        Pinpoint class, see Pinpoint.py.
    nside : int, 
        resolution in healpix grids, mean( objects / pixel ) ~ 10
    weight : 1d numpy array, 
        the weight you want to put on each objects while you are counting the histogram.
        
    Return:
    ------
    healpix : 1d array, 
        Healpix map.
    '''
    # extract the galactic coor and weight factor
    l, b = pinpoint.rot_9090[:,0], pinpoint.rot_9090[:,1]

    # instantiate a Healpix Grid
    healpix = np.zeros(hp.nside2npix(nside))

    # unit convert, phi=[0,2pi], theta=[0,pi]
    phi   = l * np.pi / 180
    theta = (90 - b) * np.pi / 180

    # add back to the healpix grids
    ipix = hp.ang2pix(nside, theta=theta, phi=phi, lonlat=0)
    for pix,w in zip(ipix, weight):
        healpix[pix] += 1 * w
        
    return healpix

def pinpoint_feature_maps(pinpoint, df, nside_in=8, nside_out=128):
    '''
    Input Pinpoint class with histogram nside input, 
    return 2 healpix feature maps in I and D.
    
    Parameters:
    ----------
    pinpoint : class, 
        Pinpoint class, see Pinpoint.py
    df : pd.DataFrame, 
        DataFrame in the Pinpoint class. 
    nisde_in : int, 
        the nside of healpix you want to use to count the objects.  mean( objects / pixel ) ~ 10
    nside_out : int, 
        output healpix feature maps' nside. choose whatever you want. 
        
    Returns:
    -------
    I_map, D_map: tuple of 2 1d numpy arrays, 
        I feature map, D feature map.
    '''
    # generate 3 types of count maps
    count_map   = healpix_hist2d(pinpoint, nside_in, np.ones(len(df)))
    count_I_map = healpix_hist2d(pinpoint, nside_in, 
                                 weight=df.shift_sum.values)
    count_D_map = healpix_hist2d(pinpoint, nside_in, 
                                 weight=df.shift_var_distance.values)

    # Normalize the maps
    I_map = count_I_map / count_map
    D_map = count_D_map / count_map

    # rotate
    I_map = pinpoint.healpix_rotate(
        pinpoint.healpix_rotate(I_map, rot=(-90,0,0)), rot=(0,-90,0))
    D_map = pinpoint.healpix_rotate(
        pinpoint.healpix_rotate(D_map, rot=(-90,0)), rot=(0,-90))

    # repixelize to large nside 
    alm_I = hp.map2alm(I_map)
    alm_D = hp.map2alm(D_map)
    I_map = hp.alm2map(alm_I, nside=nside_out, verbose=0)
    D_map = hp.alm2map(alm_D, nside=nside_out, verbose=0)
    return I_map, D_map

    