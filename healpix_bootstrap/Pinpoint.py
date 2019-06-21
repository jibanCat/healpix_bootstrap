# Location Class
import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
import healpy as hp
import functools
from multiprocessing import Pool
from itertools import compress
from scipy import stats

try: 
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.despine(bottom=True)
    cold_cmap = sns.cubehelix_palette(18, start=8, rot=0.4, dark=0, light=0.99, as_cmap=True)
    divi_cmap = sns.diverging_palette(220, 50, s=80, l=40, sep=30, n=25, as_cmap=True)
except ImportError:
    cold_cmap = 'GnBu'
    divi_cmap = 'jet'

class Pinpoint:
    '''
    read data from data/Pinpoint/your_patches_generated_data/
    currently not available on Github. 

    Parameters:
    ---
    size_of_sampling: str. 'small' or 'large'
    type_of_map: str. 'smica' or 'nilc' or 'sevem' or 'commander' or 'lensed'
    '''    
    def __init__(self, size_of_sampling, type_of_map, size, data_path='', nside=2048):
        self.size_of_sampling = size_of_sampling
        self.type_of_map = type_of_map
        if 'lensed' in self.type_of_map: self.sub_folder = 'lensed'
        else: self.sub_folder = 'com'
        self.size = size
        self.nside = nside
        # i/o
        self.rot = np.loadtxt(os.path.join(data_path, 'data/Pinpoint', 
                                           self.sub_folder,
                                           self.size_of_sampling, 
                                           '.'.join(['rot',str(self.size),'txt'])))
        self.df = pd.read_csv(os.path.join(data_path, 'data/Pinpoint',
                              self.sub_folder, self.size_of_sampling, 
                              '.'.join(['df',self.type_of_map,str(self.size),'csv'])))
        with open(os.path.join(data_path, 'data/Pinpoint',
                  self.sub_folder, self.size_of_sampling, 
                  '.'.join(['dls',self.type_of_map,str(self.size),'pkl'])), 'rb') as fid:
            self.dls = pickle.load(fid)
        # preprocess : convert rot position back into (0,0) rotated map
        self.rot_9090 = self.rot
        ori_rot = self.ipix2ipix(
            self.rot2ipix(self.rot.T, self.nside), (90,90), self.nside)
        self.rot = np.stack((ori_rot[0], ori_rot[1]), axis=1)
    
    # location sampling function
    # visulization functions
    def plot_sky(self, return_map=0):
        '''
        return:
        Addition_map, D, I
        '''
        # normalisation base
        Addition_map, Weighted_map, Directed_map = self.plot_scanning_ADI(
            xsize=2520, nside=256)
        D = np.nan_to_num(Weighted_map / Addition_map)
        I = np.nan_to_num(Directed_map / Addition_map)
        ## filtering Gal nearby
        if return_map==1:
            return Addition_map, D, I
        else:
            hp.mollview(self.healpix_rotate(
                    self.healpix_rotate(D, rot=(-90, 0,0)), rot=(0,-90,0)), 
                        cmap=cold_cmap, title='Normalized Variance Map ($\ell$)')
            hp.mollview(self.healpix_rotate(
                    self.healpix_rotate(I, rot=(-90, 0,0)), rot=(0,-90,0)), 
                        cmap=divi_cmap, title='Normalized Sum of Peaks Map ($\ell$)')
            hp.orthview(self.healpix_rotate(I, rot=(90,0,0)), cmap=divi_cmap, )
            return 0
        
    # nearby pixels and statistical significance
    def nearby_pixs(self, rot, degree=None):
        '''
        rot: given rot location
        '''
        if not degree: degree = self.size / 4 * np.sqrt(2)
        ipixels = self.rot2ipix(self.rot.T, self.nside)
        base_vecs = np.array(hp.pix2vec(self.nside, ipixels)).T
        vec = np.array(hp.pix2vec(self.nside, self.rot2ipix(np.array(rot), self.nside)))
        # True if the pixs are within the circle centered by the given rot location
        boolean_list = self.return_vec(base_vecs, vec, degree)
        indices = list(compress(range(len(boolean_list)), boolean_list))
        return indices    

    def return_vec(self, base_vecs, vec, threshold_degree):
         return [np.dot(vec,vec2) > np.cos(np.pi / 180 * threshold_degree) 
                 for vec2 in base_vecs]
        
    def stat_significance(self, rot, degree=None, return_hist=1): 
        # for a CMB map
        if 'lensed' not in self.type_of_map:
            indices = self.nearby_pixs(rot, degree=degree)
            df = self.df.loc[indices, 'shift1':'shift_var_distance']
            # plt histogram for shift_sum and shift_var_distance
            if return_hist==1:
                print(df.describe()) # give general information
                self.ttest_ind(self.df, df, return_hist=return_hist, kwds='around ' + str(rot))            
        # for lensed maps: iterate 100 times
        else:
            print('lensed locations')
            stop = self.df.index[-1] + 1
            step = int((self.df.index[-1] + 1) // 100)
            temp = []
            # if len(rot) == 1, return only one position
            if len(rot) != 100 and np.array(rot).shape != (2,):
                print('len of rot is shorter than 100')
            elif len(rot) != 100 and np.array(rot).shape == (2,):
                print('len of rot is 1, use the same positions for all simulations')
                rot = np.outer(np.ones(100), rot)
            for i,r in zip(range(0, stop, step), rot):
                indices = np.array(
                    self.nearby_pixs(r, degree=degree)) + i # + i for change to i^th map
                df = self.df.loc[indices, :] 
                temp.append(df)
            df = pd.concat(temp, axis=0)
        return df
    
    def extreme_region(self, column, degree=None, argmax=1, return_hist=1):
        if not degree: degree = self.size / 2 * np.sqrt(2)
        # for components maps
        if 'lensed' not in self.type_of_map:
            if argmax==1: idx = self.df.loc[:, column].argmax() 
            else: idx = self.df.loc[:, column].argmin() 
            indices = self.nearby_pixs(self.rot[idx], degree=degree)
            df = self.df.loc[indices, :]
            print('output rot is {}'.format(self.rot[idx]))

        # for lensed maps: iterate 100 times
        else:
            stop = self.df.index[-1] + 1
            step = int((self.df.index[-1] + 1) // 100)
            if argmax==1:
                idxs = (self.df.loc[i:i + step - 1, column].argmax() 
                       for i in range(0, stop, step))
            else:
                idxs = (self.df.loc[i:i + step - 1, column].argmin() 
                       for i in range(0, stop, step))
            temp = []
            for i,idx in zip(range(0, stop, step), idxs):
                rot = self.rot[idx % step] # I used the same rot file for all 100 lensed maps
                indices = np.array(
                    self.nearby_pixs(rot, degree=degree)) + i # + i for change to i^th map
                df = self.df.loc[indices, :] 
                temp.append(df)
            df = pd.concat(temp, axis=0)
            del temp
        self.ttest_ind(self.df, df, return_hist=return_hist, kwds='extreme ' + column)        
        return df        
    
    def ttest_ind(self, df1, df2, return_hist=1, kwds=''):
        value, p_sum = stats.ttest_ind(df1.loc[:, 'shift_sum'], 
                                       df2.loc[:, 'shift_sum'], equal_var=1)
        value, p_var = stats.ttest_ind(df1.loc[:, 'shift_var_distance'], 
                                       df2.loc[:, 'shift_var_distance'], equal_var=1)
        print ('shift_sum p value is {}; shift_var_distance p value is {}'.format(
            p_sum, p_var))
        if return_hist==1:
            fig, ax = plt.subplots(1,2, figsize=(10, 5))
            xmin = df1.loc[:, 'shift_sum'].min()
            xmax = df1.loc[:, 'shift_sum'].max()
            H1 = ax[0].hist(df1.loc[:, 'shift_sum'], bins=25, log=1, 
                range=(xmin, xmax))
            _  = ax[0].hist(df2.loc[:, 'shift_sum'], bins=25, log=1, 
                label=kwds, range=(xmin, xmax))
            ax[0].set_title('Sum of Peaks')
            ax[0].set_ylim(1/10, 10**int(np.log10(max(H1[0])) + 1))
            ax[0].legend()
            xmin = df1.loc[:, 'shift_var_distance'].min()
            xmax = df1.loc[:, 'shift_var_distance'].max()
            H2 = ax[1].hist(df1.loc[:, 'shift_var_distance'], bins=25, log=1, 
                range=(xmin, xmax))
            _  = ax[1].hist(df2.loc[:, 'shift_var_distance'], bins=25, log=1, 
                label=kwds, range=(xmin, xmax))
            ax[1].set_title('Variance of Peaks')
            ax[1].set_ylim(1/10, 10**int(np.log10(max(H2[0])) + 1))
            ax[1].legend()        
        return 0
        
    # rot and ipix identification functions 
    # from rot=(90,90) rotated map, bacl into rot=(0,0) map
    def rot2ipix(self, rot, nside):
        phi, theta = rot.copy()
        theta = (90 - theta) * np.pi / 180
        phi  *= np.pi / 180
        ipix  = hp.ang2pix(nside, theta, phi)
        return ipix

    def ipix2ipix(self, ipix, rot, nside, verbose=0):
        theta, phi = self.pix2deg(nside, ipix)
        # convert position back to the original map coordinate
        vec      = hp.pix2vec(nside, ipix)
        rot_vec  = (hp.rotator.Rotator(rot=(rot[0],0,0))).I(vec)
        rot_vec  = (hp.rotator.Rotator(rot=(0,rot[1],0))).I(rot_vec)
        ori_ipix = hp.vec2pix(nside, rot_vec[0], rot_vec[1], rot_vec[2])
        ori_theta, ori_phi = self.pix2deg(nside, ori_ipix)
        ori_rot  = (ori_phi, ori_theta)
        if verbose==True:
            print ('{} convert to original coordinate {}.'.format((phi, theta), ori_rot))
        return ori_rot
    
    def pix2deg(self, nside, ipix):
        theta, phi = hp.pix2ang(nside, ipix)
        theta *= 180 / np.pi
        phi *= 180 / np.pi
        theta = 90 - theta
        return theta, phi
    
    # plotting weighted map functions
    def ones_gaussian(self, length, sigma):
        '''
        ----
        Parameters:
        length : int, length of the matrix.
        sigma : int, in degree, degree of gaussian kernel
        '''
        # force int
        length = int(length)

        # converting sigma in degree into sigma in matrix unit
        sigma = length * sigma / self.size 
        
        # making zeros matrix with lengths
        circle = np.zeros((length, length))
        y, x = np.indices((circle.shape))

        # finding center and making the radial r in 2d matrix
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        circle = np.exp(- r**2 / 2 / sigma**2)
        return circle
    
    def lonlat_block_shorten(self, rotate, circlelength, xsize=2520, nside=256):
        lon, lat = rotate
        cartview = np.zeros((int(xsize / 2), xsize))
        
        # center block
        circlelength = int(circlelength  * xsize // 360)
        cartview[int(xsize / 4 - circlelength / 2): int(xsize / 4 + circlelength / 2), 
                 int(xsize / 2 - circlelength / 2): int(xsize / 2 + circlelength / 2)] \
        = self.ones_gaussian(circlelength, sigma=self.size // 4) 
        
        # re-pixelization
        healpix1 = self.cart_healpix(cartview, nside)
        del cartview
        # cartview back to origin: theta 
        healpix2 = self.healpix_rotate(healpix1, (0,-lat))
        del healpix1
        return self.healpix_rotate(healpix2, (-lon,0))

    def plot_scanning(self, coor, weight, xsize=2520, nside=256):
        Addition_map = np.zeros(hp.nside2npix(nside), dtype=np.double)
        number = range(len(coor))
        partial = functools.partial(self.lonlat_block_shorten, circlelength=self.size, 
                                    xsize=xsize, nside=nside)
        for i,block in zip(number, map(partial, coor)): 
            Addition_map += block * weight[i]
            del block
        return Addition_map    
    
    def plot_scanning_ADI(self, xsize=2520, nside=256):
        Addition_map = np.zeros(hp.nside2npix(nside), dtype=np.double)
        Weighted_map = np.zeros(hp.nside2npix(nside), dtype=np.double)
        Directed_map = np.zeros(hp.nside2npix(nside), dtype=np.double)
        number = range(len(self.rot_9090))
        partial = functools.partial(self.lonlat_block_shorten, circlelength=self.size, 
                                    xsize=xsize, nside=nside)
        for i,block in zip(number, map(partial, self.rot_9090)): 
            Addition_map += block * np.ones(len(self.rot_9090))[i]
            Weighted_map += block * self.df.shift_var_distance.values[i]
            Directed_map += block * self.df.shift_sum.values[i]
            del block
        return Addition_map, Weighted_map, Directed_map

    # helper functions
    def healpix_rotate(self, healpix_map, rot):
        '''
        read in an healpix map and return a healpix map with rotation=rot, 
        rot=(phi,theta,psi)
        '''        

        # pix-vec
        ipix = np.arange(len(healpix_map))
        nside = np.sqrt(len(healpix_map) / 12)
        if int(nside) != nside: return print('invalid nside');
        nside = int(nside)
        vec = hp.pix2vec(int(nside), ipix)
        rot_vec = (hp.rotator.Rotator(rot=rot)).I(vec)
        irotpix = hp.vec2pix(nside, rot_vec[0], rot_vec[1], rot_vec[2])
        
        # generate new healpix
        healpix = np.zeros(hp.nside2npix(nside), dtype=np.double)
        healpix = healpix_map[irotpix]
        return healpix

    def cart_healpix(self, cartview, nside):
        '''read in an matrix and return a healpix pixelization map'''

        if ('pix' in dir(self)):
            if (self.pix_nside == nside) and (self.pix_shape == cartview.shape):
                healpix = np.zeros(hp.nside2npix(nside), dtype=np.double)
                healpix[self.pix] = np.fliplr(np.flipud(cartview))
                return healpix

        # Generate a flat Healpix map and angular to pixels
        healpix = np.zeros(hp.nside2npix(nside), dtype=np.double)
        hptheta = np.linspace(0, np.pi, num=cartview.shape[0])[:, None]
        hpphi = np.linspace(-np.pi, np.pi, num=cartview.shape[1])
        self.pix = hp.ang2pix(nside, hptheta, hpphi)
        
        # save pix_nside & pix_shape to avoid repeat pixelized
        self.pix_nside = nside
        self.pix_shape = cartview.shape
        
        # re-pixelize
        healpix[self.pix] = np.fliplr(np.flipud(cartview))

        return healpix