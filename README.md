# healpix_bootstrap

Bootstrap errors in HEALPix bins for a given dataset with longitudes and latitudes.

Ideally, your dataset should looks like this, 

| index | longitudes | latitudes | observed feature 1 | observed feature 2 | ... |
| ----- | ---------- | --------- | ------------------ | ------------------ | --- |
| 1     |            |           |                    |                    |     |
| 2     |            |           |                    |                    |     |
| 3     |            |           |                    |                    |     |
| ...   |            |           |                    |                    |     |

## Usage

```python
from CatalogueHealpix import CatalogueHealpix

# hyperparameters
nside = 128                # the resolution of HEALPix map you want to generate

cat_healpix = CatalogueHealpix(lons, lats, observed_features)

'''
lons (np.ndarray) : 1d array represents the longitudes of each rows in the catalogue.
lats (np.ndarray) : 1d array represents the latitudes  of each rows in the catalogue.
observed_features (np.ndarray) : 1d array with the same length of lons and lats, 
    represents the columns in your catalogue you want to transform to a HEALPix array.
'''

healpix, errors = cat_healpix.get_healpix(nside)
# healpix : the HEALPix map representation of your input dataset(lons, lats, features)
# errors  : bootstrap errors of each bin in HEALPix grid

# generate MC realizations based on bootstrap errors calculated in `get_healpix`
num_realizations = 10000
realizations     = cat_healpix.healpix_realization(num_realizations)
# realizations.shape == (num_realizations, len of HEALPix array)
```
