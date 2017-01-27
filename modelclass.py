import numpy as np
import scipy.constants as sc


class chemicalmodel:

    # A dictionary to hold the string versions of the column names.
    # This should make it easy to calculate abundance weighted values.

    indices = {}
    indices['rvals'] = 0
    indices['zvals'] = 1
    indices['density'] = 2
    indices['temperature'] = 3
    indices['abundance'] = 4

    # A dictionary to quickly convert from the units used in the file to more
    # commonly used units.

    units = {}
    units['m'] = sc.au
    units['cm'] = sc.au * 1e2
    units['mm'] = sc.au * 1e3

    def __init__(self, path, **kwargs):

        # Assume the data is in the format:
        # r [au], z [au], rho [g/ccm], T [K], n(mol) [/ccm]
        # and the number of radial points on the second line and
        # the number of vertical points on the third line. The
        # data then follows.

        self.path = path
        self.data = np.loadtxt(path, skiprows=3).T

        # The data is a 1+1D grid. Each vertical column will have
        # its own gridding. Thus we can split each physical property
        # into its own dictionary for easy access.

        with open(self.path) as fp:
            for i, line in enumerate(fp):
                if i == 1:
                    self.rpnts = int(line)
                if i == 2:
                    self.zpnts = int(line)
                if i > 2:
                    break

        self.rvals = np.unique(self.data[0])

        self.zvals = {r : self.data[1][self.data[0] == r] for r in self.rvals}
        self.density = {r : self.data[2][self.data[0] == r] for r in self.rvals}
        self.temperature = {r : self.data[3][self.data[0] == r] for r in self.rvals}
        self.abundance = {r : self.data[4][self.data[0] == r] for r in self.rvals}

        self.props = {}
        self.props[2] = self.density
        self.props[3] = self.temperature
        self.props[4] = self.abundance

        if len(self.rvals) != self.rpnts:
            raise ValueError('Error in parsing grid.')

        # For each radial position, make sure the first value is relabled to 0.
        # This allows for nicer looking plots with contouring filling down
        # to the bottom of the plot.

        for i in range(self.rpnts):
            self.data[1][1+i*self.zpnts] = 0.0

        # Default values. Can be changed through **kwargs.

        self.mu = kwargs.get('mu', 2.34)

        return

    @property
    def surfacedensity(self):
        '''Gas surface density profile in [sqcm].'''
        sigma = np.array([self.integrate_column(r, 2) for r in self.rvals])
        return 2. * sigma / sc.m_p / self.mu / 1e3

    @property
    def columndensity(self):
        '''Molecular column density in [sqcm].'''
        sigma = np.array([self.integrate_column(r, -1) for r in self.rvals])
        return 2. * sigma

    def abundance_weighted(self, param):
        '''Abundance weighted radial profiles.'''
        if type(param) is str:
            param = chemicalmodel.indices[param]
        assert type(param) is int
        p = [self.wpercentiles(self.props[param],
                               self.cell_weights(r))
             for r in self.rvals]
        return np.squeeze(p)

    def cell_weights(self, r):
        '''Weights for grid at r [au].'''
        return self.cell_size(self.zvals[r]) * self.abundance[r]

    def integrate_column(self, r, c, unit='cm'):
        '''Vertically integrate column.'''
        y = self.data[c][self.data[0] == r]
        x = self.data[1][self.data[0] == r]
        return np.trapz(y, x * chemicalmodel.units[unit])

    @staticmethod
    def cell_size(axis):
        '''Returns the cell size for an unstructured grid.'''
        mx = axis.size
        dx = np.diff(axis)
        sizes = [(dx[max(0, i-1)]+dx[min(i, mx-2)])*0.5 for i in range(mx)]
        return np.array(sizes)

    @staticmethod
    def wpercentiles(data, weights, percentiles=[0.16, 0.5, 0.84]):
        '''Weighted percentiles.'''

        # Make sure all weights are positive, then sort the data.

        assert all(weights >= 0)
        idx = np.argsort(data)
        sorted_data = np.take(data, idx)
        sorted_weights = np.take(weights, idx)

        # Calculate the cumulative sum. Note this is roughly
        # five times quicker than np.cumsum().

        cum_weights = np.add.accumulate(sorted_weights)
        scaled_weights = (cum_weights - 0.5 * sorted_weights) / cum_weights[-1]
        spots = np.searchsorted(scaled_weights, percentiles)

        # Interpolate the values. If either bounds are chosen, just
        # return them. Not likely to happen.

        wp = []
        for s, p in zip(spots, percentiles):
            if s == 0:
                wp.append(sorted_data[s])
            elif s == data.size:
                wp.append(sorted_data[s-1])
            else:
                f1 = (scaled_weights[s] - p)
                f1 /= (scaled_weights[s] - scaled_weights[s-1])
                f2 = (p - scaled_weights[s-1])
                f2 /= (scaled_weights[s] - scaled_weights[s-1])
                wp.append(sorted_data[s-1] * f1 + sorted_data[s] * f2)
        return wp
