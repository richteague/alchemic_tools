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
        # its own gridding.

        with open(self.path) as fp:
            for i, line in enumerate(fp):
                if i == 1:
                    self.rpnts = int(line)
                if i == 2:
                    self.zpnts = int(line)
                if i > 2:
                    break
        self.rvals = np.unique(self.data[0])
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

    def integrate_column(self, r, c, unit='cm'):
        '''Vertically integrate column.'''
        y = self.data[c][self.data[0] == r]
        x = self.data[1][self.data[0] == r]
        return np.trapz(y, x * chemicalmodel.units[unit])
