"""Miscellaneous functions for working with ALCHEMIC output files."""

import os
import numpy as np
import scipy.constants as sc


def get_file(molecule, folder):
    """Returns the path of the file for the given molecule."""
    for f in os.listdir(folder):
        if not f.endswith('.out'):
            continue
        moltag = '_' + molecule.lower() + '.'
        if moltag in f.lower():
            return folder + f
    raise ValueError('No file found in', folder)


def combine_orthopara(folder, molecule, filename=None):
    """Combine the ortho- and para- models to a single file."""
    try:
        get_file(molecule, folder)
        return
    except ValueError:
        pass

    orth = get_file('o' + molecule, folder)
    para = get_file('p' + molecule, folder)

    with open(orth) as fn:
        header = [next(fn) for _ in xrange(3)]
    header = ''.join(header)[:-2]
    header = header.replace('#', '')

    if filename is None:
        filename = orth.replace('o'+molecule, molecule)

    orth_data = np.loadtxt(orth, skiprows=3).T
    para_data = np.loadtxt(para, skiprows=3).T
    comb_data = orth_data.copy()
    comb_data[4] += para_data[4]

    np.savetxt(filename, comb_data.T, header=header)
    print('Successfully saved to', filename)
    return


def writeHeaderString(array, name):
    """Write string to header."""
    tosave = 'const static double %s[%d] = {' % (name, array.size)
    for val in array:
        tosave += '%.3e, ' % val
    tosave = tosave[:-2] + '};\n'
    return tosave


def makeHeader(path, name):
    """Make header suitable for makeLIME."""
    assert type(path) is str
    assert type(name) is str
    data = np.loadtxt(path, skiprows=3).T

    # Make the conversions to LIME appropriate units:
    # Main collider density (H2) is in [m^-3].
    # Relative abundance is with respect to the main collider density.
    # Temperatures are all in [K].

    with np.errstate(divide='ignore'):
        data[2] /= 2.37 * sc.m_p * 1e3
        data[-1] = data[-1]/data[2]
        data[2] *= 1e6
        data = np.where(~np.isfinite(data), 0.0, data)

    # Write the arrays and save to file.

    arrnames = ['c1arr', 'c2arr', 'dens', 'temp', 'abund']
    hstring = ''
    for i, array in enumerate(data):
            hstring += writeHeaderString(array, arrnames[i])
    if name[:-2] != '.h':
        name += '.h'
    with open(name, 'w') as hfile:
        hfile.write('%s' % hstring)
    print "Written to '%s'." % name
    return
