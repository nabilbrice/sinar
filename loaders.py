import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from functools import partial
from .rays import normalize

def read_intensity_file(file_path):
    """Reads intensity data from a single file with specific formatting.

    The file should have the following format:
    - Numerical float on the first line (ignored)
    - Header line with the number of energies, mu and phi in the grids
    - Energy grid (keV)
    - mu grid (cos theta)
    - phi grid (degrees)
    - intensities...
    """
    
    with open(file_path, 'r') as f:
        _ = f.readline() # Skip first line
        header = f.readline().strip()
        num_energies, num_mu, num_phi = np.array(header.split(), dtype=int)

        # The next line is the energy grid, with a number of data points specified in the header
        energy_grid = np.array(f.readline().strip().split(), dtype=float)
        assert len(energy_grid) == num_energies

        mu_grid = np.array(f.readline().strip().split(), dtype=float)
        assert len(mu_grid) == num_mu

        phi_grid = np.array(f.readline().strip().split(), dtype=float)
        assert len(phi_grid) == num_phi
        # From here on, there are as many data points in a line as the num_phi
        # Reading num_mu lines of data at once gives a grid
        # for fixed energy value but varying mu and phi

        # Read the rest of the file as a contiguous block:
        data = np.fromfile(f, sep=' ', dtype=float)

        blocksize = num_energies * num_mu * num_phi
        # The intensities are expected to be approximately the same for any phi
        # so take a (copy) slice of the data and store only the 2D grid with phi_idx = 0
        intensity_t = data[:blocksize].reshape((num_energies, num_mu, num_phi))[:, :, 0]
        intensity_x = data[blocksize:2*blocksize].reshape((num_energies, num_mu, num_phi))[:, :, 0]
        intensity_o = data[2*blocksize:3*blocksize].reshape((num_energies, num_mu, num_phi))[:, :, 0]

    return energy_grid, mu_grid, intensity_t, intensity_x, intensity_o

def interpolate_intensity(energy_grid, mu_grid, intensity):
    """Interpolates the intensity data on a regular grid"""
    interpolator = RegularGridInterpolator((jnp.array(energy_grid), jnp.array(mu_grid[::-1])),
                                           jnp.array(intensity[:, ::-1]),
                                           method="linear")
    return interpolator

def check_increasing(grid):
    mask = np.ones(len(grid), dtype=bool)
    diffs = np.diff(grid)
    non_increasing = np.zeros(len(grid), dtype=bool)
    non_increasing[1:] = diffs <= 0

    mask = np.logical_and(mask, ~non_increasing)
    return mask

def filter_energies(energy_grid, mu_grid, *intensity_collection):
    """Filters out energies where the intensity is not monotonic"""
    mask = check_increasing(energy_grid)
    t, x, o = intensity_collection
    return energy_grid[mask], mu_grid, t[mask, :], x[mask, :], o[mask, :]

def checked_read_intensity_file(filepath):
    grids = read_intensity_file(filepath)
    if np.all(check_increasing(grids[0])):
        return grids
    else:
        return filter_energies(*grids)
    
def load_checked_interpolators(filepath):
    grid = checked_read_intensity_file(filepath)
    return {
        't' : interpolate_intensity(*grid[0:3]),
        'x' : interpolate_intensity(*grid[0:2], grid[3]),
        'o' : interpolate_intensity(*grid[0:2], grid[4])
    }

def load_checked_full_spectrum(filepath):
    grid = checked_read_intensity_file(filepath)
    interpolation = interpolate_intensity(*grid[0:3])
    return lambda uv, mu: interpolation(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)]))

def load_checked_fixed_spectrum(filepath, spectrum):
    grid = checked_read_intensity_file(filepath)
    interpolation = interpolate_intensity(*grid[0:3])
    return jax.jit(lambda uv, mu: 
                   normalize(interpolation(jnp.column_stack([spectrum, jnp.full_like(spectrum, mu)]))),
                   inline=True
                   )