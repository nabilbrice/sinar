import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from functools import partial
from ..rays import normalize

def read_intensity_file(file_path, keep_phi=False):
    """Reads intensity data from a single file with specific formatting.

    The file should have the following format:
    - Numerical float on the first line (ignored)
    - Header line with the number of energies, mu and phi in the grids
    - Energy grid (keV)
    - mu grid (cos theta)
    - phi grid (degrees), azimuth of the emission direction about the
      surface normal, measured from the plane spanned by the surface
      normal and the local magnetic field direction (phi = 0 in-plane).
      Only [0, 180] is tabulated: the atmosphere is mirror-symmetric
      about the normal--field plane.
    - intensities...

    Parameters
    ----------
    keep_phi : bool
        When False (default, backwards compatible), the phi dimension is
        collapsed by taking the phi_idx = 0 slice and the return value is
        (energy_grid, mu_grid, I_t, I_x, I_o) with 2D intensities.
        When True, returns (energy_grid, mu_grid, phi_grid, I_t, I_x, I_o)
        with intensities of shape (n_energy, n_mu, n_phi).
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
        intensity_t = data[:blocksize].reshape((num_energies, num_mu, num_phi))
        intensity_x = data[blocksize:2*blocksize].reshape((num_energies, num_mu, num_phi))
        intensity_o = data[2*blocksize:3*blocksize].reshape((num_energies, num_mu, num_phi))

    if keep_phi:
        return energy_grid, mu_grid, phi_grid, intensity_t, intensity_x, intensity_o
    # Legacy behavior: collapse phi with the phi_idx = 0 slice
    return (energy_grid, mu_grid,
            intensity_t[:, :, 0], intensity_x[:, :, 0], intensity_o[:, :, 0])

def _ascending_axis(grid, intensity, axis):
    """Returns (grid, intensity) with the given axis in ascending grid order."""
    if grid[0] > grid[-1]:
        return grid[::-1], np.flip(intensity, axis=axis)
    return grid, intensity

def interpolate_intensity(energy_grid, mu_grid, intensity):
    """Interpolates the intensity data on a regular (energy, mu) grid."""
    mu_grid, intensity = _ascending_axis(np.asarray(mu_grid),
                                         np.asarray(intensity), axis=1)
    interpolator = RegularGridInterpolator((jnp.array(energy_grid), jnp.array(mu_grid)),
                                           jnp.array(intensity),
                                           method="linear")
    return interpolator

def interpolate_intensity3(energy_grid, mu_grid, phi_grid, intensity):
    """Interpolates the intensity data on a regular (energy, mu, phi) grid.

    phi is in degrees on [0, 180], as tabulated. Axes stored in
    descending order in the file are flipped to ascending as required by
    RegularGridInterpolator.
    """
    mu_grid, intensity = _ascending_axis(np.asarray(mu_grid),
                                         np.asarray(intensity), axis=1)
    phi_grid, intensity = _ascending_axis(np.asarray(phi_grid),
                                          np.asarray(intensity), axis=2)
    interpolator = RegularGridInterpolator(
        (jnp.array(energy_grid), jnp.array(mu_grid), jnp.array(phi_grid)),
        jnp.array(intensity),
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
    """Filters out energy grid points that are not strictly increasing."""
    mask = check_increasing(energy_grid)
    t, x, o = intensity_collection
    return energy_grid[mask], mu_grid, t[mask], x[mask], o[mask]

def filter_energies3(energy_grid, mu_grid, phi_grid, *intensity_collection):
    mask = check_increasing(energy_grid)
    t, x, o = intensity_collection
    return energy_grid[mask], mu_grid, phi_grid, t[mask], x[mask], o[mask]

def read_checked_intensity_file(filepath):
    grids = read_intensity_file(filepath)
    if np.all(check_increasing(grids[0])):
        return grids
    else:
        return filter_energies(*grids)

def read_checked_intensity_file3(filepath):
    grids = read_intensity_file(filepath, keep_phi=True)
    if np.all(check_increasing(grids[0])):
        return grids
    else:
        return filter_energies3(*grids)

def load_checked_interpolators(filepath):
    grid = read_checked_intensity_file(filepath)
    return {
        't' : interpolate_intensity(*grid[0:3]),
        'x' : interpolate_intensity(*grid[0:2], grid[3]),
        'o' : interpolate_intensity(*grid[0:2], grid[4])
    }

def load_checked_interpolators3(filepath):
    grid = read_checked_intensity_file3(filepath)
    return {
        't' : interpolate_intensity3(*grid[0:4]),
        'x' : interpolate_intensity3(*grid[0:3], grid[4]),
        'o' : interpolate_intensity3(*grid[0:3], grid[5])
    }

# --- Geometry: (position, direction) -> (mu, phi, psi) relative to the local field ---

@partial(jax.jit, inline=True, static_argnames=("polarity_fold",))
def field_geometry(position, direction, bvec, eps=1e-12, polarity_fold=False):
    """Emission angles of a ray hitting a spherical surface.

    The tabulated atmosphere angles are defined relative to the surface
    normal n and the local magnetic field direction b:
    mu = cos(theta) between the *emission* direction and n, phi the
    azimuth of the emission direction about n measured from the plane
    spanned by (n, b), folded onto [0, 180] degrees by the mirror
    symmetry of the atmosphere about that plane, and psi the field
    inclination: the angle between b and n, folded onto [0, 90] degrees.

    Hemisphere fold (default, IDL convention): the northern-hemisphere
    symmetry of the raytracer is I(mu, phi; psi) = I(mu, phi; 180 - psi).
    Following the IDL (``angbv`` used as ``abs(cos(angbv))`` while the
    azimuth reference ``v = (b - n cpsi)/sqrt(1 - cpsi^2)`` keeps the
    signed ``cpsi``), psi is folded via |cos psi| but phi is measured
    from the *true* tangential component of b, with its actual polarity.

    With ``polarity_fold=True`` the whole field vector is flipped to
    ⟨b, n⟩ >= 0 before projecting instead. That is the exact B -> -B
    fold of the mode-transfer equations; it folds psi identically but
    additionally mirrors phi -> 180 - phi in the southern hemisphere,
    which is NOT what the IDL raytracer does.

    Parameters
    ----------
    position : Array (3,)
        Intersection point on the surface. For a star centered at the
        origin the outward normal is position / |position|.
    direction : Array (3,)
        The marched ray direction at the surface (pointing *into* the
        surface, as stored in the ray phase). The emission direction is
        its negation.
    bvec : Array (3,)
        The (unnormalized) magnetic field vector at `position`.
    eps : float
        Degeneracy guard. When the emission direction is along the
        normal (mu -> 1) or the field is along the normal (magnetic
        pole), phi is undefined and is returned as 0.
    polarity_fold : bool (static)
        Select the exact B -> -B fold instead of the IDL northern-
        hemisphere fold (see above). Default False (IDL convention).

    Returns
    -------
    mu : float
        Cosine of the emission angle relative to the outward normal.
    phi_deg : float
        Azimuth in degrees on [0, 180].
    psi_deg : float
        Field inclination relative to the normal, in degrees on [0, 90].
    """
    n_hat = normalize(position)
    d_hat = -normalize(direction)   # emission direction, off the surface
    mu = jnp.vecdot(d_hat, n_hat)

    b_hat = normalize(bvec)
    if polarity_fold:
        # Exact B -> -B fold: orient the field out of the surface. This
        # also reverses the tangential component, mirroring phi in the
        # southern hemisphere.
        b_hat = b_hat * jnp.sign(jnp.vecdot(b_hat, n_hat) + eps)

    cos_psi = jnp.vecdot(b_hat, n_hat)   # signed cpsi of the IDL
    psi = jnp.arccos(jnp.clip(jnp.abs(cos_psi), 0.0, 1.0))

    # Tangential components in the surface plane. b_t keeps the actual
    # field polarity (IDL's v vector, built with the signed cpsi).
    d_t = d_hat - mu * n_hat
    b_t = b_hat - cos_psi * n_hat

    b_t_norm = jnp.linalg.vector_norm(b_t)
    d_t_norm = jnp.linalg.vector_norm(d_t)
    degenerate = (b_t_norm < eps) | (d_t_norm < eps)

    # In-plane orthonormal basis anchored on the field azimuth
    e1 = b_t / jnp.maximum(b_t_norm, eps)
    e2 = jnp.linalg.cross(n_hat, e1)

    x = jnp.vecdot(d_t, e1)
    y = jnp.vecdot(d_t, e2)
    # Mirror symmetry about the (n, b) plane folds phi onto [0, pi]
    phi = jnp.abs(jnp.arctan2(y, x))
    phi = jnp.where(degenerate, 0.0, phi)

    return mu, jnp.rad2deg(phi), jnp.rad2deg(psi)

def emission_angles(position, direction, bvec, eps=1e-12, polarity_fold=False):
    """Backwards-compatible (mu, phi_deg) wrapper of field_geometry."""
    mu, phi_deg, _ = field_geometry(position, direction, bvec, eps,
                                    polarity_fold=polarity_fold)
    return mu, phi_deg

def _clip_to_grid(value, grid):
    """Clips a query coordinate into the tabulated range (edge extension)."""
    lo = jnp.minimum(grid[0], grid[-1])
    hi = jnp.maximum(grid[0], grid[-1])
    return jnp.clip(value, lo, hi)

# --- Inclination-resolved patch sets ---

def read_inclination(file_path):
    """Reads the field inclination (degrees) from the first line of a patch file.

    The inclination is the angle between the atmosphere normal and the
    magnetic field direction for which the table was computed.
    """
    with open(file_path, 'r') as f:
        return float(f.readline())

def read_checked_patch_file(filepath):
    """Reads one atmosphere patch: (incl_deg, energy, mu, phi, I_t, I_x, I_o)."""
    incl = read_inclination(filepath)
    return (incl,) + tuple(read_checked_intensity_file3(filepath))

def _load_patch_set(filepaths):
    """Reads and sorts a set of patch files by field inclination.

    The patches need not share energy, mu or phi grids: each patch keeps
    its own 3D (energy, mu, phi) interpolator, and the mu/phi query is
    clamped into each patch's own tabulated range before evaluation
    (the inclination axis itself is extrapolated, see _incl_weights).
    """
    patches = sorted((read_checked_patch_file(fp) for fp in filepaths),
                     key=lambda p: p[0])
    incl_grid = np.array([p[0] for p in patches])
    if not np.all(np.diff(incl_grid) > 0):
        raise AssertionError(
            f"Patch inclinations {incl_grid} are not strictly increasing; "
            f"duplicate inclination tables cannot be interpolated."
        )
    return jnp.array(incl_grid), patches

def _patch_interpolators(patch):
    _, E, mu, phi, T, X, O = patch
    return (interpolate_intensity3(E, mu, phi, T),
            interpolate_intensity3(E, mu, phi, X),
            interpolate_intensity3(E, mu, phi, O))

def _common_energy_range(patches):
    lo = max(p[1][0] for p in patches)
    hi = min(p[1][-1] for p in patches)
    return lo, hi

def _incl_weights(psi, incl_grid):
    """Bracketing indices and linear weight for inclination interpolation.

    Mirrors the 1D linear ``interpol`` of the IDL raytracer over
    ``angbv``: outside the tabulated inclination range the weight ``t``
    runs below 0 or above 1, so the nearest end segment is linearly
    *extrapolated* rather than clamped.
    """
    idx = jnp.clip(jnp.searchsorted(incl_grid, psi, side='right') - 1,
                   0, incl_grid.shape[0] - 2)
    t = (psi - incl_grid[idx]) / (incl_grid[idx + 1] - incl_grid[idx])
    return idx, t

def load_incl_polspec_brdf(filepaths, field_fn, spectrum=None, clamp_angles=True):
    """Loads a field-inclination-resolved polarized spectral BRDF.

    Each patch file tabulates I(E, mu, phi) for one field inclination
    psi (first line of the file, in degrees). For a ray hitting the
    surface, the local field inclination psi = angle(b, n) is computed
    from ``field_fn`` and the intensity is linearly interpolated in psi
    between the two bracketing patch tables, each first evaluated at the
    ray's (E, mu, phi) on its own grid.

    Parameters
    ----------
    filepaths : sequence of str
        Patch files; sorted internally by their inclination.
    field_fn : Callable (3,) -> (3,)
        Magnetic field vector at a position, e.g.
        ``partial(magnetic_field, components, orient)``.
    spectrum : array or None
        Energies at which to evaluate. Must lie inside the energy range
        common to all patches. When None, the (filtered) energy grid of
        the lowest-inclination patch is used, clipped to the common range.
    clamp_angles : bool
        Clip mu and phi queries into each patch's own tabulated range
        (nearest-edge extension). The patches cover different mu ranges,
        so this is on by default; without it, rays outside any single
        patch's mu coverage would produce NaNs. The field inclination
        psi is always linearly extrapolated beyond [psi_min, psi_max]
        (matching IDL), independently of this flag.

    Returns
    -------
    brdf : Callable (position, direction) -> Array (3, n_energy)
        Total, X-mode and O-mode intensities on the evaluation spectrum.
        Matches the ``brdf(position, phase[3:6])`` call convention of
        ``render_by_rayphase``.
    """
    incl_grid, patches = _load_patch_set(filepaths)
    e_lo, e_hi = _common_energy_range(patches)

    if spectrum is None:
        base = patches[0][1]
        spectrum = base[(base >= e_lo) & (base <= e_hi)]
    spectrum = jnp.array(spectrum)
    if spectrum[0] < e_lo or spectrum[-1] > e_hi:
        raise AssertionError(
            f"Spectrum endpoints {spectrum[0]}, {spectrum[-1]} outside of "
            f"the energy range [{e_lo}, {e_hi}] common to all patches. "
            f"This would result in nan values during interpolation.\n"
            f"Consider specifying a more limited spectrum."
        )

    interps = [_patch_interpolators(p) for p in patches]
    mu_grids = [jnp.array(p[2]) for p in patches]
    phi_grids = [jnp.array(p[3]) for p in patches]

    def eval_patch(i, mu, phi):
        mu_i, phi_i = mu, phi
        if clamp_angles:
            mu_i = _clip_to_grid(mu_i, mu_grids[i])
            phi_i = _clip_to_grid(phi_i, phi_grids[i])
        pts = jnp.column_stack([spectrum,
                                jnp.full_like(spectrum, mu_i),
                                jnp.full_like(spectrum, phi_i)])
        t_i, x_i, o_i = interps[i]
        return jnp.array([t_i(pts), x_i(pts), o_i(pts)])

    def brdf(position, direction):
        mu, phi, psi = field_geometry(position, direction, field_fn(position))
        # All patches are evaluated and blended with bracketing weights:
        # branch-free, so it stays jit- and vmap-friendly.
        vals = jnp.stack([eval_patch(i, mu, phi) for i in range(len(interps))])
        idx, t = _incl_weights(psi, incl_grid)
        return (1.0 - t) * vals[idx] + t * vals[idx + 1]

    return jax.jit(brdf, inline=True)

def load_incl_spec_brdf(filepaths, field_fn, spectrum=None, clamp_angles=True):
    """Field-inclination-resolved total-intensity BRDF.

    See load_incl_polspec_brdf; returns only the total intensity row.
    """
    pol = load_incl_polspec_brdf(filepaths, field_fn, spectrum, clamp_angles)
    return jax.jit(lambda position, direction: pol(position, direction)[0],
                   inline=True)

# --- BRDF loaders on the (position, direction) ray-phase interface ---

def load_full_polspec_brdf3(filepath, field_fn, clamp_angles=True):
    """Loads a phi-aware polarized spectral BRDF.

    Parameters
    ----------
    filepath : str
        Path to the tabulated atmosphere intensity file.
    field_fn : Callable (3,) -> (3,)
        Magnetic field vector at a position, e.g.
        ``partial(magnetic_field, components, orient)``.
    clamp_angles : bool
        Clip mu and phi queries into the tabulated range (nearest-edge
        extension) instead of returning NaN outside it. The tabulated mu
        range can be narrow (e.g. [0.19, 0.98]); rays outside it would
        otherwise silently produce NaNs.

    Returns
    -------
    brdf : Callable (position, direction) -> Array (3, n_energy)
        Total, X-mode and O-mode intensities over the full tabulated
        energy grid. Matches the ``brdf(position, phase[3:6])`` call
        convention of ``render_by_rayphase``.
    """
    grid = read_checked_intensity_file3(filepath)
    energy_grid, mu_grid, phi_grid = (jnp.array(grid[0]), jnp.array(grid[1]),
                                      jnp.array(grid[2]))
    total = interpolate_intensity3(*grid[0:4])
    xmode = interpolate_intensity3(*grid[0:3], grid[4])
    omode = interpolate_intensity3(*grid[0:3], grid[5])

    def brdf(position, direction):
        mu, phi = emission_angles(position, direction, field_fn(position))
        if clamp_angles:
            mu = _clip_to_grid(mu, mu_grid)
            phi = _clip_to_grid(phi, phi_grid)
        pts = jnp.column_stack([energy_grid,
                                jnp.full_like(energy_grid, mu),
                                jnp.full_like(energy_grid, phi)])
        return jnp.array([total(pts), xmode(pts), omode(pts)])

    return jax.jit(brdf, inline=True)

def load_full_spec_brdf3(filepath, field_fn, clamp_angles=True):
    """Phi-aware total-intensity BRDF over the full tabulated energy grid."""
    grid = read_checked_intensity_file3(filepath)
    energy_grid, mu_grid, phi_grid = (jnp.array(grid[0]), jnp.array(grid[1]),
                                      jnp.array(grid[2]))
    total = interpolate_intensity3(*grid[0:4])

    def brdf(position, direction):
        mu, phi = emission_angles(position, direction, field_fn(position))
        if clamp_angles:
            mu = _clip_to_grid(mu, mu_grid)
            phi = _clip_to_grid(phi, phi_grid)
        pts = jnp.column_stack([energy_grid,
                                jnp.full_like(energy_grid, mu),
                                jnp.full_like(energy_grid, phi)])
        return total(pts)

    return jax.jit(brdf, inline=True)

def load_fixed_polspec_brdf3(filepath, spectrum, field_fn, clamp_angles=True):
    """Phi-aware polarized BRDF evaluated on a caller-specified spectrum."""
    grid = read_checked_intensity_file3(filepath)
    if grid[0][0] > spectrum[0] or grid[0][-1] < spectrum[-1]:
        raise AssertionError(
            f"Spectrum endpoints {spectrum[0]}, {spectrum[-1]} outside of "
            f"energy grid range {grid[0][0]} to {grid[0][-1]}. "
            f"This would result in nan values during interpolation.\n"
            f"Consider specifying a more limited spectrum."
        )
    mu_grid, phi_grid = jnp.array(grid[1]), jnp.array(grid[2])
    spectrum = jnp.array(spectrum)
    total = interpolate_intensity3(*grid[0:4])
    xmode = interpolate_intensity3(*grid[0:3], grid[4])
    omode = interpolate_intensity3(*grid[0:3], grid[5])

    def brdf(position, direction):
        mu, phi = emission_angles(position, direction, field_fn(position))
        if clamp_angles:
            mu = _clip_to_grid(mu, mu_grid)
            phi = _clip_to_grid(phi, phi_grid)
        pts = jnp.column_stack([spectrum,
                                jnp.full_like(spectrum, mu),
                                jnp.full_like(spectrum, phi)])
        return jnp.array([total(pts), xmode(pts), omode(pts)])

    return jax.jit(brdf, inline=True)

# --- Legacy (uv, mu) interface, unchanged below ---

def load_full_spec_brdf(filepath):
    grid = read_checked_intensity_file(filepath)
    total = interpolate_intensity(*grid[0:3])
    return jax.jit(
        lambda uv, mu: total(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)])),
        inline=True
        )

def load_full_polspec_brdf(filepath):
    grid = read_checked_intensity_file(filepath)
    total = interpolate_intensity(*grid[0:3])
    xmode = interpolate_intensity(*grid[0:2], grid[3])
    omode = interpolate_intensity(*grid[0:2], grid[4])
    return jax.jit(
        lambda uv, mu: jnp.array([
            total(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)])),
            xmode(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)])),
            omode(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)]))
        ]),
        inline=True
        )

def load_full_stokes_brdf(filepath):
    grid = read_checked_intensity_file(filepath)

    stokes_I = interpolate_intensity(*grid[0:3])
    stokes_Q = interpolate_intensity(*grid[0:2], grid[3] - grid[4])
    return jax.jit(
        lambda uv, mu: jnp.array([
            stokes_I(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)])),
            stokes_Q(jnp.column_stack([grid[0], jnp.full_like(grid[0], mu)])),
            jnp.zeros_like(grid[0])
        ]),
        inline=True
        )

def load_fixed_spec_brdf(filepath, spectrum):
    grid = read_checked_intensity_file(filepath)
    if grid[0][0] > spectrum[0] or grid[0][-1] < spectrum[-1]:
        raise AssertionError(
            f"Spectrum endpoints {spectrum[0]}, {spectrum[-1]} outside of "
            f"energy grid range {grid[0][0]} to {grid[0][-1]}. "
            f"This would result in nan values during interpolation.\n"
            f"Consider specifying a more limited spectrum."
        )

    total = interpolate_intensity(*grid[0:3])
    return jax.jit(lambda uv, mu: 
                   jnp.array(
                        total(jnp.column_stack([spectrum, jnp.full_like(spectrum, mu)]))
                   ),
                   inline=True
                   )

def load_fixed_polspec_brdf(filepath, spectrum):
    grid = read_checked_intensity_file(filepath)
    if grid[0][0] > spectrum[0] or grid[0][-1] < spectrum[-1]:
        raise AssertionError(
            f"Spectrum endpoints {spectrum[0]}, {spectrum[-1]} outside of "
            f"energy grid range {grid[0][0]} to {grid[0][-1]}. "
            f"This would result in nan values during interpolation.\n"
            f"Consider specifying a more limited spectrum."
        )

    total = interpolate_intensity(*grid[0:3])
    xmode = interpolate_intensity(*grid[0:2], grid[3])
    omode = interpolate_intensity(*grid[0:2], grid[4])
    return jax.jit(lambda uv, mu: 
                   jnp.array(
                    [
                        total(jnp.column_stack([spectrum, jnp.full_like(spectrum, mu)])),
                        xmode(jnp.column_stack([spectrum, jnp.full_like(spectrum, mu)])),
                        omode(jnp.column_stack([spectrum, jnp.full_like(spectrum, mu)]))
                    ]
                   ),
                   inline=True
                   )