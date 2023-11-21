from typing import List, Tuple
import discretize
import numpy as np
import pytest
from geoana.em.static import MagneticPrism
from scipy.constants import mu_0

from SimPEG import maps, utils
from SimPEG.potential_fields import magnetics as mag


@pytest.fixture
def mag_mesh() -> discretize.TensorMesh:
    """
    a small tensor mesh for testing magnetic simulations

    Returns
    -------
    discretize.TensorMesh
        the tensor mesh for testing
    """
    # Define a mesh
    cs = 0.2
    hxind = [(cs, 41)]
    hyind = [(cs, 41)]
    hzind = [(cs, 41)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")
    return mesh


@pytest.fixture
def two_blocks() -> tuple[np.ndarray, np.ndarray]:
    """
    The parameters defining two blocks

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tuple of (3, 2) arrays of (xmin, xmax), (ymin, ymax), (zmin, zmax) dimensions of each block
    """
    block1 = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    block2 = np.array([[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]])
    return block1, block2


@pytest.fixture
def receiver_locations() -> np.ndarray:
    """
    a grid of receivers for testing

    Returns
    -------
    np.ndarray
        (n, 3) array of receiver locations
    """
    # Create plane of observations
    nx, ny = 5, 5
    xr = np.linspace(-20, 20, nx)
    yr = np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    return np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]


@pytest.fixture
def inducing_field() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    inducing field two ways-- (amplitude, inclination , declination) and (b_x, b_y, b_z)

    Returns
    -------
    tuple[tuple[float, float, float], tuple[float, float, float]]
        (amplitude, inclination, declination), (b_x, b_y, b_z)
    """
    H0 = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])
    return H0, b0


def get_block_inds(grid: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    get the indices for a block

    Parameters
    ----------
    grid : np.ndarray
        (n, 3) array of xyz locations
    block : np.ndarray
        (3, 2) array of (xmin, xmax), (ymin, ymax), (zmin, zmax) dimensions of the block

    Returns
    -------
    np.ndarray
        boolean array of indices corresponding to the block
    """

    return np.where(
        (grid[:, 0] > block[0, 0])
        & (grid[:, 0] < block[0, 1])
        & (grid[:, 1] > block[1, 0])
        & (grid[:, 1] < block[1, 1])
        & (grid[:, 2] > block[2, 0])
        & (grid[:, 2] < block[2, 1])
    )


def create_block_model(
    mesh: discretize.TensorMesh,
    blocks: tuple[np.ndarray, ...],
    block_params: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a magnetic model from a sequence of blocks

    Parameters
    ----------
    mesh : discretize.TensorMesh
        TensorMesh object to put the model on
    blocks : tuple[np.ndarray, ...]
        tuple of block definitions (each element is (3, 2) array of (xmin, xmax), (ymin, ymax), (zmin, zmax)
        dimensions of the block)
    block_params : tuple[np.ndarray, ...]
        tuple of parameters to assign for each block. Must be the same length as ``blocks``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tuple of the magnetic model and active_cells (a boolean array)

    Raises
    ------
    ValueError
        if ``blocks`` and ``block_params`` have incompatible dimensions
    """
    if len(blocks) != len(block_params):
        raise ValueError(
            "'blocks' and 'block_params' must have the same number of elements"
        )
    model = np.zeros((mesh.n_cells, np.atleast_1d(block_params[0]).shape[0]))
    for block, params in zip(blocks, block_params):
        block_ind = get_block_inds(mesh.cell_centers, block)
        model[block_ind] = params
    active_cells = np.any(np.abs(model) > 0, axis=1)
    return model.squeeze(), active_cells


def create_mag_survey(
    components: list[str],
    receiver_locations: np.ndarray,
    inducing_field_params: tuple[float, float, float],
) -> mag.Survey:
    """
    create a magnetic Survey

    Parameters
    ----------
    components : list[str]
        list of components to model
    receiver_locations : np.ndarray
        (n, 3) array of xyz receiver locations
    inducing_field_params : tuple[float, float, float]
        amplitude, inclination, and declination of the inducing field

    Returns
    -------
    mag.Survey
        a magnetic Survey instance
    """

    receivers = mag.Point(receiver_locations, components=components)
    source_field = mag.UniformBackgroundField([receivers], *inducing_field_params)
    return mag.Survey(source_field)


@pytest.mark.parametrize(
    "engine,parallel_kwargs",
    [
        ("geoana", {"n_processes": None}),
        ("geoana", {"n_processes": 1}),
        ("choclo", {"choclo_parallel": False}),
        ("choclo", {"choclo_parallel": True}),
    ],
    ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
)
@pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
def test_ana_mag_forward(
    engine,
    parallel_kwargs,
    store_sensitivities,
    tmp_path,
    mag_mesh,
    two_blocks,
    receiver_locations,
    inducing_field,
):
    inducing_field_params, b0 = inducing_field

    chi1 = 0.01
    chi2 = 0.02
    model, active_cells = create_block_model(mag_mesh, two_blocks, [chi1, chi2])
    model_reduced = model[active_cells]
    # Create reduced identity map for Linear Problem
    identity_map = maps.IdentityMap(nP=int(sum(active_cells)))

    survey = create_mag_survey(
        components=["bx", "by", "bz", "tmi"],
        receiver_locations=receiver_locations,
        inducing_field_params=inducing_field_params,
    )
    sim = mag.Simulation3DIntegral(
        mag_mesh,
        survey=survey,
        chiMap=identity_map,
        ind_active=active_cells,
        sensitivity_path=str(tmp_path / f"{engine}"),
        store_sensitivities=store_sensitivities,
        engine=engine,
        **parallel_kwargs,
    )

    data = sim.dpred(model_reduced)
    d_x = data[0::4]
    d_y = data[1::4]
    d_z = data[2::4]
    d_t = data[3::4]

    tmi = sim.tmi_projection
    d_t2 = d_x * tmi[0] + d_y * tmi[1] + d_z * tmi[2]
    np.testing.assert_allclose(d_t, d_t2)  # double check internal projection

    # Compute analytical response from magnetic prism
    block1, block2 = two_blocks
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

    d = (
        prism_1.magnetic_flux_density(receiver_locations)
        + prism_2.magnetic_flux_density(receiver_locations)
        + prism_3.magnetic_flux_density(receiver_locations)
    )

    np.testing.assert_allclose(d_x, d[:, 0])
    np.testing.assert_allclose(d_y, d[:, 1])
    np.testing.assert_allclose(d_z, d[:, 2])
    np.testing.assert_allclose(d_t, d @ tmi)


@pytest.mark.parametrize(
    "engine, parallel_kwargs",
    [
        ("geoana", {"n_processes": None}),
        ("geoana", {"n_processes": 1}),
        ("choclo", {"choclo_parallel": False}),
        ("choclo", {"choclo_parallel": True}),
    ],
    ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
)
@pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
def test_ana_mag_grad_forward(
    engine,
    parallel_kwargs,
    store_sensitivities,
    tmp_path,
    mag_mesh,
    two_blocks,
    receiver_locations,
    inducing_field,
):
    inducing_field_params, b0 = inducing_field

    chi1 = 0.01
    chi2 = 0.02
    model, active_cells = create_block_model(mag_mesh, two_blocks, [chi1, chi2])
    model_reduced = model[active_cells]
    # Create reduced identity map for Linear Problem
    identity_map = maps.IdentityMap(nP=int(sum(active_cells)))

    survey = create_mag_survey(
        components=["bxx", "bxy", "bxz", "byy", "byz", "bzz"],
        receiver_locations=receiver_locations,
        inducing_field_params=inducing_field_params,
    )
    sim = mag.Simulation3DIntegral(
        mag_mesh,
        survey=survey,
        chiMap=identity_map,
        ind_active=active_cells,
        sensitivity_path=str(tmp_path / f"{engine}"),
        store_sensitivities=store_sensitivities,
        engine=engine,
        **parallel_kwargs,
    )
    if engine == "choclo":
        # gradient simulation not implemented for choclo yet
        with pytest.raises(NotImplementedError):
            data = sim.dpred(model_reduced)
    else:
        data = sim.dpred(model_reduced)
        d_xx = data[0::6]
        d_xy = data[1::6]
        d_xz = data[2::6]
        d_yy = data[3::6]
        d_yz = data[4::6]
        d_zz = data[5::6]

        # Compute analytical response from magnetic prism
        block1, block2 = two_blocks
        prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
        prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
        prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

        d = (
            prism_1.magnetic_field_gradient(receiver_locations)
            + prism_2.magnetic_field_gradient(receiver_locations)
            + prism_3.magnetic_field_gradient(receiver_locations)
        ) * mu_0

        # Check results
        rtol, atol = 1e-7, 1e-6
        np.testing.assert_allclose(d_xx, d[..., 0, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_xy, d[..., 0, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_xz, d[..., 0, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_yy, d[..., 1, 1], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_yz, d[..., 1, 2], rtol=rtol, atol=atol)
        np.testing.assert_allclose(d_zz, d[..., 2, 2], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "engine, parallel_kwargs",
    [
        ("geoana", {"n_processes": None}),
        ("geoana", {"n_processes": 1}),
        ("choclo", {"choclo_parallel": False}),
        ("choclo", {"choclo_parallel": True}),
    ],
    ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
)
@pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
def test_ana_mag_vec_forward(
    engine,
    parallel_kwargs,
    store_sensitivities,
    tmp_path,
    mag_mesh,
    two_blocks,
    receiver_locations,
    inducing_field,
):
    inducing_field_params, b0 = inducing_field
    M1 = (utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05).squeeze()
    M2 = (utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1).squeeze()

    model, active_cells = create_block_model(mag_mesh, two_blocks, [M1, M2])
    model_reduced = model[active_cells].reshape(-1, order="F")
    # Create reduced identity map for Linear Problem
    identity_map = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

    survey = create_mag_survey(
        components=["bx", "by", "bz", "tmi"],
        receiver_locations=receiver_locations,
        inducing_field_params=inducing_field_params,
    )

    sim = mag.Simulation3DIntegral(
        mag_mesh,
        survey=survey,
        chiMap=identity_map,
        ind_active=active_cells,
        sensitivity_path=str(tmp_path / f"{engine}"),
        store_sensitivities=store_sensitivities,
        model_type="vector",
        engine=engine,
        **parallel_kwargs,
    )

    data = sim.dpred(model_reduced).reshape(-1, 4)

    # Compute analytical response from magnetic prism
    block1, block2 = two_blocks
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0)

    d = (
        prism_1.magnetic_flux_density(receiver_locations)
        + prism_2.magnetic_flux_density(receiver_locations)
        + prism_3.magnetic_flux_density(receiver_locations)
    )
    tmi = sim.tmi_projection

    np.testing.assert_allclose(data[:, 0], d[:, 0])
    np.testing.assert_allclose(data[:, 1], d[:, 1])
    np.testing.assert_allclose(data[:, 2], d[:, 2])
    np.testing.assert_allclose(data[:, 3], d @ tmi)


@pytest.mark.parametrize(
    "engine, parallel_kwargs",
    [
        ("geoana", {"n_processes": None}),
        ("geoana", {"n_processes": 1}),
        ("choclo", {"choclo_parallel": False}),
        ("choclo", {"choclo_parallel": True}),
    ],
    ids=["geoana_serial", "geoana_parallel", "choclo_serial", "choclo_parallel"],
)
@pytest.mark.parametrize("store_sensitivities", ("ram", "disk", "forward_only"))
def test_ana_mag_amp_forward(
    engine,
    parallel_kwargs,
    store_sensitivities,
    tmp_path,
    mag_mesh,
    two_blocks,
    receiver_locations,
    inducing_field,
):
    inducing_field_params, b0 = inducing_field
    M1 = (utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05).squeeze()
    M2 = (utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1).squeeze()

    model, active_cells = create_block_model(mag_mesh, two_blocks, [M1, M2])
    model_reduced = model[active_cells].reshape(-1, order="F")
    # Create reduced identity map for Linear Problem
    identity_map = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

    survey = create_mag_survey(
        components=["bx", "by", "bz"],
        receiver_locations=receiver_locations,
        inducing_field_params=inducing_field_params,
    )

    sim = mag.Simulation3DIntegral(
        mag_mesh,
        survey=survey,
        chiMap=identity_map,
        ind_active=active_cells,
        sensitivity_path=str(tmp_path / f"{engine}"),
        store_sensitivities=store_sensitivities,
        model_type="vector",
        is_amplitude_data=True,
        engine=engine,
        **parallel_kwargs,
    )

    data = sim.dpred(model_reduced)

    # Compute analytical response from magnetic prism
    block1, block2 = two_blocks
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0)

    d = (
        prism_1.magnetic_flux_density(receiver_locations)
        + prism_2.magnetic_flux_density(receiver_locations)
        + prism_3.magnetic_flux_density(receiver_locations)
    )
    d_amp = np.linalg.norm(d, axis=1)

    # Check results
    rtol, atol = 1e-7, 1e-6
    np.testing.assert_allclose(data, d_amp, rtol=rtol, atol=atol)
