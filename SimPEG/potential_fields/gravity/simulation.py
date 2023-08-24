import numpy as np
import scipy.constants as constants
from geoana.kernels import prism_fz, prism_fzx, prism_fzy, prism_fzz
from scipy.constants import G as NewtG
import discretize

from SimPEG import props
from SimPEG.utils import mkvc, sdiag

from ...base import BasePDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation
from ...utils import validate_active_indices

try:
    import choclo
except ImportError:
    # Define dummy jit decorator
    def jit(*args, **kwargs):
        return lambda f: f

    choclo = None
else:
    from numba import jit, prange

    @jit(nopython=True)
    def kernel_uv(easting, northing, upward, radius):
        """Kernel for Guv gradiometry component."""
        result = 0.5 * (
            choclo.prism.kernel_nn(easting, northing, upward, radius)
            - choclo.prism.kernel_ee(easting, northing, upward, radius)
        )
        return result

    CHOCLO_KERNELS = {
        "gx": choclo.prism.kernel_e,
        "gy": choclo.prism.kernel_n,
        "gz": choclo.prism.kernel_u,
        "gxx": choclo.prism.kernel_ee,
        "gyy": choclo.prism.kernel_nn,
        "gzz": choclo.prism.kernel_uu,
        "gxy": choclo.prism.kernel_en,
        "gxz": choclo.prism.kernel_eu,
        "gyz": choclo.prism.kernel_nu,
        "guv": kernel_uv,
    }


class Simulation3DChoclo:
    """
    Gravity simulation in integral form using Choclo as computational engine.

    .. important::

        Density model is assumed to be in g/cc.

    .. important::

        Acceleration components ("gx", "gy", "gz") are returned in mgal
        (:math:`10^{-5} m/s^2`).

    .. important::

        Gradient components ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz") are
        returned in Eotvos (:math:`10^{-9} s^{-2}`).

    Parameters
    ----------
    survey : SimPEG.potential_fields.gravity.Survey
        Gravity survey with information of the receivers.
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the gravity simulation.
    ind_active : (n_cells) array, optional
        Array that indicates which cells in ``mesh`` are active cells.
    sensitivity_dtype : numpy.dtype, optional
        Data type that will be used to build the sensitivity matrix.
    parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial.
    """

    def __init__(
        self, survey, mesh, ind_active=None, sensitivity_dtype=np.float32, parallel=True
    ):
        if choclo is None:
            raise ImportError("Choclo is not installed")
        self.survey = survey
        self.mesh = mesh
        self.sensitivity_dtype = sensitivity_dtype

        if ind_active is None:
            ind_active = np.ones(mesh.n_cells, dtype=bool)
        else:
            ind_active = validate_active_indices("ind_active", ind_active, mesh.n_cells)
        self.ind_active = ind_active

        # initialize private attributes
        self._G = None
        self._cell_nodes = None
        self._active_cell_nodes = None
        self._n_active_cells = None
        if parallel:
            self._fill_sensitivity_matrix = _fill_sensitivity_matrix_parallel
        else:
            self._fill_sensitivity_matrix = _fill_sensitivity_matrix_serial

    @property
    def G(self):
        """
        Sensitivity matrix for the given survey and mesh.
        """
        if self._G is None:
            self._G = self._sensitivity_matrix()
        return self._G

    @property
    def cell_nodes(self):
        """
        Indices of nodes for each cell in the mesh.
        """
        if self._cell_nodes is None:
            if isinstance(self.mesh, discretize.TreeMesh):
                self._cell_nodes = self.mesh.cell_nodes
            elif isinstance(self.mesh, discretize.TensorMesh):
                self._cell_nodes = self._get_tensormesh_cell_nodes()
        return self._cell_nodes

    @property
    def n_active_cells(self):
        """
        Number of active cells in the mesh.
        """
        if self._n_active_cells is None:
            self._n_active_cells = np.sum(self.ind_active)
        return self._n_active_cells

    @property
    def active_cell_nodes(self):
        """
        Indices of nodes for each active cell in the mesh.
        """
        if self._active_cell_nodes is None:
            self._active_cell_nodes = self.cell_nodes[self.ind_active]
        return self._active_cell_nodes

    def _get_tensormesh_cell_nodes(self):
        """Dumb implementation of cell_nodes for a TensorMesh"""
        inds = np.arange(self.mesh.n_nodes).reshape(self.mesh.shape_nodes, order="F")
        cell_nodes = [
            inds[:-1, :-1, :-1].reshape(-1, order="F"),
            inds[1:, :-1, :-1].reshape(-1, order="F"),
            inds[:-1, 1:, :-1].reshape(-1, order="F"),
            inds[1:, 1:, :-1].reshape(-1, order="F"),
            inds[:-1, :-1, 1:].reshape(-1, order="F"),
            inds[1:, :-1, 1:].reshape(-1, order="F"),
            inds[:-1, 1:, 1:].reshape(-1, order="F"),
            inds[1:, 1:, 1:].reshape(-1, order="F"),
        ]
        cell_nodes = np.stack(cell_nodes, axis=-1)
        return cell_nodes

    def fields(self, m):
        """
        Forward model the gravity field of the mesh on the receivers in the survey

        Parameters
        ----------
        m : (n_active_cells,) array
            Array with values for the model.

        Returns
        -------
        (nD,) array
            Gravity fields generated by the given model on every receiver
            location.
        """
        fields = self.G @ m.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(fields)

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) array
        """
        # Gather observation points
        receivers, components = self._get_receivers()
        # Gather nodes
        nodes = self._get_nodes()
        # Allocate sensitivity matrix
        shape = (self.survey.nD, self.n_active_cells)
        sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
        active_cell_nodes = self.active_cell_nodes
        # Start filling the sensitivity matrix
        for component, receiver_indices in components.items():
            kernel_func = CHOCLO_KERNELS[component]
            conversion_factor = self._get_conversion_factor(component)
            self._fill_sensitivity_matrix(
                receivers,
                receiver_indices,
                nodes,
                sensitivity_matrix,
                active_cell_nodes,
                kernel_func,
                conversion_factor,
            )
        sensitivity_matrix *= constants.G
        return sensitivity_matrix

    def _get_conversion_factor(self, component):
        """
        Return conversion factor for the given component
        """
        if component in ("gx", "gy", "gz"):
            conversion_factor = 1e8
        elif component in ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz"):
            conversion_factor = 1e12
        return conversion_factor

    def _get_nodes(self):
        """Gather nodes from mesh"""
        if isinstance(self.mesh, discretize.TreeMesh):
            nodes = self.mesh.total_nodes
        elif isinstance(self.mesh, discretize.TensorMesh):
            nodes = self.mesh.nodes
        else:
            raise TypeError(f"Invalid mesh of type {self.mesh.__class__.__name__}.")
        return nodes

    def _get_receivers(self):
        """Gather receivers in the survey and their corresponding components"""
        # Get receivers locations
        receivers = np.vstack(
            [
                loc
                for r in self.survey.source_field.receiver_list
                for loc in r.locations
                for _ in r.components
            ]
        )
        # Get field components and indices of receivers for each component
        components_per_receiver = [
            comp
            for r in self.survey.source_field.receiver_list
            for _ in r.locations
            for comp in r.components
        ]
        components = {}
        for i, c in enumerate(components_per_receiver):
            if c not in components:
                components[c] = []
            components[c].append(i)
        for key, value in components.items():
            components[key] = np.array(value)
        return receivers, components


def _fill_sensitivity_matrix(
    receivers,
    receivers_indices,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    kernel_func,
    conversion_factor,
):
    """
    Fill the sensitivity matrix

    Notes
    -----
    The conversion factor is applied here to each row of the sensitivity matrix
    because it's more efficient than doing it afterwards: it would require to
    index the rows that corresponds to each component.
    """
    n_receivers = receivers_indices.size
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        receiver_index = receivers_indices[i]
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[receiver_index, 0]
            dy = nodes[j, 1] - receivers[receiver_index, 1]
            dz = nodes[j, 2] - receivers[receiver_index, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kernels[j] = kernel_func(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            sensitivity_matrix[i, k] = np.float32(
                conversion_factor
                * (
                    -kernels[cell_nodes[k, 0]]
                    + kernels[cell_nodes[k, 1]]
                    + kernels[cell_nodes[k, 2]]
                    - kernels[cell_nodes[k, 3]]
                    + kernels[cell_nodes[k, 4]]
                    - kernels[cell_nodes[k, 5]]
                    - kernels[cell_nodes[k, 6]]
                    + kernels[cell_nodes[k, 7]]
                )
            )


_fill_sensitivity_matrix_parallel = jit(nopython=True, parallel=True)(
    _fill_sensitivity_matrix
)
_fill_sensitivity_matrix_serial = jit(nopython=True, parallel=False)(
    _fill_sensitivity_matrix
)


class Simulation3DIntegral(BasePFSimulation):
    """
    Gravity simulation in integral form.

    .. important::

        Density model is assumed to be in g/cc.

    .. important::

        Acceleration components ("gx", "gy", "gz") are returned in mgal
        (:math:`10^{-5} m/s^2`).

    .. important::

        Gradient components ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz") are
        returned in Eotvos (:math:`10^{-9} s^{-2}`).
    """

    rho, rhoMap, rhoDeriv = props.Invertible("Density")

    def __init__(self, mesh, rho=None, rhoMap=None, engine="geoana", **kwargs):
        super().__init__(mesh, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap
        if engine == "choclo" and choclo is None:
            raise ImportError("Choclo is not installed.")
        self.engine = engine

    def fields(self, m):
        self.model = m

        if self.store_sensitivities == "forward_only":
            # Compute the linear operation without forming the full dense G
            fields = mkvc(self.linear_operator())
        else:
            fields = self.G @ (self.rho).astype(self.sensitivity_dtype, copy=False)

        return np.asarray(fields)

    def getJtJdiag(self, m, W=None, f=None):
        """
        Return the diagonal of JtJ
        """
        self.model = m

        if W is None:
            W = np.ones(self.survey.nD)
        else:
            W = W.diagonal() ** 2
        if getattr(self, "_gtg_diagonal", None) is None:
            diag = np.zeros(self.G.shape[1])
            for i in range(len(W)):
                diag += W[i] * (self.G[i] * self.G[i])
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))

    def getJ(self, m, f=None):
        """
        Sensitivity matrix
        """
        return self.G.dot(self.rhoDeriv)

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        dmu_dm_v = self.rhoDeriv @ v
        return self.G @ dmu_dm_v.astype(self.sensitivity_dtype, copy=False)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = self.G.T @ v.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(self.rhoDeriv.T @ Jtvec)

    @property
    def G(self):
        """
        Gravity forward operator
        """
        if getattr(self, "_G", None) is None:
            if self.engine == "geoana":
                self._G = self.linear_operator()
            elif self.engine == "choclo":
                self._G = self._compute_g_with_choclo()

        return self._G

    def _compute_g_with_choclo(self):
        """
        Compute sensitivity matrix using Choclo
        """
        # Get observation points
        coords = np.array(
            list(c[0] for c in self.survey._location_component_iterator())
        ).T
        # Get prisms
        if isinstance(self.mesh, discretize.TensorMesh):
            prisms = np.array(tensormesh_to_prisms(self.mesh))[self.ind_active]
        else:
            raise NotImplementedError(
                "Using choclo as engine only works with TensorMesh for now."
            )
        jacobian = build_jacobian(coords, prisms, dtype=self.sensitivity_dtype)
        return jacobian

    @property
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        if getattr(self, "_gtg_diagonal", None) is None:
            return None

        return self._gtg_diagonal

    def evaluate_integral(self, receiver_location, components):
        """
        Compute the forward linear relationship between the model and the physics at a point
        and for all components of the survey.

        :param numpy.ndarray receiver_location:  array with shape (n_receivers, 3)
            Array of receiver locations as x, y, z columns.
        :param list[str] components: List of gravity components chosen from:
            'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'guv'
        :param float tolerance: Small constant to avoid singularity near nodes and edges.
        :rtype numpy.ndarray: rows
        :returns: ndarray with shape (n_components, n_cells)
            Dense array mapping of the contribution of all active cells to data components::

                rows =
                    g_1 = [g_1x g_1y g_1z]
                    g_2 = [g_2x g_2y g_2z]
                           ...
                    g_c = [g_cx g_cy g_cz]

        """
        dr = self._nodes - receiver_location
        dx = dr[..., 0]
        dy = dr[..., 1]
        dz = dr[..., 2]

        node_evals = {}
        if "gx" in components:
            node_evals["gx"] = prism_fz(dy, dz, dx)
        if "gy" in components:
            node_evals["gy"] = prism_fz(dz, dx, dy)
        if "gz" in components:
            node_evals["gz"] = prism_fz(dx, dy, dz)
        if "gxy" in components:
            node_evals["gxy"] = prism_fzx(dy, dz, dx)
        if "gxz" in components:
            node_evals["gxz"] = prism_fzx(dx, dy, dz)
        if "gyz" in components:
            node_evals["gyz"] = prism_fzy(dx, dy, dz)
        if "gxx" in components or "guv" in components:
            node_evals["gxx"] = prism_fzz(dy, dz, dx)
        if "gyy" in components or "guv" in components:
            node_evals["gyy"] = prism_fzz(dz, dx, dy)
            if "guv" in components:
                node_evals["guv"] = (node_evals["gyy"] - node_evals["gxx"]) * 0.5
                # (NN - EE) / 2
        inside_adjust = False
        if "gzz" in components:
            node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # The below should be uncommented when we are able to give the index of a
            # containing cell.
            # if "gxx" not in node_evals or "gyy" not in node_evals:
            #     node_evals["gzz"] = prism_fzz(dx, dy, dz)
            # else:
            #     inside_adjust = True
            #     # The below need to be adjusted for observation points within a cell.
            #     # because `gxx + gyy + gzz = -4 * pi * G * rho`
            #     # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
            #     node_evals["gzz"] = -node_evals["gxx"] - node_evals["gyy"]

        rows = {}
        for component in set(components):
            vals = node_evals[component]
            if self._unique_inv is not None:
                vals = vals[self._unique_inv]
            cell_vals = (
                vals[0]
                - vals[1]
                - vals[2]
                + vals[3]
                - vals[4]
                + vals[5]
                + vals[6]
                - vals[7]
            )
            if inside_adjust and component == "gzz":
                # should subtract 4 * pi to the cell containing the observation point
                # just need a little logic to find the containing cell
                # cell_vals[inside_cell] += 4 * np.pi
                pass
            if self.store_sensitivities == "forward_only":
                rows[component] = cell_vals @ self.rho
            else:
                rows[component] = cell_vals
            if len(component) == 3:
                rows[component] *= constants.G * 1e12  # conversion for Eotvos
            else:
                rows[component] *= constants.G * 1e8  # conversion for mGal

        return np.stack(
            [
                rows[component].astype(self.sensitivity_dtype, copy=False)
                for component in components
            ]
        )


class SimulationEquivalentSourceLayer(
    BaseEquivalentSourceLayerSimulation, Simulation3DIntegral
):
    """
    Equivalent source layer simulations

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A 2D tensor or tree mesh defining discretization along the x and y directions
    cell_z_top : numpy.ndarray or float
        Define the elevations for the top face of all cells in the layer. If an array it should be the same size as
        the active cell set.
    cell_z_bottom : numpy.ndarray or float
        Define the elevations for the bottom face of all cells in the layer. If an array it should be the same size as
        the active cell set.
    """


class Simulation3DDifferential(BasePDESimulation):
    r"""Finite volume simulation class for gravity.

    Notes
    -----
    From Blakely (1996), the scalar potential :math:`\phi` outside the source region
    is obtained by solving a Poisson's equation:

    .. math::
        \nabla^2 \phi = 4 \pi \gamma \rho

    where :math:`\gamma` is the gravitational constant and :math:`\rho` defines the
    distribution of density within the source region.

    Applying the finite volumn method, we can solve the Poisson's equation on a
    3D voxel grid according to:

    .. math::
        \big [ \mathbf{D M_f D^T} \big ] \mathbf{u} = - \mathbf{M_c \, \rho}
    """

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)")

    def __init__(self, mesh, rho=1.0, rhoMap=None, **kwargs):
        super().__init__(mesh, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap

        self._Div = self.mesh.face_divergence

    def getRHS(self):
        """Return right-hand side for the linear system"""
        Mc = self.Mcc
        rho = self.rho
        return -Mc * rho

    def getA(self):
        r"""
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\Mf Mui)^{-1}\Div^{T}
        """
        # Constructs A with 0 dirichlet
        if getattr(self, "_A", None) is None:
            self._A = self._Div * self.Mf * self._Div.T.tocsr()
        return self._A

    def fields(self, m=None):
        r"""Compute fields

        **INCOMPLETE**

        Parameters
        ----------
        m: (nP) np.ndarray
            The model

        Returns
        -------
        dict
            The fields
        """
        if m is not None:
            self.model = m

        A = self.getA()
        RHS = self.getRHS()

        Ainv = self.solver(A)
        u = Ainv * RHS

        gField = 4.0 * np.pi * NewtG * 1e8 * self._Div * u

        return {"G": gField, "u": u}


@jit(nopython=True, parallel=True)
def build_jacobian(coordinates, prisms, dtype):
    """
    Build a sensitivity matrix for gravity_u of a prism
    """
    # Unpack coordinates of the observation points
    easting, northing, upward = coordinates[:]
    # Initialize an empty 2d array for the sensitivity matrix
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    jacobian = np.empty((n_coords, n_prisms), dtype=dtype)
    # Compute the gravity_u field that each prism generate on every observation
    # point, considering that they have a unit density
    for i in prange(len(easting)):
        for j in range(prisms.shape[0]):
            jacobian[i, j] = choclo.prism.gravity_u(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                1.0,
            )
    return jacobian


def tensormesh_to_prisms(mesh):
    """
    Converts a :class:`discretize.TensorMesh` into a set of prisms

    The prisms are listed following a FORTRAN order (the one used in
    :mod:`SimPEG`): first changing the ``upward`` coordinate, then the
    ``northing`` and then the ``easting`` one.

    Parameters
    ----------
    mesh : :class:`discretize.TensorMesh`

    Returns
    -------
    prisms : list
    """
    # Get centers of the prisms along each dimension
    centers_easting = mesh.cell_centers_x
    centers_northing = mesh.cell_centers_y
    centers_upward = mesh.cell_centers_z
    # Compute the west, east, south, north, bottom and top boundaries
    west = centers_easting - mesh.h[0] / 2
    east = centers_easting + mesh.h[0] / 2
    south = centers_northing - mesh.h[1] / 2
    north = centers_northing + mesh.h[1] / 2
    bottom = centers_upward - mesh.h[2] / 2
    top = centers_upward + mesh.h[2] / 2
    # Build the prisms
    prisms = []
    for b, t in zip(bottom, top):
        for s, n in zip(south, north):
            for w, e in zip(west, east):
                prisms.append([w, e, s, n, b, t])
    return prisms
