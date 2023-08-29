import numpy as np
import scipy.constants as constants
from geoana.kernels import prism_fz, prism_fzx, prism_fzy, prism_fzz
from scipy.constants import G as NewtG
import discretize

from SimPEG import props
from SimPEG.utils import mkvc, sdiag

from ...base import BasePDESimulation
from ..base import BaseEquivalentSourceLayerSimulation, BasePFSimulation
from ...simulation import LinearSimulation
from ...utils import validate_active_indices, validate_string

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


class Simulation3DChoclo(LinearSimulation):
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
    mesh : discretize.TreeMesh or discretize.TensorMesh
        Mesh use to run the gravity simulation.
    survey : SimPEG.potential_fields.gravity.Survey
        Gravity survey with information of the receivers.
    ind_active : (n_cells) array, optional
        Array that indicates which cells in ``mesh`` are active cells.
    rho : array
    rhoMap : mapping
    sensitivity_dtype : numpy.dtype, optional
        Data type that will be used to build the sensitivity matrix.
    store_sensitivities : str
        Options for storing sensitivity matrix. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and
          sensitivities do not need to be stored

    parallel : bool, optional
        If True, the simulation will run in parallel. If False, it will
        run in serial.
    """

    rho, rhoMap, rhoDeriv = props.Invertible("Density")

    def __init__(
        self,
        mesh,
        survey,
        ind_active=None,
        rho=None,
        rhoMap=None,
        sensitivity_dtype=np.float32,
        store_sensitivities="ram",
        parallel=True,
        n_processes=None,
        sensitivity_path=None,
    ):
        if choclo is None:
            raise ImportError("Choclo is not installed")
        super().__init__(mesh=mesh)
        self.survey = survey
        self.sensitivity_dtype = sensitivity_dtype
        self.store_sensitivities = store_sensitivities
        self.ind_active = ind_active
        # Define physical property and maps
        self.rho = rho
        self.rhoMap = rhoMap
        self.modelMap = self.rhoMap
        # Define jit function for filling the sensitivity matrix
        self._fill_sensitivity_matrix = jit(nopython=True, parallel=parallel)(
            _fill_sensitivity_matrix
        )
        # Support n_process for backward compatibility
        if n_processes is not None:
            raise NotImplementedError(
                "Choosing number of processes is not implemented in this "
                "simulation class. You can use `parallel=True` or `parallel=False` "
                "to enable or disable parallelization."
            )
        # Support sensitivity_path for backward compatibility
        if sensitivity_path is not None:
            raise NotImplementedError(
                "Storing sensitivites in disk is not yet implemented in this "
                "simulation class."
            )

    @property
    def store_sensitivities(self):
        """Options for storing sensitivities.

        There are 3 options:

        - 'ram': sensitivity matrix stored in RAM
        - 'disk': sensitivities written and stored to disk
        - 'forward_only': sensitivities are not store (only use for forward simulation)

        Returns
        -------
        {'disk', 'ram', 'forward_only'}
            A string defining the model type for the simulation.
        """
        if not hasattr(self, "_store_sensitivities"):
            self._store_sensitivities = "ram"
        return self._store_sensitivities

    @store_sensitivities.setter
    def store_sensitivities(self, value):
        if value == "disk":
            raise NotImplementedError(
                "Storing sensitivities on disk is not currently supported."
            )
        self._store_sensitivities = validate_string(
            "store_sensitivities", value, ["disk", "ram", "forward_only"]
        )

    @property
    def ind_active(self) -> np.ndarray:
        """
        Array of bools that indicate active cells in the mesh
        """
        if not hasattr(self, "_ind_active"):
            self._ind_active = np.ones(self.mesh.n_cells, dtype=bool)
        return self._ind_active

    @ind_active.setter
    def ind_active(self, value):
        """
        Set ind_active property
        """
        self._ind_active = validate_active_indices(
            "ind_active", value, self.mesh.n_cells
        )

    @property
    def n_active_cells(self) -> int:
        """
        Number of active cells in the mesh
        """
        if not hasattr(self, "_n_active_cells"):
            self._n_active_cells = self.ind_active.sum()
        return self._n_active_cells

    @property
    def G(self):
        """
        Sensitivity matrix for the given survey and mesh.
        """
        if not hasattr(self, "_G"):
            self._G = self._sensitivity_matrix()
        return self._G

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
        # Forward model the fields
        fields = self.G @ m.astype(self.sensitivity_dtype, copy=False)
        return np.asarray(fields)

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

    def _get_cell_nodes(self):
        """
        Return indices of nodes for each cell in the mesh.
        """
        if isinstance(self.mesh, discretize.TreeMesh):
            cell_nodes = self.mesh.cell_nodes
        elif isinstance(self.mesh, discretize.TensorMesh):
            cell_nodes = self._get_tensormesh_cell_nodes()
        else:
            raise TypeError(f"Invalid mesh of type {self.mesh.__class__.__name__}.")
        return cell_nodes

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
        # Gather indices of nodes for each active cell in the mesh
        active_cell_nodes = self._get_cell_nodes()[self.ind_active]
        # Allocate sensitivity matrix
        shape = (self.survey.nD, self.n_active_cells)
        sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
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
        elif component in ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz", "guv"):
            conversion_factor = 1e12
        else:
            raise ValueError(f"Invalid component '{component}'.")
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
            sensitivity_matrix[receiver_index, k] = np.float32(
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

    def __init__(self, mesh, rho=None, rhoMap=None, **kwargs):
        super().__init__(mesh, **kwargs)
        self.rho = rho
        self.rhoMap = rhoMap
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

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
            self._G = self.linear_operator()

        return self._G

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
