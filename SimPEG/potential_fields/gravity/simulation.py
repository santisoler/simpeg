import warnings
import numpy as np
import scipy.constants as constants
from scipy.constants import G as NewtG

import discretize
from SimPEG import props
from SimPEG.utils import mkvc, sdiag

from ...base import BasePDESimulation
from ..base import BaseEquivalentSourceLayerSimulation
from ...simulation import LinearSimulation
from ...utils import validate_active_indices, validate_string

import choclo
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


def _forward_gravity(
    receivers,
    nodes,
    densities,
    fields,
    cell_nodes,
    kernel_func,
    constant_factor,
):
    """
    Forward model the gravity field of active cells on receivers
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kernels[j] = kernel_func(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            fields[i] += (
                constant_factor
                * densities[k]
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


def _fill_sensitivity_matrix(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    kernel_func,
    constant_factor,
):
    """
    Fill the sensitivity matrix

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    sensitivity_matrix : (n_receivers, n_active_nodes) array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
    cell_nodes : (n_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.

    Notes
    -----
    The conversion factor is applied here to each row of the sensitivity matrix
    because it's more efficient than doing it afterwards: it would require to
    index the rows that corresponds to each component.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kernels[j] = kernel_func(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            sensitivity_matrix[i, k] = np.float32(
                constant_factor
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


class Simulation3DIntegral(LinearSimulation):
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
    ind_active : (n_cells) array of int or bool, optional
        Array that indicates which cells in ``mesh`` are active cells.
    rho : array
        Density model for the mesh.
    rhoMap : mapping
        Model map.
    sensitivity_dtype : numpy.dtype, optional
        Data type that will be used to build the sensitivity matrix.
    store_sensitivities : str
        Options for storing sensitivity matrix. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and
          sensitivities do not need to be stored

    sensitivity_path : str
        Path to store the sensitivity matrix if ``store_sensitivities`` is set
        to ``"disk"``.
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
        sensitivity_path=None,
        parallel=True,
        **kwargs,
    ):
        # If deprecated property set with kwargs
        if "actInd" in kwargs:
            raise AttributeError(
                "actInd was removed in SimPEG 0.17.0, please use ind_active"
            )
        if "forwardOnly" in kwargs:
            raise AttributeError(
                "forwardOnly was removed in SimPEG 0.17.0, please set "
                "store_sensitivities='forward_only'"
            )
        if "n_processes" in kwargs:
            warnings.warn(
                "Passing 'n_processes' is no longer supported in this simulation class."
                "Use the 'parallel' argument for activating or deactivating"
                " parallelization.",
                DeprecationWarning,
                stacklevel=1,
            )
        super().__init__(
            mesh=mesh,
            survey=survey,
            **kwargs,
        )
        self.rho = rho
        self.rhoMap = rhoMap
        self.ind_active = ind_active
        self.sensitivity_dtype = sensitivity_dtype
        self.store_sensitivities = store_sensitivities
        if self.store_sensitivities == "disk":
            if sensitivity_path is None:
                raise ValueError(
                    "When passing 'store_sensitivities=\"disk\"', a value other"
                    " than 'None' should be passed to 'sensitivity_path'."
                )
            self.sensitivity_path = sensitivity_path
        # Define jit functions
        self._fill_sensitivity_matrix = jit(nopython=True, parallel=parallel)(
            _fill_sensitivity_matrix
        )
        self._forward_gravity = jit(nopython=True, parallel=parallel)(_forward_gravity)

    def linear_operator(self):
        """
        Deprecated method.

        Overrides the linear_operator method of the parent class.
        """
        raise AttributeError(
            "The 'linear_operator' method has been deprecated in the gravity"
            " simulation class."
        )

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
        self.model = m
        if self.store_sensitivities == "forward_only":
            fields = self._forward(m)
        else:
            fields = self.G @ m.astype(self.sensitivity_dtype, copy=False)
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
        if not hasattr(self, "_gtg_diagonal"):
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
        Sensitivity matrix for the given survey and mesh.
        """
        if not hasattr(self, "_G"):
            self._G = self._sensitivity_matrix()
        return self._G

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
        if self._store_sensitivities is None:
            self._store_sensitivities = "ram"
        return self._store_sensitivities

    @store_sensitivities.setter
    def store_sensitivities(self, value):
        self._store_sensitivities = validate_string(
            "store_sensitivities", value, ["disk", "ram", "forward_only"]
        )

    @property
    def sensitivity_dtype(self):
        """dtype of the sensitivity matrix.

        Returns
        -------
        numpy.float32 or numpy.float64
            The dtype used to store the sensitivity matrix
        """
        if self.store_sensitivities == "forward_only":
            return np.float64
        return self._sensitivity_dtype

    @sensitivity_dtype.setter
    def sensitivity_dtype(self, value):
        if value is not np.float32 and value is not np.float64:
            raise TypeError(
                "sensitivity_dtype must be either np.float32 or np.float64."
            )
        self._sensitivity_dtype = value

    @property
    def n_processes(self):
        raise AttributeError(
            "'n_processes' is not longer supported in the gravity simulation class."
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
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        if getattr(self, "_gtg_diagonal", None) is None:
            return None

        return self._gtg_diagonal

    def _forward(self, densities):
        """
        Forward model the fields of active cells in the mesh on receivers.

        Parameters
        ----------
        densities : (n_active_cells) array
            Array containing the densities of the active cells in the mesh, in
            g/cc.

        Returns
        -------
        (nD,) array
            Always return a ``np.float64`` array.
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate fields array
        fields = np.zeros(self.survey.nD, dtype=np.float32)
        # Start filling the sensitivity matrix
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_elements = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = self._get_conversion_factor(component)
                vector_slice = slice(
                    index_offset + i, index_offset + n_elements, n_components
                )
                self._forward_gravity(
                    receivers,
                    active_nodes,
                    densities,
                    fields[vector_slice],
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                )
            index_offset += n_elements
        return fields

    def _sensitivity_matrix(self):
        """
        Compute the sensitivity matrix G

        Returns
        -------
        (nD, n_active_cells) array
        """
        # Gather active nodes and the indices of the nodes for each active cell
        active_nodes, active_cell_nodes = self._get_active_nodes()
        # Allocate sensitivity matrix
        shape = (self.survey.nD, self.n_active_cells)
        if self.store_sensitivities == "disk":
            sensitivity_matrix = np.memmap(
                self.sensitivity_path,
                shape=shape,
                dtype=self.sensitivity_dtype,
                order="C",  # it's more efficient to write in row major
                mode="w+",
            )
        else:
            sensitivity_matrix = np.empty(shape, dtype=self.sensitivity_dtype)
        # Start filling the sensitivity matrix
        index_offset = 0
        for components, receivers in self._get_components_and_receivers():
            n_components = len(components)
            n_rows = n_components * receivers.shape[0]
            for i, component in enumerate(components):
                kernel_func = CHOCLO_KERNELS[component]
                conversion_factor = self._get_conversion_factor(component)
                matrix_slice = slice(
                    index_offset + i, index_offset + n_rows, n_components
                )
                self._fill_sensitivity_matrix(
                    receivers,
                    active_nodes,
                    sensitivity_matrix[matrix_slice, :],
                    active_cell_nodes,
                    kernel_func,
                    constants.G * conversion_factor,
                )
            index_offset += n_rows
        return sensitivity_matrix

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
        """
        Temporary implementation of ``TensorMesh.cell_nodes``.

        This method should be deleted after ``cell_nodes`` is added to
        ``TensorMesh`` in discretize.
        """
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

    def _get_active_nodes(self):
        """
        Return locations of nodes only for active cells.

        Also return an array containing the indices of the active nodes for
        each active cell in the mesh
        """
        # Get all nodes in the mesh
        if isinstance(self.mesh, discretize.TreeMesh):
            nodes = self.mesh.total_nodes
        elif isinstance(self.mesh, discretize.TensorMesh):
            nodes = self.mesh.nodes
        else:
            raise TypeError(f"Invalid mesh of type {self.mesh.__class__.__name__}.")
        # Get original cell_nodes but only for active cells
        cell_nodes = self._get_cell_nodes()
        # If all cells in the mesh are active, return nodes and cell_nodes
        if self.n_active_cells == self.mesh.n_cells:
            return nodes, cell_nodes
        # Keep only the cell_nodes for active cells
        cell_nodes = cell_nodes[self.ind_active]
        # Get the unique indices of the nodes that belong to every active cell
        # (these indices correspond to the original `nodes` array)
        unique_nodes, active_cell_nodes = np.unique(cell_nodes, return_inverse=True)
        # Select only the nodes that belong to the active cells (active nodes)
        active_nodes = nodes[unique_nodes]
        # Reshape indices of active cells for each active cell in the mesh
        active_cell_nodes = active_cell_nodes.reshape(cell_nodes.shape)
        return active_nodes, active_cell_nodes

    def _get_components_and_receivers(self):
        """Generator for receiver locations and their field components."""
        for receiver_object in self.survey.source_field.receiver_list:
            yield receiver_object.components, receiver_object.locations

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
