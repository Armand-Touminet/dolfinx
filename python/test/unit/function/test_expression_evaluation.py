# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import basix
import cffi
import ctypes
import ctypes.util
import dolfinx
import dolfinx.io
import dolfiny.la
import matplotlib.pyplot as plt
import numba
import numpy as np
import ufl
from mpi4py import MPI
import petsc4py.lib
from petsc4py import PETSc
import pytest


@pytest.mark.skipif(np.issubdtype(PETSc.ScalarType, np.complexfloating),
                    reason="Complex expression not implemented in ufc")
def test_rank0():
    """Test evaluation of UFL expression.

    This test evaluates gradient of P2 function at vertices of reference
    triangle. Because these points coincide with positions of point evaluation
    degrees-of-freedom of vector P1 space, values could be used to interpolate
    the expression into this space.

    For a donor function f(x, y) = x^2 + 2*y^2 result is compared with the
    exact gradient grad f(x, y) = [2*x, 4*y].
    """
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))
    vdP1 = dolfinx.VectorFunctionSpace(mesh, ("DG", 1))

    f = dolfinx.Function(P2)

    def expr1(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    f.interpolate(expr1)

    ufl_expr = ufl.grad(f)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    compiled_expr, module, code = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), (ufl_expr, points))

    ffi = cffi.FFI()

    @numba.njit
    def assemble_expression(b, kernel, mesh, dofmap, coeff, coeff_dofmap):
        pos, x_dofmap, x = mesh
        geometry = np.zeros((3, 3))
        w = np.zeros(6, dtype=PETSc.ScalarType)
        constants = np.zeros(1, dtype=PETSc.ScalarType)
        b_local = np.zeros(6, dtype=PETSc.ScalarType)

        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = x_dofmap[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    vec[2 * dofmap[i * 3 + j] + k] = array_evaluated[i, 2 * j + k]


    # Data structure for the result
    b = dolfinx.Function(vdP1)

    dofmap = vdP1.dofmap.list.array
    scatter(b.vector.array, array_evaluated, dofmap)

    def grad_expr1(x):
        values = np.empty((2, x.shape[1]))
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]
        return values

    b2 = dolfinx.Function(vdP1)
    b2.interpolate(grad_expr1)

    assert np.isclose((b2.vector - b.vector).norm(), 0.0)


def test_rank1():
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))
    vdP1 = dolfinx.VectorFunctionSpace(mesh, ("DG", 1))

    f = ufl.TrialFunction(P2)
    ufl_expr = ufl.grad(f)

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    compiled_expr = dolfinx.Expression(ufl_expr, points)

    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(np.arange(num_cells))

    @numba.njit
    def scatter(A, array_evaluated, dofmap0, dofmap1):
        for i in range(num_cells):
            rows = dofmap0[i, :]
            cols = dofmap1[i, :]
            A_local = array_evaluated[i, :]
            MatSetValues(A, 6, rows.ctypes, 6, cols.ctypes, A_local.ctypes, 1)

    a = ufl.TrialFunction(P2) * ufl.TestFunction(vdP1)[0] * ufl.dx
    A = dolfinx.fem.create_matrix(a)

    dofmap_col = P2.dofmap.list.array.reshape(num_cells, -1)
    dofmap_row = vdP1.dofmap.list.array

    dofmap_row_unrolled = (2 * np.repeat(dofmap_row, 2).reshape(-1, 2)
                           + np.arange(2)).flatten().astype(dofmap_row.dtype)
    dofmap_row = dofmap_row_unrolled.reshape(num_cells, -1)

    scatter(A.handle, array_evaluated, dofmap_row, dofmap_col)
    A.assemble()

    g = dolfinx.Function(P2, name="g")

    def expr1(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    g.interpolate(expr1)

    def grad_expr1(x):
        values = np.empty((2, x.shape[1]))
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        return values

    h = dolfinx.Function(vdP1)
    h.interpolate(grad_expr1)

    A_scipy = dolfiny.la.petsc_to_scipy(A)

    plt.spy(A_scipy, markersize=0.4)
    plt.xticks(np.arange(0, 401, step=100))
    # plt.yticks(np.arange(0, 1, step=0.2))
    plt.tight_layout()
    plt.savefig("grad.pdf")

    h2 = A * g.vector

    assert np.isclose((h2 - h.vector).norm(), 0.0)

    A_dense = A_scipy.todense()
    U, S, V = scipy.linalg.svd(A_dense)

    g1 = dolfinx.Function(P2, name="g1")
    g2 = dolfinx.Function(P2, name="g2")
    g3 = dolfinx.Function(P2, name="g3")

    g.vector.array[:] = V[-1, :]
    g1.vector.array[:] = V[-2, :]
    g2.vector.array[:] = V[-3, :]
    g3.vector.array[:] = V[-4, :]

    gs = [g, g1, g2, g3]

    for i in range(4):
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"eig{i}.xdmf", "w") as file:
            file.write_mesh(mesh)
            file.write_function(gs[i])


def test_rank1_div():
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    vP1 = dolfinx.VectorFunctionSpace(mesh, ("P", 1))
    dP0 = dolfinx.FunctionSpace(mesh, ("DG", 0))

    f = ufl.TrialFunction(vP1)
    ufl_expr = ufl.div(f)

    points = np.array([[0.25, 0.25]])
    compiled_expr = dolfinx.Expression(ufl_expr, points)

    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(np.arange(num_cells))

    @numba.njit
    def scatter(A, array_evaluated, dofmap0, dofmap1):
        for i in range(num_cells):
            rows = dofmap0[i, :]
            cols = dofmap1[i, :]
            A_local = array_evaluated[i, :]
            MatSetValues(A, 1, rows.ctypes, 6, cols.ctypes, A_local.ctypes, 1)

    a = ufl.TrialFunction(vP1)[0] * ufl.TestFunction(dP0) * ufl.dx
    A = dolfinx.fem.create_matrix(a)

    dofmap_col = vP1.dofmap.list.array
    dofmap_row = dP0.dofmap.list.array.reshape(num_cells, -1)

    dofmap_col_unrolled = (2 * np.repeat(dofmap_col, 2).reshape(-1, 2)
                           + np.arange(2)).flatten().astype(dofmap_col.dtype)
    dofmap_col = dofmap_col_unrolled.reshape(num_cells, -1)

    scatter(A.handle, array_evaluated, dofmap_row, dofmap_col)
    A.assemble()

    A_scipy = dolfiny.la.petsc_to_scipy(A)

    plt.spy(A_scipy, markersize=0.4)
    plt.tight_layout()
    plt.savefig("div.pdf")

    g = dolfinx.Function(vP1)

    def expr1(x):
        values = np.empty((2, x.shape[1]))
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        return values

    g.interpolate(expr1)

    def div_expr1(x):
        values = np.empty((1, x.shape[1]))
        values[0] = 6.0
        return values

    h = dolfinx.Function(dP0)
    h.interpolate(div_expr1)

    h2 = A * g.vector

    assert np.isclose((h2 - h.vector).norm(), 0.0)

def test_simple_evaluation():
    """Test evaluation of UFL Expression.

    This test evaluates a UFL Expression on cells of the mesh and compares the
    result with an analytical expression.

    For a function f(x, y) = 3*(x^2 + 2*y^2) the result is compared with the
    exact gradient:

        grad f(x, y) = 3*[2*x, 4*y].

    (x^2 + 2*y^2) is first interpolated into a P2 finite element space. The
    scaling by a constant factor of 3 and the gradient is calculated using code
    generated by FFCx. The analytical solution is found by evaluating the
    spatial coordinates as an Expression using UFL/FFCx and passing the result
    to a numpy function that calculates the exact gradient.
    """
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))

    # NOTE: The scaling by a constant factor of 3.0 to get f(x, y) is
    # implemented within the UFL Expression. This is to check that the
    # Constants are being set up correctly.
    def exact_expr(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    # Unused, but remains for clarity.
    def f(x):
        return 3 * (x[0] ** 2 + 2.0 * x[1] ** 2)

    def exact_grad_f(x):
        values = np.zeros_like(x)
        values[:, 0::2] = 2 * x[:, 0::2]
        values[:, 1::2] = 4 * x[:, 1::2]
        values *= 3.0
        return values

    expr = dolfinx.Function(P2)
    expr.interpolate(exact_expr)

    ufl_grad_f = dolfinx.Constant(mesh, PETSc.ScalarType(3.0)) * ufl.grad(expr)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grad_f_expr = dolfinx.Expression(ufl_grad_f, points)
    assert grad_f_expr.num_points == points.shape[0]
    assert grad_f_expr.value_size == 2

    # NOTE: Cell numbering is process local.
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    grad_f_evaluated = grad_f_expr.eval(cells)
    assert grad_f_evaluated.shape[0] == cells.shape[0]
    assert grad_f_evaluated.shape[1] == grad_f_expr.value_size * grad_f_expr.num_points

    # Evaluate points in global space
    ufl_x = ufl.SpatialCoordinate(mesh)
    x_expr = dolfinx.Expression(ufl_x, points)
    assert x_expr.num_points == points.shape[0]
    assert x_expr.value_size == 2
    x_evaluated = x_expr.eval(cells)
    assert x_evaluated.shape[0] == cells.shape[0]
    assert x_evaluated.shape[1] == x_expr.num_points * x_expr.value_size

    # Evaluate exact gradient using global points
    grad_f_exact = exact_grad_f(x_evaluated)

    assert np.allclose(grad_f_evaluated, grad_f_exact)


def test_assembly_into_quadrature_function():
    """Test assembly into a Quadrature function.

    This test evaluates a UFL Expression into a Quadrature function space by
    evaluating the Expression on all cells of the mesh, and then inserting the
    evaluated values into a PETSc Vector constructed from a matching Quadrature
    function space.

    Concretely, we consider the evaluation of:

        e = B*(K(T)))**2 * grad(T)

    where

        K = 1/(A + B*T)

    where A and B are Constants and T is a Coefficient on a P2 finite element
    space with T = x + 2*y.

    The result is compared with interpolating the analytical expression of e
    directly into the Quadrature space.

    In parallel, each process evaluates the Expression on both local cells and
    ghost cells so that no parallel communication is required after insertion
    into the vector.
    """
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 6)

    quadrature_degree = 2
    quadrature_points, wts = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    Q_element = ufl.VectorElement("Quadrature", ufl.triangle, quadrature_degree, quad_scheme="default")
    Q = dolfinx.FunctionSpace(mesh, Q_element)

    def T_exact(x):
        return x[0] + 2.0 * x[1]

    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))
    T = dolfinx.Function(P2)
    T.interpolate(T_exact)
    A = dolfinx.Constant(mesh, PETSc.ScalarType(1.0))
    B = dolfinx.Constant(mesh, PETSc.ScalarType(2.0))

    K = 1.0 / (A + B * T)
    e = B * K**2 * ufl.grad(T)

    e_expr = dolfinx.Expression(e, quadrature_points)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    e_eval = e_expr.eval(cells)

    # Assemble into Function
    e_Q = dolfinx.Function(Q)
    with e_Q.vector.localForm() as e_Q_local:
        e_Q_local.setBlockSize(e_Q.function_space.dofmap.bs)
        e_Q_local.setValuesBlocked(Q.dofmap.list.array, e_eval, addv=PETSc.InsertMode.INSERT)

    def e_exact(x):
        T = x[0] + 2.0 * x[1]
        K = 1.0 / (A.value + B.value * T)

        grad_T = np.zeros((2, x.shape[1]))
        grad_T[0, :] = 1.0
        grad_T[1, :] = 2.0

        e = B.value * K**2 * grad_T
        return e

    # FIXME: Below is only for testing purposes,
    # never to be used in user code!
    #
    # Replace when interpolation into Quadrature element works.
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    Q_dofs = Q.dofmap.list.array.reshape(num_cells, quadrature_points.shape[0])
    bs = Q.dofmap.bs

    Q_dofs_unrolled = bs * np.repeat(Q_dofs, bs).reshape(-1, bs) + np.arange(bs)
    Q_dofs_unrolled = Q_dofs_unrolled.reshape(-1, bs * quadrature_points.shape[0]).astype(Q_dofs.dtype)

    with e_Q.vector.localForm() as local:
        e_exact_eval = np.zeros_like(local.array)

        for cell in range(num_cells):
            xg = x_g[coord_dofs.links(cell), :tdim]
            x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
            e_exact_eval[Q_dofs_unrolled[cell]] = e_exact(x.T).T.flatten()

        assert np.allclose(local.array, e_exact_eval)
