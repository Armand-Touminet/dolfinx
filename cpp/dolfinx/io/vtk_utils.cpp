// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "vtk_utils.h"
#include "cells.h"
// #include "cells.h"
// #include "pugixml.hpp"
// #include <adios2.h>
// #include <algorithm>
// #include <complex>
// #include <dolfinx/fem/FiniteElement.h>
// #include <dolfinx/fem/Function.h>
// #include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
// #include <dolfinx/mesh/utils.h>
// #include <xtensor/xtensor.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
xt::xtensor<std::int64_t, 2>
io::extract_vtk_connectivity(const mesh::Mesh& mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& dofmap_x = mesh.geometry().dofmap();
  const std::size_t num_nodes = dofmap_x.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh.topology().cell_type() == mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  // Extract mesh 'nodes'
  const int tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().index_map(tdim)->size_local();

  // Build mesh connectivity

  // Loop over cells
  xt::xtensor<std::int64_t, 2> topology({num_cells, num_nodes});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      topology(c, i) = dofs_x[map[i]];
  }

  return topology;
}
//-----------------------------------------------------------------------------

// /// Tabulate the coordinate for every 'node' in a Lagrange function
// /// space.
// /// @param[in] V The function space. Must be a Lagrange space.
// /// @return An array with shape (num_dofs, 3) array where the ith row
// /// corresponds to the coordinate of the ith dof in `V` (local to
// /// process)
// xt::xtensor<double, 2>
// tabulate_lagrange_dof_coordinates(const dolfinx::fem::FunctionSpace& V)
// {
//   std::shared_ptr<const mesh::Mesh> mesh = V.mesh();
//   assert(mesh);
//   const std::size_t gdim = mesh->geometry().dim();
//   const int tdim = mesh->topology().dim();

//   // Get dofmap data
//   std::shared_ptr<const fem::DofMap> dofmap = V.dofmap();
//   assert(dofmap);
//   std::shared_ptr<const common::IndexMap> map_dofs = dofmap->index_map;
//   assert(map_dofs);
//   const int index_map_bs = dofmap->index_map_bs();
//   const int dofmap_bs = dofmap->bs();

//   // Get element data
//   std::shared_ptr<const fem::FiniteElement> element = V.element();
//   assert(element);
//   const int e_block_size = element->block_size();
//   const std::size_t scalar_dofs = element->space_dimension() / e_block_size;
//   const std::int32_t num_dofs
//       = index_map_bs * (map_dofs->size_local() + map_dofs->num_ghosts())
//         / dofmap_bs;

//   // Get the dof coordinates on the reference element and the  mesh
//   // coordinate map
//   const xt::xtensor<double, 2>& X = element->interpolation_points();
//   const fem::CoordinateElement& cmap = mesh->geometry().cmap();

//   // Prepare cell geometry
//   const graph::AdjacencyList<std::int32_t>& dofmap_x
//       = mesh->geometry().dofmap();
//   xtl::span<const double> x_g = mesh->geometry().x();
//   const std::size_t num_dofs_g = dofmap_x.num_links(0);

//   xtl::span<const std::uint32_t> cell_info;
//   if (element->needs_dof_transformations())
//   {
//     mesh->topology_mutable().create_entity_permutations();
//     cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
//   }
//   const auto apply_dof_transformation
//       = element->get_dof_transformation_function<double>();

//   // Tabulate basis functions at node reference coordinates
//   const xt::xtensor<double, 2> phi
//       = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);

//   // Loop over cells and tabulate dofs
//   auto map = mesh->topology().index_map(tdim);
//   assert(map);
//   const std::int32_t num_cells = map->size_local() + map->num_ghosts();
//   xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
//   xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});
//   xt::xtensor<double, 2> coords = xt::zeros<double>({num_dofs, 3});
//   for (std::int32_t c = 0; c < num_cells; ++c)
//   {
//     // Extract cell geometry
//     auto dofs_x = dofmap_x.links(c);
//     for (std::size_t i = 0; i < dofs_x.size(); ++i)
//     {
//       std::copy_n(std::next(x_g.begin(), 3 * dofs_x[i]), gdim,
//                   std::next(coordinate_dofs.begin(), i * gdim));
//     }

//     // Tabulate dof coordinates on cell
//     cmap.push_forward(x, coordinate_dofs, phi);
//     apply_dof_transformation(xtl::span(x.data(), x.size()),
//                              xtl::span(cell_info.data(), cell_info.size()),
//                              c, x.shape(1));

//     // Copy dof coordinates into vector
//     auto dofs = dofmap->cell_dofs(c);
//     for (std::size_t i = 0; i < dofs.size(); ++i)
//       for (std::size_t j = 0; j < gdim; ++j)
//         coords(dofs[i], j) = x(i, j);
//   }

//   return coords;
// }
// //-----------------------------------------------------------------------------
