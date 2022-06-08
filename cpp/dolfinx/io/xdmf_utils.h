// Copyright (C) 2012 Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <array>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/cell_types.h>
#include <filesystem>
#include <pugixml.hpp>
#include <string>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace pugi
{
class xml_node;
} // namespace pugi

namespace dolfinx
{

namespace fem
{
template <typename T>
class Function;
} // namespace fem

namespace fem
{
class CoordinateElement;
}

namespace mesh
{
class Mesh;
}

namespace io::xdmf_utils
{

// Get DOLFINx cell type string from XML topology node
// @return DOLFINx cell type and polynomial degree
std::pair<std::string, int> get_cell_type(const pugi::xml_node& topology_node);

// Return (0) HDF5 filename and (1) path in HDF5 file from a DataItem
// node
std::array<std::string, 2> get_hdf5_paths(const pugi::xml_node& dataitem_node);

std::filesystem::path
get_hdf5_filename(const std::filesystem::path& xdmf_filename);

/// Get dimensions from an XML DataSet node
std::vector<std::int64_t> get_dataset_shape(const pugi::xml_node& dataset_node);

/// Get number of cells from an XML Topology node
std::int64_t get_num_cells(const pugi::xml_node& topology_node);

/// Get point data values for linear or quadratic mesh into flattened 2D
/// array
std::vector<double> get_point_data_values(const fem::Function<double>& u);
std::vector<std::complex<double>>
get_point_data_values(const fem::Function<std::complex<double>>& u);

/// Get cell data values as a flattened 2D array
std::vector<double> get_cell_data_values(const fem::Function<double>& u);
std::vector<std::complex<double>>
get_cell_data_values(const fem::Function<std::complex<double>>& u);

/// Get the VTK string identifier
std::string vtk_cell_type_str(mesh::CellType cell_type, int num_nodes);

/// @warning Do not use. This function will be removed.
///
/// Send in_values[p0] to process p0 and receive values from process p1
/// in out_values[p1]
template <typename T>
graph::AdjacencyList<T> all_to_all(MPI_Comm comm,
                                   const graph::AdjacencyList<T>& send_data)
{
  const std::vector<std::int32_t>& send_offsets = send_data.offsets();
  const std::vector<T>& values_in = send_data.array();

  const int comm_size = dolfinx::MPI::size(comm);
  assert(send_data.num_nodes() == comm_size);

  // Data size per destination rank
  std::vector<int> send_size(comm_size);
  std::adjacent_difference(std::next(send_offsets.begin()), send_offsets.end(),
                           send_size.begin());

  // Get received data sizes from each rank
  std::vector<int> recv_size(comm_size);
  MPI_Alltoall(send_size.data(), 1, MPI_INT, recv_size.data(), 1, MPI_INT,
               comm);

  // Compute receive offset
  std::vector<std::int32_t> recv_offset(comm_size + 1, 0);
  std::partial_sum(recv_size.begin(), recv_size.end(),
                   std::next(recv_offset.begin()));

  // Send/receive data
  std::vector<T> recv_values(recv_offset.back());
  MPI_Alltoallv(values_in.data(), send_size.data(), send_offsets.data(),
                dolfinx::MPI::mpi_type<T>(), recv_values.data(),
                recv_size.data(), recv_offset.data(),
                dolfinx::MPI::mpi_type<T>(), comm);

  return graph::AdjacencyList<T>(std::move(recv_values),
                                 std::move(recv_offset));
}

/// Get owned entities and associated data from input entities defined
/// by global 'node' indices. The input entities and data can be
/// supplied on any rank and this function will manage the
/// communication.
///
/// @param[in] mesh A mesh
/// @param[in] entity_dim Topological dimension of entities to extract
/// @param[in] entities Mesh entities defined using global input indices
/// ('nodes'), typically from an input mesh file, e.g. [gi0, gi1, gi2]
/// for a triangle. Let [v0, v1, v2] be the vertex indices of some
/// triangle (using local indexing). Each vertex has a 'node' (geometry
/// dof) index, and each node has a persistent input global index, so
/// the triangle [gi0, gi1, gi2] could be identified with [v0, v1, v2].
/// The data is flattened and the shape is `(num_entities,
/// nodes_per_entity)`.
/// @param[in] data Data associated with each entity in `entities`.
/// @return (entity-vertex connectivity of owned entities, associated
/// data (values) with each entity)
/// @note This function involves parallel distribution and must be
/// called collectively. Global input indices for entities which are not
/// owned by current rank could passed to this function. E.g., rank0
/// provides an entity with global input indices [gi0, gi1, gi2], but
/// this identifies a triangle that is owned by rank1. It will be
/// distributed and rank1 will receive (local) cell-vertex connectivity
/// for this triangle.
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>>
distribute_entity_data(const mesh::Mesh& mesh, int entity_dim,
                       const xtl::span<const std::int64_t>& entities,
                       const xtl::span<const T>& data)
{
  LOG(INFO) << "XDMF distribute entity data";

  // Use ElementDofLayout of the cell to get vertex dof indices (local
  // to a cell), i.e. build a map from local vertex index to associated
  // local dof index
  std::vector<int> cell_vertex_dofs;
  {
    // Get layout of dofs on 0th cell entity of dimension entity_dim
    const fem::ElementDofLayout cmap_dof_layout
        = mesh.geometry().cmap().create_dof_layout();
    for (int i = 0; i < mesh::cell_num_entities(mesh.topology().cell_type(), 0);
         ++i)
    {
      const std::vector<int>& local_index = cmap_dof_layout.entity_dofs(0, i);
      assert(local_index.size() == 1);
      cell_vertex_dofs.push_back(local_index[0]);
    }
  }

  // -------------------
  // 1. Send this rank's global "input" nodes indices to the
  //    'postmaster' rank, and receive global "input" nodes for which
  //    this rank is the postmaster

  auto postmaster_global_nodes_sendrecv = [](const mesh::Mesh& mesh)
  {
    const MPI_Comm comm = mesh.comm();
    const int comm_size = dolfinx::MPI::size(comm);

    // Get "input" global node indices (as in the input file before any
    // internal re-ordering)
    const std::vector<std::int64_t>& nodes_g
        = mesh.geometry().input_global_indices();

    // Send input global indices to 'post master' rank, based on input
    // global index value
    const std::int64_t num_nodes_g = mesh.geometry().index_map()->size_global();
    // NOTE: could make this int32_t be sending: index <- index -
    // dest_rank_offset
    std::vector<std::vector<std::int64_t>> nodes_g_send(comm_size);
    for (std::int64_t node : nodes_g)
    {
      // Figure out which process is the postmaster for the input global
      // index
      const std::int32_t p
          = dolfinx::MPI::index_owner(comm_size, node, num_nodes_g);
      nodes_g_send[p].push_back(node);
    }

    // Send/receive
    LOG(INFO) << "XDMF send entity nodes size:(" << num_nodes_g << ")";
    graph::AdjacencyList<std::int64_t> nodes_g_recv
        = all_to_all(comm, graph::AdjacencyList<std::int64_t>(nodes_g_send));

    return nodes_g_recv;
  };

  const graph::AdjacencyList<std::int64_t> nodes_g_recv
      = postmaster_global_nodes_sendrecv(mesh);

  // -------------------
  // 2. Send the entity key (nodes list) and tag to the postmaster based
  //    on the lowest index node in the entity 'key'
  //
  //    NOTE: Stage 2 doesn't depend on the data received in Step 1, so
  //    data (i) the communication could be combined, or (ii) the
  //    communication in Step 1 could be make non-blocking.

  auto postmaster_global_ent_sendrecv
      = [&cell_vertex_dofs](const mesh::Mesh& mesh, int entity_dim,
                            const xtl::span<const std::int64_t>& entities,
                            const xtl::span<const T>& data)
  {
    const MPI_Comm comm = mesh.comm();
    const int comm_size = dolfinx::MPI::size(comm);
    const std::int64_t num_nodes_g = mesh.geometry().index_map()->size_global();

    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(mesh.topology().cell_type(), entity_dim, 0), 0);
    auto c_to_v = mesh.topology().connectivity(mesh.topology().dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");

    const fem::ElementDofLayout cmap_dof_layout
        = mesh.geometry().cmap().create_dof_layout();
    const std::vector<int> entity_layout
        = cmap_dof_layout.entity_closure_dofs(entity_dim, 0);

    // Find map from entity vertex to local (w.r.t. dof numbering on the
    // entity) dof number. E.g., if there are dofs on entity [0 3 6 7 9]
    // and dofs 3 and 7 belong to vertices, then this produces map [1,
    // 3].
    std::vector<int> entity_vertex_dofs;
    for (std::size_t i = 0; i < cell_vertex_dofs.size(); ++i)
    {
      auto it = std::find(entity_layout.begin(), entity_layout.end(),
                          cell_vertex_dofs[i]);
      if (it != entity_layout.end())
        entity_vertex_dofs.push_back(std::distance(entity_layout.begin(), it));
    }

    const std::size_t shape_e_1 = entity_layout.size();
    const std::size_t shape_e_0 = entities.size() / shape_e_1;
    std::vector<std::int64_t> entities_vertices(shape_e_0
                                                * num_vert_per_entity);
    for (std::size_t e = 0; e < shape_e_0; ++e)
    {
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
      {
        entities_vertices[e * num_vert_per_entity + i]
            = entities[e * shape_e_1 + entity_vertex_dofs[i]];
      }
    }

    std::vector<std::vector<std::int64_t>> entities_send(comm_size);
    std::vector<std::vector<T>> data_send(comm_size);
    for (std::size_t e = 0; e < shape_e_0; ++e)
    {
      xtl::span<std::int64_t> entity(entities_vertices.data()
                                         + e * num_vert_per_entity,
                                     num_vert_per_entity);
      std::sort(entity.begin(), entity.end());

      // Determine postmaster based on lowest entity node
      const std::int32_t p
          = dolfinx::MPI::index_owner(comm_size, entity.front(), num_nodes_g);
      entities_send[p].insert(entities_send[p].end(), entity.begin(),
                              entity.end());
      data_send[p].push_back(data[e]);
    }

    LOG(INFO) << "XDMF send entity keys size: (" << shape_e_0 << ")";
    // TODO: Pack into one MPI call
    graph::AdjacencyList<std::int64_t> entities_recv
        = all_to_all(comm, graph::AdjacencyList<std::int64_t>(entities_send));
    graph::AdjacencyList<T> data_recv
        = all_to_all(comm, graph::AdjacencyList<T>(data_send));

    return std::pair(entities_recv, data_recv);
  };

  const auto [entities_recv, data_recv]
      = postmaster_global_ent_sendrecv(mesh, entity_dim, entities, data);

  // -------------------
  // 3. As 'postmaster', send back the entity key (vertex list) and tag
  //    value to ranks that possibly need the data. Do this based on the
  //    first node index in the entity key.

  // NOTE: Could: (i) use a std::unordered_multimap, or (ii) only send
  // owned nodes to the postmaster and use map, unordered_map or
  // std::vector<pair>>, followed by a neighborhood all_to_all at the
  // end.

  auto postmaster_send_to_candidates
      = [](const mesh::Mesh& mesh, int entity_dim,
           const graph::AdjacencyList<std::int64_t>& nodes_g_recv,
           const graph::AdjacencyList<std::int64_t>& entities_recv,
           const graph::AdjacencyList<T>& data_recv)
  {
    const MPI_Comm comm = mesh.comm();
    const int comm_size = dolfinx::MPI::size(comm);

    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(mesh.topology().cell_type(), entity_dim, 0), 0);

    // Build map from global node index to ranks that have the node
    std::multimap<std::int64_t, int> node_to_rank;
    for (int p = 0; p < nodes_g_recv.num_nodes(); ++p)
    {
      auto nodes = nodes_g_recv.links(p);
      for (std::int32_t node : nodes)
        node_to_rank.insert({node, p});
    }

    // Figure out which processes are owners of received nodes
    std::vector<std::vector<std::int64_t>> send_nodes_owned(comm_size);
    std::vector<std::vector<T>> send_vals_owned(comm_size);
    const std::size_t shape0
        = entities_recv.array().size() / num_vert_per_entity;
    const std::size_t shape1 = num_vert_per_entity;
    const std::vector<T>& _data_recv = data_recv.array();
    assert(_data_recv.size() == shape0);
    for (std::size_t e = 0; e < shape0; ++e)
    {
      xtl::span e_recv(entities_recv.array().data() + e * shape1, shape1);

      // Find ranks that have node0
      auto [it0, it1] = node_to_rank.equal_range(e_recv.front());
      for (auto it = it0; it != it1; ++it)
      {
        const int p1 = it->second;
        send_nodes_owned[p1].insert(send_nodes_owned[p1].end(), e_recv.begin(),
                                    e_recv.end());
        send_vals_owned[p1].push_back(_data_recv[e]);
      }
    }

    // TODO: Pack into one MPI call
    const int send_val_size = std::transform_reduce(
        send_vals_owned.begin(), send_vals_owned.end(), 0, std::plus<int>(),
        [](const std::vector<T>& v) { return v.size(); });
    LOG(INFO) << "XDMF return entity and value data size:(" << send_val_size
              << ")";
    graph::AdjacencyList<std::int64_t> recv_ents = all_to_all(
        comm, graph::AdjacencyList<std::int64_t>(send_nodes_owned));
    graph::AdjacencyList<T> recv_vals
        = all_to_all(comm, graph::AdjacencyList<T>(send_vals_owned));

    return std::pair(std::move(recv_ents), std::move(recv_vals));
  };

  const auto [recv_ents, recv_vals] = postmaster_send_to_candidates(
      mesh, entity_dim, nodes_g_recv, entities_recv, data_recv);

  // -------------------
  // 4. From the received (key, value) data, determine which keys
  //    (entities) are on this process.

  // TODO: Rather than using std::map<std::vector<std::int64_t>,
  //       std::int32_t>, use a rectangular array to avoid the
  //       cost of std::vector<std::int64_t> allocations, and sort the
  //       Array by row.
  //
  // TODO: We have already received possibly tagged entities from other
  //       ranks, so we could use the received data to avoid creating
  //       the std::map for *all* entities and just for candidate
  //       entities.

  auto determine_my_entities
      = [&cell_vertex_dofs](const mesh::Mesh& mesh, int entity_dim,
                            const graph::AdjacencyList<std::int64_t>& recv_ents,
                            const graph::AdjacencyList<T>& recv_vals)
  {
    // Build map from input global indices to local vertex numbers
    LOG(INFO) << "XDMF build map";

    const std::size_t num_vert_per_entity = mesh::cell_num_entities(
        mesh::cell_entity_type(mesh.topology().cell_type(), entity_dim, 0), 0);
    auto c_to_v = mesh.topology().connectivity(mesh.topology().dim(), 0);
    if (!c_to_v)
      throw std::runtime_error("Missing cell-vertex connectivity.");

    const std::vector<std::int64_t>& nodes_g
        = mesh.geometry().input_global_indices();

    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh.geometry().dofmap();
    std::map<std::int64_t, std::int32_t> igi_to_vertex;
    for (int c = 0; c < c_to_v->num_nodes(); ++c)
    {
      auto vertices = c_to_v->links(c);
      auto x_dofs = x_dofmap.links(c);
      for (std::size_t v = 0; v < vertices.size(); ++v)
        igi_to_vertex[nodes_g[x_dofs[cell_vertex_dofs[v]]]] = vertices[v];
    }

    std::vector<std::int32_t> entities_new;
    entities_new.reserve(recv_ents.array().size());
    std::vector<T> data_new;
    data_new.reserve(recv_vals.array().size());
    std::vector<std::int32_t> entity(num_vert_per_entity);
    const std::vector<std::int64_t>& recv_ents_array = recv_ents.array();
    for (std::size_t e = 0; e < recv_ents_array.size() / num_vert_per_entity;
         ++e)
    {
      bool entity_found = true;
      for (std::size_t i = 0; i < num_vert_per_entity; ++i)
      {
        if (auto it
            = igi_to_vertex.find(recv_ents_array[e * num_vert_per_entity + i]);
            it == igi_to_vertex.end())
        {
          // As soon as this received index is not in locally owned input
          // global indices skip the entire entity
          entity_found = false;
          break;
        }
        else
          entity[i] = it->second;
      }

      if (entity_found == true)
      {
        entities_new.insert(entities_new.end(), entity.begin(), entity.end());
        data_new.push_back(recv_vals.array()[e]);
      }
    }

    return std::pair(std::move(entities_new), std::move(data_new));
  };

  auto [entities_new, data_new]
      = determine_my_entities(mesh, entity_dim, recv_ents, recv_vals);

  return {std::move(entities_new), std::move(data_new)};
}
//-----------------------------------------------------------------------------

/// TODO: Document
template <typename T>
void add_data_item(pugi::xml_node& xml_node, const hid_t h5_id,
                   const std::string& h5_path, const T& x, std::int64_t offset,
                   const std::vector<std::int64_t>& shape,
                   const std::string& number_type, bool use_mpi_io)
{
  // Add DataItem node
  assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  assert(data_item_node);

  // Add dimensions attribute
  std::string dims;
  for (auto d : shape)
    dims += std::to_string(d) + std::string(" ");
  dims.pop_back();
  data_item_node.append_attribute("Dimensions") = dims.c_str();

  // Set type for topology data (needed by XDMF to prevent default to
  // float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    assert(shape.size() == 2);
    std::ostringstream s;
    s.precision(16);
    for (std::size_t i = 0; i < (std::size_t)x.size(); ++i)
    {
      if ((i + 1) % shape[1] == 0 and shape[1] != 0)
        s << x.data()[i] << std::endl;
      else
        s << x.data()[i] << " ";
    }

    data_item_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
  }
  else
  {
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file, including path
    const std::filesystem::path p = HDF5Interface::get_filename(h5_id);
    const std::filesystem::path filename = p.filename().c_str();

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path
        = filename.string() + std::string(":") + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute data offset and range of values
    std::int64_t local_shape0 = x.size();
    for (std::size_t i = 1; i < shape.size(); ++i)
    {
      assert(local_shape0 % shape[i] == 0);
      local_shape0 /= shape[i];
    }

    const std::array local_range{offset, offset + local_shape0};
    HDF5Interface::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                                 use_mpi_io, false);

    // Add partitioning attribute to dataset
    // std::vector<std::size_t> partitions;
    // std::vector<std::size_t> offset_tmp(1, offset);
    // dolfinx::MPI::gather(comm, offset_tmp, partitions);
    // dolfinx::MPI::broadcast(comm, partitions);
    // HDF5Interface::add_attribute(h5_id, h5_path, "partition", partitions);
  }
}

} // namespace io::xdmf_utils
} // namespace dolfinx
