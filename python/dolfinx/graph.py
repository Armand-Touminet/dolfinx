# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Graph module"""

import numpy as np

from dolfinx import cpp as _cpp

__all__ = ["create_adjacencylist"]


def create_adjacencylist(data: np.ndarray, offsets=None):
    """
    Create an AdjacencyList for int32 or int64 datasets.

    Parameters
    ----------
    data
        The adjacency array. If the array is one-dimensional, offsets should be supplied.
        If the array is two-dimensional the number of edges per node is the second dimension.
    offsets
        The offsets array with the number of edges per node.
    """
    if offsets is None:
        try:
            return _cpp.graph.AdjacencyList_int32(data)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data)
    else:
        try:
            return _cpp.graph.AdjacencyList_int32(data, offsets)
        except TypeError:
            return _cpp.graph.AdjacencyList_int64(data, offsets)
