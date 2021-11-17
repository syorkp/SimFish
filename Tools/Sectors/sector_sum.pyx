#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
from Tools.MyUtils.my_utils import rounding, minmax
cimport cython
from cython.parallel import prange

cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def sector_sum_vectorised_iterated_conditionals(cnp.ndarray coordinate_grid,
               cnp.ndarray triangle_vertices,
               cnp.ndarray masked_arena_pixels_1):

    # Define triangle ABC, which have ascending x values for vertices. TODO: unpack more efficiently
    cdef double xa = triangle_vertices[0][0]
    cdef double xb = triangle_vertices[1][0]
    cdef double xc = triangle_vertices[2][0]
    cdef double ya = triangle_vertices[0][1]
    cdef double yb = triangle_vertices[1][1]
    cdef double yc = triangle_vertices[2][1]

    # Create vectors for triangle sides
    cdef list ab = [xb - xa, yb - ya]
    cdef list bc = [xc - xb, yc - yb]
    cdef list ca = [xa - xc, ya - yc]

    # Create matrix B of triangle vertices (w.h.3.2) TODO: Note is just triangle_vertices
    cdef cnp.ndarray[double, ndim=2] repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])

    cdef int xmin = round(min(np.array(triangle_vertices)[:, 0]))
    cdef int xmax = round(max(np.array(triangle_vertices)[:, 0]))
    cdef int ymin = round(min(np.array(triangle_vertices)[:, 1]))
    cdef int ymax = round(max(np.array(triangle_vertices)[:, 1]))
    cdef cnp.ndarray[long, ndim=4] coordinates_to_test = coordinate_grid[ymin:ymax, xmin:xmax, :]
    cdef cnp.ndarray[double, ndim=4] full_field = np.tile(repeating_unit, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

    # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] new_vector_points = coordinates_to_test - full_field

    # Flip C along final axis (so y is where x was previously) (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] new_vector_points_flipped = np.flip(new_vector_points, 3)

    # Create matrix D with repeated triangle side vectors (w.h.3.2)
    cdef cnp.ndarray[double, ndim=2] old_vector_points_list = np.array([ab, bc, ca])
    cdef cnp.ndarray[double, ndim=4] old_vector_points = np.tile(old_vector_points_list, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

    # Perform E = C * D (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] cross_product_components = new_vector_points_flipped * old_vector_points

    # Subtract y from x values (final axis) from E (w.h.3)
    cdef cnp.ndarray[double, ndim=3] cross_product = cross_product_components[:, :, :, 0] - cross_product_components[:, :, :, 1]
    # cdef double[:, :, :] cross_product = cross_product_py

    cdef cnp.ndarray[double, ndim=3] masked_arena_pixels = masked_arena_pixels_1[ymin:ymax, xmin:xmax, :]

    cdef cnp.ndarray[double, ndim=1] segment_sum = np.zeros((3), dtype=np.double)

    cdef Py_ssize_t i_shape = cross_product.shape[0]
    cdef Py_ssize_t j_shape = cross_product.shape[1]

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    # # Note that this seems to be the part that takes a long time.
    for i in prange(i_shape, nogil=True):
        for j in range(j_shape):
            if (cross_product[i, j, 0] > 0 and cross_product[i, j, 1] > 0 and cross_product[i, j, 0] > 1) or \
                    (cross_product[i, j, 0] < 0 and cross_product[i, j, 1] < 0 and cross_product[i, j, 0] < 1):
                segment_sum[0] += masked_arena_pixels[i, j, 0]
                segment_sum[1] += masked_arena_pixels[i, j, 1]
                segment_sum[2] += masked_arena_pixels[i, j, 2]

    return segment_sum


@cython.boundscheck(False)
@cython.wraparound(False)
def sector_sum_conditional_iteration(cnp.ndarray coordinate_grid,
               cnp.ndarray triangle_vertices,
               cnp.ndarray masked_arena_pixels_py):
    #cdef long[:, :, :, :] coordinate_grid = coordinate_grid_py
    cdef double[:, :, :] masked_arena_pixels = masked_arena_pixels_py

    # Define triangle ABC, which have ascending x values for vertices. TODO: unpack more efficiently
    cdef double xa = triangle_vertices[0][0]
    cdef double xb = triangle_vertices[1][0]
    cdef double xc = triangle_vertices[2][0]
    cdef double ya = triangle_vertices[0][1]
    cdef double yb = triangle_vertices[1][1]
    cdef double yc = triangle_vertices[2][1]

    # Create vectors for triangle sides
    cdef list ab = [xb - xa, yb - ya]
    cdef list bc = [xc - xb, yc - yb]
    cdef list ca = [xa - xc, ya - yc]

    cdef cnp.ndarray[double, ndim=2] repeating_unit_py = np.array([[ya, xa], [yb, xb], [yc, xc]])
    cdef double[:, :] repeating_unit = repeating_unit_py

    cdef cnp.ndarray[double, ndim=2] old_vector_points_list_py = np.array([ab, bc, ca])
    cdef double[:, :] old_vector_points_list = old_vector_points_list_py

    cdef int xmin = round(min(np.array(triangle_vertices)[:, 0]))
    cdef int xmax = round(max(np.array(triangle_vertices)[:, 0]))
    cdef int ymin = round(min(np.array(triangle_vertices)[:, 1]))
    cdef int ymax = round(max(np.array(triangle_vertices)[:, 1]))
    cdef cnp.ndarray[long, ndim=4] coordinates_to_test = coordinate_grid[ymin:ymax, xmin:xmax, :]
    cdef cnp.ndarray[long, ndim=4] coordinates_to_test_flipped = np.flip(coordinates_to_test, 3)

    cdef chosen_arena_pixels = masked_arena_pixels[ymin:ymax, xmin:xmax, :]

    cdef int i_shape = coordinates_to_test_flipped.shape[0]
    cdef int j_shape = coordinates_to_test_flipped.shape[1]
    cdef int k_shape = coordinates_to_test_flipped.shape[2]
    cdef int l_shape = coordinates_to_test_flipped.shape[3]

    cdef cnp.ndarray[double, ndim=2] v_p = np.empty((k_shape, l_shape))
    cdef double[:, :] v = v_p

    cdef cnp.ndarray[double, ndim=1] cp_values = np.empty((3))
    cdef cnp.ndarray[double, ndim=2] sectors_p = np.zeros((i_shape, j_shape), dtype=np.double)
    cdef double[:, :] sectors = sectors_p

    for i in range(i_shape):
        for j in range(j_shape):
            for k in range(k_shape):
                for l in range(l_shape):
                    v[k, l] = (coordinates_to_test_flipped[i, j, k, l] - repeating_unit[k, l]) * old_vector_points_list[k, l]
                cp_values[k] = v[l, 0] - v[l, 1]
            if np.all(cp_values > 0) or np.all(cp_values < 0):
                sectors[i, j] = 1

    cdef cnp.ndarray[double, ndim=1] segment_sum = np.zeros((3), dtype=np.double)

    i_shape = sectors.shape[0]
    j_shape = sectors.shape[1]

    #
    for i in range(i_shape):
        for j in range(j_shape):
            if sectors[i, j]:
                segment_sum[0] = segment_sum[0] + chosen_arena_pixels[i, j, 0]
                segment_sum[1] = segment_sum[1] + chosen_arena_pixels[i, j, 1]
                segment_sum[2] = segment_sum[2] + chosen_arena_pixels[i, j, 2]
    return segment_sum


@cython.boundscheck(False)
@cython.wraparound(False)
def sector_sum_full_vectorisation(cnp.ndarray coordinate_grid,
               cnp.ndarray triangle_vertices,
               cnp.ndarray masked_arena_pixels):

    # Define triangle ABC, which have ascending x values for vertices. TODO: unpack more efficiently
    cdef double xa = triangle_vertices[0][0]
    cdef double xb = triangle_vertices[1][0]
    cdef double xc = triangle_vertices[2][0]
    cdef double ya = triangle_vertices[0][1]
    cdef double yb = triangle_vertices[1][1]
    cdef double yc = triangle_vertices[2][1]

    # Create vectors for triangle sides
    cdef list ab = [xb - xa, yb - ya]
    cdef list bc = [xc - xb, yc - yb]
    cdef list ca = [xa - xc, ya - yc]

    # Create matrix B of triangle vertices (w.h.3.2) TODO: Note is just triangle_vertices
    cdef cnp.ndarray[double, ndim=2] repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])

    cdef int xmin = round(min(np.array(triangle_vertices)[:, 0]))
    cdef int xmax = round(max(np.array(triangle_vertices)[:, 0]))
    cdef int ymin = round(min(np.array(triangle_vertices)[:, 1]))
    cdef int ymax = round(max(np.array(triangle_vertices)[:, 1]))
    cdef cnp.ndarray[long, ndim=4] coordinates_to_test = coordinate_grid[ymin:ymax, xmin:xmax, :]
    cdef cnp.ndarray[double, ndim=4] full_field = np.tile(repeating_unit, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

    # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] new_vector_points = coordinates_to_test - full_field

    # Flip C along final axis (so y is where x was previously) (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] new_vector_points_flipped = np.flip(new_vector_points, 3)

    # Create matrix D with repeated triangle side vectors (w.h.3.2)
    cdef list old_vector_points_list = [ab, bc, ca]
    cdef cnp.ndarray[double, ndim=4] old_vector_points = np.tile(old_vector_points_list, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))

    # Perform E = C * D (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] cross_product_components = new_vector_points_flipped * old_vector_points

    # Subtract y from x values (final axis) from E (w.h.3)
    cdef cnp.ndarray[double, ndim=3] cross_product = cross_product_components[:, :, :, 0] - cross_product_components[:, :, :, 1]

    # If points in cross product are negative, set equal to 1, else 0. (w.h.3)
    cdef cnp.ndarray[long, ndim=3] cross_product_less_than = (cross_product < 0) * 1

    # Along 3rd axis, sum values.
    cdef cnp.ndarray[long, ndim=2] cross_product_boolean_axis = np.sum(cross_product_less_than, axis=2)

    # Set points to 1 if that point cross product sum is 0 or 3.
    cdef cnp.ndarray[long, ndim=2] cross_product_boolean_axis_sum = ((cross_product_boolean_axis == 0) | (cross_product_boolean_axis == 3)) * 1

    # Expand enclosed points to dimensions in order to multiply by mask. (w.h.3)
    cdef cnp.ndarray[long, ndim=3] sector_points = np.expand_dims(cross_product_boolean_axis_sum, 2)
    sector_points = np.repeat(sector_points, 3, 2)

    # Multiply enclosion mask by pixel mask (w.h.3)
    masked_arena_pixels = masked_arena_pixels[ymin:ymax, xmin:xmax, :]
    cdef cnp.ndarray[double, ndim=3] weighted_points = sector_points * masked_arena_pixels

    # Sum values from entire matrix along all but final axis (3)
    cdef cnp.ndarray[double, ndim=1] total_sum = weighted_points.sum(axis=(0, 1))

    return total_sum

@cython.boundscheck(False)
@cython.wraparound(False)
def sector_sum_iterations(cnp.ndarray coordinate_grid,
               cnp.ndarray triangle_vertices,
               cnp.ndarray masked_arena_pixels):
    cdef double xa = triangle_vertices[0][0]
    cdef double xb = triangle_vertices[1][0]
    cdef double xc = triangle_vertices[2][0]
    cdef double ya = triangle_vertices[0][1]
    cdef double yb = triangle_vertices[1][1]
    cdef double yc = triangle_vertices[2][1]

    # Create vectors for triangle sides
    cdef list ab = [xb - xa, yb - ya]
    cdef list bc = [xc - xb, yc - yb]
    cdef list ca = [xa - xc, ya - yc]

    # Create matrix B of triangle vertices (w.h.3.2)
    cdef cnp.ndarray[double, ndim=2] repeating_unit = np.array([[xa, ya], [xb, yb], [xc, yc]])

    # Select coordinates from grid to test (between max and min values of vertices)
    cdef double xmin
    cdef double xmax
    cdef double ymin
    cdef double ymax
    xmin, xmax = minmax(np.array(triangle_vertices)[:, 0])
    ymin, ymax = minmax(np.array(triangle_vertices)[:, 1])
    cdef int xmin_index = rounding(xmin)
    cdef int xmax_index = rounding(xmax)
    cdef int ymin_index = rounding(ymin)
    cdef int ymax_index = rounding(ymax)

    cdef cnp.ndarray[cnp.int_t, ndim=4] coordinates_to_test_py = coordinate_grid[ymin_index:ymax_index,
                                                                                 xmin_index:xmax_index,
                                                                                 :, :]
    cdef long[:, :, :, :] coordinates_to_test = coordinates_to_test_py

    cdef cnp.ndarray[double, ndim=4] full_field_py = np.tile(repeating_unit, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))
    cdef double[:, :, :, :] full_field = full_field_py


    # Compute C = A-B (corresponds to subtracting vertices points from each coordinate in space (w.h.3.2)
    # cdef double new_vector_points = coordinates_to_test - full_field
    cdef int i_shape = coordinates_to_test.shape[0]
    cdef int j_shape = coordinates_to_test.shape[1]
    cdef int k_shape = coordinates_to_test.shape[2]
    cdef int l_shape = coordinates_to_test.shape[3]
    cdef cnp.ndarray[double, ndim=4] new_vector_points = np.zeros((i_shape, j_shape, k_shape, l_shape))
    for i in range(i_shape):
        for j in range(j_shape):
            for k in range(k_shape):
                for l in range(l_shape):
                    new_vector_points[i, j, k, l] = coordinates_to_test[i, j, k, l] - full_field[i, j, k, l]

    # Flip C along final axis (so y is where x was previously) (w.h.3.2)
    cdef cnp.ndarray[double, ndim=4] new_vector_points_flipped_py = np.flip(new_vector_points, 3)
    cdef double[:, :, :, :] new_vector_points_flipped = new_vector_points_flipped_py

    # Create matrix D with repeated triangle side vectors (w.h.3.2)
    cdef cnp.ndarray[double, ndim=2] old_vector_points_py = np.array([ab, bc, ca])
    cdef cnp.ndarray[double, ndim=4] old_vector_points_py_2 = np.tile(old_vector_points_py, (coordinates_to_test.shape[0], coordinates_to_test.shape[1], 1, 1))
    cdef double[:, :, :, :] old_vector_points = old_vector_points_py_2

    # Perform E = C * D (w.h.3.2)
    # cdef double cross_product_components = new_vector_points_flipped * old_vector_points
    i_shape = new_vector_points_flipped.shape[0]
    j_shape = new_vector_points_flipped.shape[1]
    k_shape = new_vector_points_flipped.shape[2]
    l_shape = new_vector_points_flipped.shape[3]
    cdef cnp.ndarray[double, ndim=4] cross_product_components_py = np.zeros((i_shape, j_shape, k_shape, l_shape))
    for i in range(i_shape):
        for j in range(j_shape):
            for k in range(k_shape):
                for l in range(l_shape):
                    cross_product_components_py[i, j, k, l] = new_vector_points_flipped[i, j, k, l] * old_vector_points[i, j, k, l]

    cdef double[:, :, :, :] cross_product_components = cross_product_components_py

    # Subtract y from x values (final axis) from E (w.h.3)
    # cdef cnp.ndarray[cnp.npy_float32, ndim=3] cross_product = cross_product_components[:, :, :, 0] - cross_product_components[:, :, :, 1]
    cdef double[:, :, :] cross_product_components_1 = cross_product_components[:, :, :, 0]
    # cdef double[:, :, :] cross_product_components_1 = cross_product_components_1_py
    # cdef cnp.ndarray[double, ndim=3] cross_product_components_2_py = cross_product_components[:, :, :, 1]
    cdef double[:, :, :] cross_product_components_2 = cross_product_components[:, :, :, 1]
    i_shape = cross_product_components_1.shape[0]
    j_shape = cross_product_components_1.shape[1]
    k_shape = cross_product_components_1.shape[2]
    cdef cnp.ndarray[double, ndim=3] cross_product_py = np.zeros((i_shape, j_shape, k_shape))
    for i in range(i_shape):
        for j in range(j_shape):
            for k in range(k_shape):
                cross_product_py[i, j, k] = cross_product_components_1[i, j, k] * cross_product_components_2[i, j, k]

    cdef double[:, :, :] cross_product = cross_product_py

    # If points in cross product are negative, set equal to 1, else 0. (w.h.3)
    # cdef cnp.ndarray[cnp.int_t, ndim=3] cross_product_less_than = (cross_product < 0) * 1
    i_shape = cross_product.shape[0]
    j_shape = cross_product.shape[1]
    k_shape = cross_product.shape[2]
    cdef cnp.ndarray[cnp.int32_t, ndim=3] cross_product_less_than_py = np.zeros((i_shape, j_shape, k_shape), dtype=np.int32)
    for i in range(i_shape):
        for j in range(j_shape):
            for k in range(k_shape):
                if cross_product[i, j, k] < 0:
                    cross_product_less_than_py[i, j, k] = 1

    cdef int[:, :, :] cross_product_less_than = cross_product_less_than_py

    # Along 3rd axis, sum values.
    cdef cnp.ndarray[cnp.int_t, ndim=2] cross_product_boolean_axis = np.sum(cross_product_less_than, axis=2)

    # Set points to 1 if that point cross product sum is 0 or 3.
    cdef cnp.ndarray[cnp.int_t, ndim=2] cross_product_boolean_axis_sum = ((cross_product_boolean_axis == 0) | (cross_product_boolean_axis == 3)) * 1

    # Expand enclosed points to dimensions in order to multiply by mask. (w.h.3)
    cdef cnp.ndarray[cnp.int_t, ndim=3] sector_points = np.expand_dims(cross_product_boolean_axis_sum, 2)
    sector_points = np.repeat(sector_points, 3, 2)

    # Multiply enclosion mask by pixel mask (w.h.3)
    cdef double[:, :, :] masked_arena_pixels_cy = masked_arena_pixels
    masked_arena_pixels_cy = masked_arena_pixels_cy[ymin_index:ymax_index, xmin_index:xmax_index, :]
    cdef cnp.ndarray[double, ndim=3] weighted_points = sector_points * masked_arena_pixels_cy

    # Sum values from entire matrix along all but final axis (3)
    cdef cnp.ndarray[double, ndim=1] total_sum = weighted_points.sum(axis=(0, 1))

    return total_sum


# sector_sum = sector_sum_vectorised_iterated_conditionals
sector_sum = sector_sum_iterations




