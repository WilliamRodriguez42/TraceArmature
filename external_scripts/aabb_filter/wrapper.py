import ctypes
import glob
import os
import pdb
import numpy as np

parent_directory = os.path.abspath(os.path.join(__file__, '..'))

if os.name == 'nt':
	library_paths = glob.glob(os.path.join(parent_directory, '*.pyd'))
else:
	library_paths = glob.glob(os.path.join(parent_directory, '*.so'))

assert len(library_paths) == 1 # Make sure we found exactly 1 library
library_path = library_paths[0]

aabb_filter_lib = ctypes.cdll.LoadLibrary(library_path)

aabb_filter_c = aabb_filter_lib.aabb_filter
aabb_filter_c.argtypes = [
	ctypes.POINTER(ctypes.c_longlong),
	ctypes.POINTER(ctypes.c_longlong),
	ctypes.POINTER(ctypes.c_longlong),
	ctypes.POINTER(ctypes.c_double),
	ctypes.c_longlong,
	ctypes.c_longlong,
	ctypes.POINTER(ctypes.c_longlong),
	ctypes.c_longlong,
	ctypes.c_longlong,

	ctypes.POINTER(ctypes.c_longlong),
	ctypes.POINTER(ctypes.c_double),
	ctypes.POINTER(ctypes.c_longlong),
]
aabb_filter_c.restype = ctypes.c_longlong


def aabb_filter(
	sparse,
	sparse_coordinates,
	sparse_aabb,
	sparse_info,

	dense=None, # This is just an intermediary that could be reused instead of redefined every call
):
	assert sparse.dtype == np.int64
	assert sparse_coordinates.dtype == np.int64
	assert sparse_aabb.dtype == np.int64
	assert sparse_info.dtype == np.float64

	if dense is None:
		dense = np.zeros((
				sparse_coordinates[:, 0].max() + sparse_aabb[:, 0].max(),
				sparse_coordinates[:, 1].max() + sparse_aabb[:, 1].max(),
		), dtype=np.int64)

	# Create inputs
	sparse_c = sparse.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	sparse_coordinates_c = sparse_coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	sparse_aabb_c = sparse_aabb.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	sparse_info_c = sparse_info.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	dense[:] = -1 # Wipe the dense matrix
	sparse_indices = np.arange(sparse.size)
	dense[sparse_coordinates[:, 0], sparse_coordinates[:, 1]] = sparse_indices
	dense_c = dense.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	# Create outputs
	sparse_filtered = np.zeros_like(sparse)
	sparse_filtered_c = sparse_filtered.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	sparse_filtered_info = np.zeros_like(sparse_info)
	sparse_filtered_info_c = sparse_filtered_info.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	sparse_filtered_counts = np.zeros_like(sparse)
	sparse_filtered_counts_c = sparse_filtered_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))

	num_remaining = aabb_filter_c(
		sparse_c,
		sparse_coordinates_c,
		sparse_aabb_c,
		sparse_info_c,
		sparse_info.shape[1],
		sparse.shape[0],
		dense_c,
		dense.shape[1],
		dense.shape[0],

		sparse_filtered_c,
		sparse_filtered_info_c,
		sparse_filtered_counts_c,
	)

	sparse_filtered = sparse_filtered[:num_remaining]
	sparse_filtered_info = sparse_filtered_info[:num_remaining]
	sparse_filtered_counts = sparse_filtered_counts[:num_remaining]

	return sparse_filtered, sparse_filtered_info, sparse_filtered_counts


if __name__ == '__main__':
	test_sparse = np.array(
		[0, 0, 0, 0, 1, 1, 2, 3],
	dtype=np.int64)

	test_sparse_coordinates = np.array([
		[2, 2],
		[2, 3],
		[3, 2],
		[3, 3],
		[4, 4],
		[4, 5],
		[5, 4],
		[5, 5],
	], dtype=np.int64)

	test_sparse_aabb = np.array([
		[1, 1]
	], dtype=np.int64)
	test_sparse_aabb = np.repeat(test_sparse_aabb, test_sparse.shape[0], axis=0) # All entries have an aabb of 5x5

	dense = np.zeros((8, 8), dtype=np.int64)

	filtered, filtered_info, filtered_coordinates = aabb_filter(
		test_sparse,
		test_sparse_coordinates,
		test_sparse_aabb,
		test_sparse_coordinates.astype(np.float64),

		dense=dense,
	)

	dense_ids = dense.copy()
	dense_ids[dense_ids != -1] = test_sparse[dense_ids[dense_ids != -1]]
	print(dense_ids)

	print(filtered)
	print(filtered_info)

	# pdb.set_trace()