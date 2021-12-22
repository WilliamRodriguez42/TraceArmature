#include <iostream>
#include <stdint.h>
#include <string.h>

using namespace std;

#define NUM_IDS (1 << NUM_BITS)
int64_t id_counts[NUM_IDS];
int64_t id_coordinates[NUM_IDS*2];

extern "C" void PyInit_aabb_filter() { // Usually the PyInit_ syntax is required when creating a library that is directly importable. However I'm lazy and I want to use ctypes instead of standard python C, so we will abide by the naming convention, then simply not use it. Since this returns void, trying to import it directly will probably segfault
	cout << "You're doing it wrong" << endl;
}

extern "C" int64_t aabb_filter(
	int64_t* sparse, // sparse representation of ids
	int64_t* sparse_coordinates, // i, j dense matrix coordinates of each entry in the sparse array
	int64_t* sparse_aabb, // half height and half width of the aabb for each entry in the sparse array
	double* sparse_info, // Any additional information associated with an element in the sparse array that should be averaged
	int64_t info_length, // Length of the additional info for an element
	int64_t num_sparse_elements, // Length of sparse array
	int64_t* dense, // 2d matrix of indices mapping to the sparse array
	int64_t width, // width of the 2d dense matrix
	int64_t height, // height of the 2d dense matrix

	int64_t* sparse_filtered, // Return variable. The most frequent id within an aabb
	double* sparse_filtered_info, // Return variable. Average of each info element associated with the most frequent id within an aabb
	int64_t* sparse_filtered_counts // Return variable. The number of times we found the most frequent id within an aabb
) {
	int64_t num_remaining = 0; // Return variable

	bool* visited = new bool[num_sparse_elements](); // Indicates for each entry in the sparse array if it has been visited
	double* id_info = new double[NUM_IDS * info_length]; // The sum of the additional info for an ID

	// Iterate over elements in the sparse array
	for (int64_t sparse_index = 0; sparse_index < num_sparse_elements; sparse_index ++) {

		if (visited[sparse_index]) continue; // Skip any elements that are marked for deletion

		int64_t i = sparse_coordinates[sparse_index * 2 + 0]; // Unpack coordinates
		int64_t j = sparse_coordinates[sparse_index * 2 + 1];
		int64_t half_height = sparse_aabb[sparse_index * 2 + 0]; // Unpack aabb half height and half width
		int64_t half_width  = sparse_aabb[sparse_index * 2 + 1];

		// Find the mode within the bounding box
		int64_t mode_id = NUM_IDS; // Same as -1 casted to unsigned using NUM_BITS
		int64_t mode_counts = 0;
		memset(id_counts, 0, NUM_IDS*sizeof(int64_t)); // Clear the previous id counts
		memset(id_coordinates, 0, NUM_IDS*2*sizeof(int64_t)); // Clear the previous id coordinates
		memset(id_info, 0, NUM_IDS*info_length*sizeof(double)); // Clear the previous id info
		for (int64_t di = -half_height; di <= half_height; di ++) { // Iterate over the surrounding bounding box
			int64_t si = i + di; // Calculate coordinates of surrounding entry in the dense array

			for (int64_t dj = -half_width; dj <= half_width; dj ++) {

				int64_t sj = j + dj;
				int64_t surrounding_index = si * width + sj;

				int64_t surrounding_sparse_index = dense[surrounding_index];
				bool is_valid = surrounding_sparse_index != -1;

				if (is_valid) {
					visited[surrounding_sparse_index] = true; // Mark as visited

					int64_t surrounding_marker_id = sparse[surrounding_sparse_index];

					id_counts[surrounding_marker_id] ++;
					id_coordinates[surrounding_marker_id * 2 + 0] += si;
					id_coordinates[surrounding_marker_id * 2 + 1] += sj;

					int64_t info_start_index = surrounding_sparse_index * info_length;
					int64_t info_for_id_start_index = surrounding_marker_id * info_length;
					for (int64_t k = 0; k < info_length; k ++) {
						id_info[info_for_id_start_index + k] += sparse_info[info_start_index + k]; // Sum the id info
					}

					bool replace = id_counts[surrounding_marker_id] > mode_counts;
					mode_id = replace ? surrounding_marker_id : mode_id; // Update the mode if the current marker id has more counts than the existing mode
					mode_counts += replace; // If we need to update the mode, we know it'll only increase by 1
				}
			}
		}

		if (mode_id != NUM_IDS) {
			sparse_filtered[num_remaining] = mode_id; // Store the mode
			for (int64_t k = 0; k < info_length; k ++) {
				sparse_filtered_info[num_remaining * info_length + k] = id_info[mode_id * info_length + k] / mode_counts; // Take the average of each info element
			}
			sparse_filtered_counts[num_remaining] = mode_counts; // Store how many times we found the mode (measures our confidence)

			num_remaining ++;
		}

	}

	delete[] visited;
	delete[] id_info;

	return num_remaining;
}
