import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cupyx.scipy.signal
import pdb
import time
# import bpy


import external_scripts.helper_functions as hf
from external_scripts.aabb_filter.wrapper import aabb_filter

def ellipse_distance_along_vector_from_center(coefficients, vx, vy, bk=np):
	A = coefficients[:, 0]
	B = coefficients[:, 1]
	C = coefficients[:, 2]
	D = coefficients[:, 3]
	E = coefficients[:, 4]

	return bk.sqrt((4*A*C + A*E**2 - B**2 - B*D*E + C*D**2)/((4*A*C - B**2)*(A*vx**2 + B*vx*vy + C*vy**2)))

def fit_ellipse(points, ray_origins, backend=np):
	bk = backend

	points = points - ray_origins[:, bk.newaxis, :]
	points = points.astype(bk.float64)

	# Extract x coords and y coords of the ellipse as batched column vectors
	X = points[:, :, 0, bk.newaxis]
	Y = points[:, :, 1, bk.newaxis]

	# Create a matrix of the linear terms
	A = bk.concatenate([X**2, X*Y, Y**2, X, Y], axis=-1)
	A_T = bk.swapaxes(A, -2, -1)
	A_T_A = bk.matmul(A_T, A)

	# Find which matrices are full column rank
	valid_indices = bk.where(bk.linalg.matrix_rank(A_T_A) == 5)[0]
	# show_valid_indices(projected_marker, cp.asnumpy(ray_origins[valid_indices]), bk=np)

	# Remove entries without full column rank
	A_T = A_T[valid_indices]
	A_T_A = A_T_A[valid_indices]

	# Solve the least squares problem
	b = bk.ones((A_T.shape[0], A_T.shape[2], 1), dtype=bk.float64)
	x = bk.linalg.solve(A_T_A, bk.matmul(A_T, b))
	x = x[:, :, 0] # Remove unnecessary last axis

	# Find the discriminant
	A = x[:, 0]
	B = x[:, 1]
	C = x[:, 2]
	D = x[:, 3]
	E = x[:, 4]
	discriminant = B*B - 4*A*C

	# Find which coefficients represent non-hyperbolas and only use them going forward
	non_hyperbola_indices = bk.where(discriminant <= 0)[0]

	x = x[non_hyperbola_indices, :] # Filter coefficients
	A = x[:, 0] # Re-unpack A-E
	B = x[:, 1]
	C = x[:, 2]
	D = x[:, 3]
	E = x[:, 4]

	valid_indices = valid_indices[non_hyperbola_indices] # Filter the valid indices (Now they represent the indices of points that are both full column rank and non-hyperbolas)

	valid_points = points[valid_indices, :, :] # Filter sampled points (Unlike all of the other terms being filtered in this section, we never filtered these based on full column rank, so we can gather from the filtered valid_indices directly)
	X = valid_points[:, :, 0, bk.newaxis] # Re-unpack X and Y
	Y = valid_points[:, :, 1, bk.newaxis]

	# Find the center coordinates of each fitted ellipse
	cx = (B*E - 2*C*D)/(4*A*C - B**2)
	cy = (-2*A*E + B*D)/(4*A*C - B**2)
	center = bk.column_stack([cx, cy])

	# Move sampled points to the origin of the fitted ellipse
	centered_points = valid_points - center[:, bk.newaxis, :]

	# Find the vectors of the major and minor axes
	dy = (-A + C - bk.sqrt(A**2 - 2*A*C + B**2 + C**2)) # (dx, dy) represents the vector of either the major or minor axis (this implementation doesn't care which one the angle points to), from the ellipse center
	dx = B
	first_axis_vector = bk.column_stack([dx, dy])
	first_axis_vector /= bk.linalg.norm(first_axis_vector, axis=1)[:, bk.newaxis]
	second_axis_vector = first_axis_vector[:, ::-1].copy() # Perpendicular to first axis
	second_axis_vector[:, 0] *= -1

	# Find the radius of the major and minor axes
	first_axis_radius = ellipse_distance_along_vector_from_center(x, first_axis_vector[:, 0], first_axis_vector[:, 1])
	second_axis_radius = ellipse_distance_along_vector_from_center(x, second_axis_vector[:, 0], second_axis_vector[:, 1])

	# Scale valid points along the major and minor axis vectors by 1/r (that way they are positioned relative to a unit circle, not a wacky ellipse)
	scaled_first_axis_vector = first_axis_vector / first_axis_radius[:, bk.newaxis]
	scaled_second_axis_vector = second_axis_vector / second_axis_radius[:, bk.newaxis]

	transformation_matrices = bk.concatenate([scaled_first_axis_vector[:, :, bk.newaxis], scaled_second_axis_vector[:, :, bk.newaxis]], axis=2) # Converts points on a scaled/rotated ellipse to a unit circle
	transformed_points = bk.matmul(centered_points, transformation_matrices)

	# Now we can do distance to the unit circle
	distance = bk.linalg.norm(transformed_points, axis=2) - 1 # Distance to unit circle is the distance to the origin - 1
	mse = bk.mean(distance**2, axis=1)

	# Add back in the ray origin
	center += ray_origins[valid_indices]

	return valid_indices, x, mse, transformation_matrices, center


ray_matcher = cp.ElementwiseKernel(
	'int32 _i, int32 _j, float32 di, float32 dj, raw uint8 image, int32 image_height, int32 image_width, int32 num_rays',
	'int8 bailed_out, int32 last_i, int32 last_j',
	'''

	int ci_int = _i; // Create writeable copies
	int cj_int = _j;

	float ci = ci_int; // Cast to float (ci, cj stand for center i, j)
	float cj = cj_int;

	unsigned char pixel_value;

	int k = 0;
	for (; k < 100; k ++) { // Bailout of 100 iterations
		ci += di;
		cj += dj;
		ci_int = (int)(ci + 0.5f); // Round ci and cj to integers
		cj_int = (int)(cj + 0.5f);

		pixel_value = image[ci_int*image_width + cj_int];

		if (pixel_value > 128) { // If pixel value became white
			last_i = ci_int;
			last_j = cj_int;
			break;
		}
	}

	bailed_out = (k == 100) || (k == 0); // We bailed out if k is at the bailout limit, or if we immediately hit a white pixel

	'''
)

def show_valid_indices(
	projected_marker,
	valid_ij,
	bk=cp
):
	if bk == cp:
		valid_ij = cp.asnumpy(valid_ij)
		projected_marker = cp.asnumpy(projected_marker)

	projected_marker = np.repeat(projected_marker[:, :, np.newaxis], 3, axis=2)

	annotated = projected_marker.copy()

	for i in range(valid_ij.shape[0]):
		(y, x) = valid_ij[i, :]
		annotated = cv2.circle(annotated, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

	annotated = annotated[:1000, :1000]

	cv2.imshow('image', annotated)
	cv2.waitKey(0)

def find_markers(
	projected_marker,
	num_rays=64,
	percent_symmetry_threshold=0.17,
	mse_threshold=0.01,
	num_sample_offsets=6,
	num_bits=8,
	bk=cp,
	orientation_region_radius_factor=1.8,
	encoding_region_radius_factor=2.5,
):
	image = cp.array(projected_marker, dtype=cp.uint8)

	border_pixels = 101 # 1 more than the bailout
	cropped_shape = np.array(image.shape, dtype=np.int64) - border_pixels*2


	i = cp.arange(cropped_shape[0]) + border_pixels
	j = cp.arange(cropped_shape[1]) + border_pixels
	ij = cp.zeros((cropped_shape[0], cropped_shape[1], 2), dtype=cp.int32)
	ij[:, :, 0] = i[:, cp.newaxis]
	ij[:, :, 1] = j[cp.newaxis, :] # Contains i, j coordinates for each pixel in image

	ray_angles = cp.linspace(0, 2*cp.pi - 2*cp.pi / num_rays, num_rays, dtype=cp.float32)
	ray_vector = cp.column_stack([cp.cos(ray_angles), cp.sin(ray_angles)]) # Contains the change in i and the change in j necessary to move in a direction specified by ray_angles
	ray_delta = ray_vector / cp.max(cp.abs(ray_vector), axis=1)[:, cp.newaxis] # Divide by the maximum, which will guarantee that moving by this amount will land us on a new, adjacent pixel, since we will move in either i or j by 1

	last_ij = cp.zeros((ij.shape[0], ij.shape[1], num_rays, 2), dtype=cp.int32)
	bailed_out = cp.zeros((ij.shape[0], ij.shape[1], num_rays), dtype=cp.int8)

	ray_matcher(
		ij[:, :, cp.newaxis, 0],
		ij[:, :, cp.newaxis, 1],
		ray_delta[cp.newaxis, cp.newaxis, :, 0],
		ray_delta[cp.newaxis, cp.newaxis, :, 1],
		image[:, :, cp.newaxis],
		image.shape[0],
		image.shape[1],
		num_rays,

		bailed_out[:, :, :],
		last_ij[:, :, :, 0],
		last_ij[:, :, :, 1],
	)

	# Convert cupy arrays to numpy for the remaining steps
	if bk == np:
		ij = cp.asnumpy(ij)
		last_ij = cp.asnumpy(last_ij)
		ray_vector = cp.asnumpy(ray_vector)
		bailed_out = cp.asnumpy(bailed_out)
	else:
		projected_marker = image


	# Filter #1: Bailout Check. Invalidate any pixels with a ray that wondered off past the bailout
	valid_indices = bk.where((bailed_out == False).all(axis=2))
	if valid_indices[0].size == 0: # Early return if no elements are valid
		return

	valid_indices = bk.array(valid_indices, dtype=bk.int64) # Convert from tuple to numpy array for easier filtering
	valid_last_ij = last_ij[valid_indices[0], valid_indices[1], :]
	valid_ij = ij[valid_indices[0], valid_indices[1]]

	# show_valid_indices(projected_marker, ij[valid_indices[0], valid_indices[1]], bk=bk)


	# Filter #3: Symmetry check. Filter out entries that are asymmetrical about the center
	delta_from_center = valid_last_ij - valid_ij[:, bk.newaxis, :]
	distance_from_center = bk.linalg.norm(delta_from_center, axis=2)
	symmetric_distance_from_center = bk.roll(distance_from_center, num_rays // 2, axis=1)
	percent_difference_in_symmetry = bk.abs(distance_from_center - symmetric_distance_from_center) / distance_from_center

	average_percent_difference_in_symmetry = percent_difference_in_symmetry.mean(axis=1)
	valid = (average_percent_difference_in_symmetry < percent_symmetry_threshold) # All rays must be within 20 percent symmetry error

	valid_ij = valid_ij[valid, :]
	valid_last_ij = valid_last_ij[valid, :, :] # Apply our symmetry filter
	valid_indices = valid_indices[:, valid]

	average_percent_difference_in_symmetry = average_percent_difference_in_symmetry[valid] # Filter our metrics

	# show_valid_indices(projected_marker, ij[valid_indices[0], valid_indices[1]], bk=bk)

	# Boy oh boy, two filters in one
	# Filter #4: Hyperbola and invalid matrix filter. Remove elements that produce non-full-column-rank matrices and elements that are best represented by hyperbolas. The remaining element indices are in ellipse_valid_indices
	# Filter #5: Ellipse Mean Squared Error. Fit ellipses to the sampled points, if the ellipses have a large error, invalidate them.
	ellipse_valid_indices, coefficients, mse, transformation_matrices, centers = fit_ellipse(valid_last_ij, valid_ij, backend=bk)

	# Invalidate ellipse variables associated with elements with too large of an mse
	mse_valid_indices = bk.where(mse < mse_threshold)
	ellipse_valid_indices = ellipse_valid_indices[mse_valid_indices] # Filter variables returned from fit_ellipse
	coefficients = coefficients[mse_valid_indices]
	transformation_matrices = transformation_matrices[mse_valid_indices]
	centers = centers[mse_valid_indices]

	# Invalidate entries with either too large an mse, best represented by hyperbolas, or result in non-full-column-rank matrices
	valid_indices = valid_indices[:, ellipse_valid_indices]

	valid_ij = valid_ij[ellipse_valid_indices, :]
	valid_last_ij = valid_last_ij[ellipse_valid_indices, :]

	# show_valid_indices(projected_marker, ij[valid_indices[0], valid_indices[1]], bk=bk)

	average_percent_difference_in_symmetry = average_percent_difference_in_symmetry[ellipse_valid_indices]
	mse = mse[mse_valid_indices]

	# The transformation_matrices converted the centered coordinates from ellipse space to unit circle space, so the inverse will take us back
	transformation_matrices_inv = bk.linalg.inv(transformation_matrices)

	# Now sample points on a unit circle then scale them so they are ever so slightly larger.
	num_start_bit_sample_points = num_sample_offsets * num_bits
	unit_circle_angles = bk.linspace(0, 2*bk.pi - 2*bk.pi/num_start_bit_sample_points, num_start_bit_sample_points)
	unit_circle_x = bk.cos(unit_circle_angles)
	unit_circle_y = bk.sin(unit_circle_angles)
	unit_circle_points = bk.column_stack([unit_circle_x, unit_circle_y])
	sample_circle_points = unit_circle_points * orientation_region_radius_factor # Slightly scale up radius

	# Apply the ellipse transformation
	sample_ellipse_points = bk.matmul(sample_circle_points[bk.newaxis, :, bk.newaxis, :], transformation_matrices_inv[:, bk.newaxis, :, :]) # Result is (number of remaining entries, num_start_bit_sample_points, 1, 2),
	sample_ellipse_points = sample_ellipse_points[:, :, 0, :] # Remove extraneous broadcasting dimension
	sample_ellipse_points += centers[:, bk.newaxis, :]
	sample_ellipse_points_int = (sample_ellipse_points + 0.5).astype(bk.int64)

	# Find which sampled ellipse points are white
	sample_ellipse_values = projected_marker[sample_ellipse_points_int[:, :, 0], sample_ellipse_points_int[:, :, 1]]
	sample_ellipse_binary = sample_ellipse_values > 128

	# Find the average phase offset of samples that correctly represent the orientation encoding of 0b01010100
	sample_ellipse_binary = sample_ellipse_binary.reshape(-1, num_bits, num_sample_offsets) # Group samples by the phase offset they represent
	sample_ellipse_binary = bk.swapaxes(sample_ellipse_binary, 1, 2) # (num markers, num phase offsets, num bits)

	first_bit = sample_ellipse_binary[:, :, 0, bk.newaxis]
	first_order_changes = bk.diff(sample_ellipse_binary, axis=-1, append=first_bit) # The diff of a boolean array is True if a change happens and False if it remains constant between two consecutive elements. Append bit_0 to make circular
	first_bit = first_order_changes[:, :, 0, bk.newaxis]
	second_order_changes = bk.diff(first_order_changes.astype(bk.int8), axis=-1, append=first_bit) # Take the second order difference with the changes as an integer array. We are looking for a 1, indicating a first order False to True, indicating a change from a constant region to an alternating region, indicating the second to last False in the 0b000 region of a 0b10100010 or a rolled counterpart
	middle_constant_index = bk.argmax(second_order_changes, axis=-1) # argmax finds the index of the first 1

	marker_arange = bk.arange(valid_ij.shape[0])[:, bk.newaxis, bk.newaxis]
	offset_arange = bk.arange(num_sample_offsets)[bk.newaxis, :, bk.newaxis]
	shifted_bit_indices = bk.zeros((valid_ij.shape[0], num_sample_offsets, num_bits), dtype=bk.int64)
	shifted_bit_indices[:, :, :] = bk.arange(num_bits)[bk.newaxis, bk.newaxis, :]
	shifted_bit_indices += middle_constant_index[:, :, bk.newaxis]
	shifted_bit_indices[shifted_bit_indices >= num_bits] -= num_bits

	rolled_binary = sample_ellipse_binary[marker_arange, offset_arange, shifted_bit_indices] # Rolls each axis left by the corresponding element in first_constant_index


	expected_value = bk.zeros(num_bits, dtype=bool)
	expected_value[::2] = 1 # Alternate zeros and ones
	expected_value[0:2] = 0 # First two bits must be zero
	expected_value = expected_value[bk.newaxis, bk.newaxis, :] # 0b01010100 as an array (with least significant bit first) with proper broadcasting
	is_correct = (rolled_binary == expected_value).all(axis=-1)


	# We interrupt this phase angle calculation to bring you
	# Filter #6: Orientation filter. If the orientation encoding was not found using any of the sampled offsets, remove it
	orientation_found = is_correct.any(axis=-1)
	orientation_found_indices = bk.where(orientation_found)[0]

	coefficients = coefficients[orientation_found_indices]
	transformation_matrices = transformation_matrices[orientation_found_indices]
	transformation_matrices_inv = transformation_matrices_inv[orientation_found_indices]
	centers = centers[orientation_found_indices]
	valid_indices = valid_indices[:, orientation_found_indices]
	valid_ij = valid_ij[orientation_found_indices, :]
	valid_last_ij = valid_last_ij[orientation_found_indices, :]
	middle_constant_index = middle_constant_index[orientation_found_indices]
	is_correct = is_correct[orientation_found_indices]
	sample_ellipse_points = sample_ellipse_points[orientation_found]


	# And back to finding the phase angle
	unit_circle_start_index = middle_constant_index * num_sample_offsets + offset_arange[:, :, 0]
	unit_circle_start_vector = unit_circle_points[unit_circle_start_index, :] # Find the unit circle vector representing the middle of the 0b000 region for each of the sampled orientation encodings

	average_unit_circle_start_vector = (unit_circle_start_vector * is_correct[:, :, bk.newaxis]).sum(axis=1) / is_correct.sum(axis=1)[:, bk.newaxis]
	phase_angle = bk.arctan2(average_unit_circle_start_vector[:, 1], average_unit_circle_start_vector[:, 0])

	# Now we can re-sample the unit circle with the correct phase angle of the start bit, but scale up so we land in the bit layer of the marker
	unit_circle_angles = bk.linspace(0, 2*bk.pi - 2*bk.pi/num_bits, num_bits)
	phase_shifted_unit_circle_angles = unit_circle_angles[bk.newaxis, :] + phase_angle[:, bk.newaxis]

	unit_circle_x = bk.cos(phase_shifted_unit_circle_angles)
	unit_circle_y = bk.sin(phase_shifted_unit_circle_angles)
	unit_circle_points = bk.concatenate([unit_circle_x[:, :, bk.newaxis], unit_circle_y[:, :, bk.newaxis]], axis=2)
	sample_circle_points = unit_circle_points * encoding_region_radius_factor # Scale up radius

	ellipse_bit_layer_points = bk.matmul(sample_circle_points[:, :, bk.newaxis, :], transformation_matrices_inv[:, bk.newaxis, :, :]) # Result is (number of remaining entries, num_bits, 1, 2)
	ellipse_bit_layer_points = ellipse_bit_layer_points[:, :, 0, :] # Remove extraneous broadcasting dimension
	ellipse_bit_layer_points += centers[:, bk.newaxis, :]
	ellipse_bit_layer_points_int = (ellipse_bit_layer_points + 0.5).astype(bk.int64)

	# Finally we can read the marker id
	bit_layer_value = projected_marker[ellipse_bit_layer_points_int[:, :, 0], ellipse_bit_layer_points_int[:, :, 1]]
	bit_layer_bits = bit_layer_value > 128

	bit_index_multiplier = bk.power(2, bk.arange(num_bits))
	marker_ids = bk.matmul(bit_layer_bits, bit_index_multiplier[:, bk.newaxis])
	marker_ids = marker_ids[:, 0] # Remove broadcasting dimension



	# Convert variables to numpy for displaying
	valid_last_ij = cp.asnumpy(valid_last_ij)
	valid_ij = cp.asnumpy(valid_ij)
	projected_marker = cp.asnumpy(projected_marker)
	centers = cp.asnumpy(centers)
	marker_ids = cp.asnumpy(marker_ids)
	coefficients = cp.asnumpy(coefficients)
	average_percent_difference_in_symmetry = cp.asnumpy(average_percent_difference_in_symmetry)
	mse = cp.asnumpy(mse)
	sample_ellipse_points = cp.asnumpy(sample_ellipse_points)
	ellipse_bit_layer_points = cp.asnumpy(ellipse_bit_layer_points)
	is_correct = cp.asnumpy(is_correct)

	# Convert projected marker to 3 channel
	projected_marker = np.repeat(projected_marker[:, :, np.newaxis], 3, axis=2)

	# Display a point at the coordinates where a marker was identified
	annotated = projected_marker.copy()
	annotated = cv2.resize(annotated, (annotated.shape[1] // 4, annotated.shape[0] // 4))
	for i in range(marker_ids.shape[0]):

		y, x = (centers[i] * 0.25 + 0.5).astype(np.int32)
		annotated = cv2.circle(annotated, (x, y), radius=3, color=(255, 255, 0), thickness=-1)

	cv2.imshow('annotated', annotated)
	cv2.waitKey(0)



	# Create info for aabb filtering
	sparse = marker_ids
	sparse_coordinates = valid_ij.astype(np.int64)
	sparse_aabb = np.zeros_like(sparse_coordinates)
	sparse_aabb[:, 0] = np.abs(valid_last_ij[:, :, 0] - valid_ij[:, 0, np.newaxis]).max(axis=1)
	sparse_aabb[:, 1] = np.abs(valid_last_ij[:, :, 1] - valid_ij[:, 1, np.newaxis]).max(axis=1)

	sparse_info = np.concatenate([
		centers,
		sample_ellipse_points.reshape(marker_ids.shape[0], -1),
		ellipse_bit_layer_points.reshape(marker_ids.shape[0], -1)
	], axis=1)

	dense = np.zeros((projected_marker.shape[0], projected_marker.shape[1]), dtype=np.int64)
	sparse = sparse.astype(np.int64)
	sparse_coordinates = sparse_coordinates.astype(np.int64)
	sparse_aabb = sparse_aabb.astype(np.int64)


	filtered_marker_ids, filtered_info, filtered_counts = aabb_filter(
		sparse,
		sparse_coordinates,
		sparse_aabb,
		sparse_info,

		dense=dense
	)

	# Unpack additional info
	filtered_centers = filtered_info[:, :2]
	filtered_orientation_points = filtered_info[:, 2:num_start_bit_sample_points*2+2].reshape(-1, num_start_bit_sample_points, 2)
	filtered_encoding_points = filtered_info[:, num_start_bit_sample_points*2+2:].reshape(-1, num_bits, 2)


	# Print the accuracy
	correct = (filtered_marker_ids == 0x55)
	num_correct = correct.sum()
	print(num_correct)
	print(num_correct / filtered_marker_ids.size)


	# Zoom in on which points were incorrectly classified
	for i in range(filtered_marker_ids.shape[0]):
		# if correct[i]: continue

		print(correct[i])

		scale = 10

		try:
			# coeffs = coefficients[i]
			hr, wr = 20, 20
			origin = (filtered_centers[i] + 0.5).astype(np.int64)
			ci, cj = (filtered_centers[i] + 0.5).astype(np.int64)
			mij = origin - (hr, wr)
			annotated = projected_marker[ci-hr:ci+hr, cj-wr:cj+wr].copy()

			shape = annotated.shape

			annotated = cv2.resize(annotated, (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

			# print(average_percent_difference_in_symmetry[i])
			# print(mse[i])
			print(filtered_marker_ids[i])
			# print(coeffs)

			y, x = ((centers[i] - mij) * scale + (scale / 2) + 0.5).astype(np.int32)
			annotated = cv2.circle(annotated, (x, y), radius=3, color=(255, 255, 0), thickness=-1)


			# for j, ij in enumerate(filtered_valid_last_ij[i]):
			# 	y, x = ((ij - mij) * scale + (scale / 2) + 0.5).astype(np.int32)
			# 	annotated = cv2.circle(annotated, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

			for j, ellipse_point in enumerate(filtered_orientation_points[i]):
				y, x = ((ellipse_point - mij) * scale + (scale / 2) + 0.5).astype(np.int32)
				index = j % num_sample_offsets
				val = int(is_correct[i, index] * 127) + 128

				annotated = cv2.circle(annotated, (x, y), radius=3, color=(0, val, 0), thickness=-1)

			for j, extended_point in enumerate(filtered_encoding_points[i]):
				y, x = ((extended_point - mij) * scale + (scale / 2) + 0.5).astype(np.int32)
				val = j / num_bits * 255
				annotated = cv2.circle(annotated, (x, y), radius=3, color=(val, 0, val), thickness=-1)


			# I = np.arange(annotated.shape[0])
			# J = np.arange(annotated.shape[1])

			# points = np.zeros((annotated.shape[0], annotated.shape[1], 2), dtype=np.int64)
			# points[:, :, 0] = I[:, np.newaxis]
			# points[:, :, 1] = J[np.newaxis, :]

			# I = points[:, :, 0].reshape(-1, 1)
			# J = points[:, :, 1].reshape(-1, 1)

			# X = I / scale - shape[0] / 2
			# Y = J / scale - shape[1] / 2

			# A = np.concatenate([X**2, X*Y, Y**2, X, Y], axis=-1)
			# response = np.matmul(A, coeffs[:, np.newaxis])

			# mask = np.logical_and(response > 0.95, response < 1.05)
			# I = I[mask]
			# J = J[mask]

			# I += scale // 2 # Shift coordinates to pixel centers
			# J += scale // 2

			# annotated[I, J] = (255, 0, 0)

			cv2.imshow("image", annotated)
			cv2.waitKey(0)
		except:
			continue


# Starts a view at the specified frame
def start_video_at_frame(video_path, frame):
	green_screen_capture = cv2.VideoCapture(video_path)
	green_screen_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
	return green_screen_capture

if __name__ == '__main__':
	# projected_marker = cv2.imread("external_scripts/tmp/marker_output/image_0.png")

	cap = start_video_at_frame("resources/marker_laminated_test.mp4", 820)
	res, projected_marker = cap.read()

	projected_marker = cv2.cvtColor(projected_marker, cv2.COLOR_BGRA2GRAY)
	projected_marker = ((projected_marker > 100) * 255).astype(np.uint8)

	# for i in range(10):

	# 	cv2.imshow('p.png', fuck_off[1000:, 1000:])
	# 	cv2.waitKey(0)

	avg = np.array([
		64,     # num_rays
		1.0,   # percent_symmetry_threshold
		0.02,   # mse_threshold
		# 64,     # num_start_bit_sample_points
	])

	num_variations = 1
	space = np.linspace(-1, 1, num_variations)
	space = space[:, np.newaxis]
	space = space * avg[np.newaxis, :] * 0.25
	space += avg[np.newaxis, :]


	start_time = time.time()
	# for i in range(4):
	i = 2
	if True:
		for j in range(num_variations):
			num_rays =                        int(space[j, 0]) if i == 0 else int(avg[0])
			percent_symmetry_threshold =      space[j, 1]      if i == 1 else avg[1]
			mse_threshold =                   space[j, 2]      if i == 2 else avg[2]
			# num_start_bit_sample_points =     int(space[j, 4]) if i == 4 else int(avg[4])

			if i == 0: print(f"\n\nMessing with num_rays {num_rays}")
			if i == 2: print(f"\n\nMessing with percent_symmetry_threshold {percent_symmetry_threshold}")
			if i == 3: print(f"\n\nMessing with mse_threshold {mse_threshold}")
			# if i == 4: print(f"\n\nMessing with num_start_bit_sample_points {num_start_bit_sample_points}")

			find_markers(
				projected_marker.copy(),
				num_rays=num_rays,
				percent_symmetry_threshold=percent_symmetry_threshold,
				mse_threshold=mse_threshold,
				# num_start_bit_sample_points=num_start_bit_sample_points,
				num_bits=7,
				orientation_region_radius_factor=1.7,
				encoding_region_radius_factor=2.5,
			)

	stop_time = time.time()

	print(stop_time - start_time)