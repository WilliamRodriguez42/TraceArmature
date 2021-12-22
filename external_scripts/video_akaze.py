import numpy as np
import cv2
import os
import time

from helper_functions import print, start_video_at_frame, get_next_frame, read_pickled, write_pickled, look_at_mat4, get_keypoints_and_descriptors, match_descriptors


if __name__ == '__main__':

	video_path = '../resources/suit_test.mp4'

	video_capture = start_video_at_frame(video_path, 0)
	detector = cv2.AKAZE_create()

	prev_keypoints_descriptors = None
	descriptor_match_counts = None
	descriptor_discovery_frame = None
	descriptor_identifiers = None
	descriptor_distance_travelled = None
	prev_image = None


	frame = 0
	while True:
		for i in range(100):
			res, image = get_next_frame(video_capture, (    False,  False,  False))

			if not res:
				exit(0)

		image = image[600:-1000, 500:-500]

		curr_keypoints_descriptors = get_keypoints_and_descriptors(image, detector)

		if prev_keypoints_descriptors is not None:
			(
				prev_matched_indices,
				curr_matched_indices,
			) = match_descriptors(
				prev_keypoints_descriptors,
				curr_keypoints_descriptors,
				return_indices=True,
				# images=(prev_image, image),
			)

			# Increment the distance travelled by a descriptor
			prev_point = np.array([keypoint.pt for keypoint in prev_keypoints_descriptors[0][prev_matched_indices]])
			curr_point = np.array([keypoint.pt for keypoint in curr_keypoints_descriptors[0][curr_matched_indices]])
			descriptor_distance_travelled[prev_matched_indices] += np.linalg.norm(prev_point - curr_point) / image.shape[0]

			# Update the position of keypoints that matched
			prev_keypoints = prev_keypoints_descriptors[0][prev_matched_indices].copy() # Copy previous keypoints before update for showing debug info
			prev_keypoints_descriptors[0][prev_matched_indices] = curr_keypoints_descriptors[0][curr_matched_indices]
			curr_keypoints = prev_keypoints_descriptors[0][prev_matched_indices].copy()
			prev_descriptor_identifiers = descriptor_identifiers[prev_matched_indices].copy()


			# Increment the match counts for descriptors that have successfully matched
			descriptor_match_counts[prev_matched_indices] += 1


			# Remove any descriptors if their counts are 0 (meaning they weren't matched on the current frame, which is the frame that has the highest likelihood of matching)
			frames_since_discovery = frame - descriptor_discovery_frame
			keep_mask = np.logical_or((descriptor_match_counts / frames_since_discovery) > 0.5, frames_since_discovery > 10) # Must be found 50 percent of the time after discovery or have lasted at least 10 frames
			prev_keypoints_descriptors = (prev_keypoints_descriptors[0][keep_mask], prev_keypoints_descriptors[1][keep_mask])
			descriptor_match_counts = descriptor_match_counts[keep_mask]
			descriptor_discovery_frame = descriptor_discovery_frame[keep_mask]
			descriptor_identifiers = descriptor_identifiers[keep_mask]
			descriptor_distance_travelled = descriptor_distance_travelled[keep_mask]

			print("Oldest:", descriptor_discovery_frame.min())


			# Find the new descriptors (descriptors in the current set that didn't match)
			unmatched_curr_keypoints_descriptors = (
				np.delete(curr_keypoints_descriptors[0], curr_matched_indices, axis=0),
				np.delete(curr_keypoints_descriptors[1], curr_matched_indices, axis=0),
			)

			# Initialize an array of zeros to store the number of times the new descriptors have been matched
			num_new_descriptors = unmatched_curr_keypoints_descriptors[0].shape[0]
			new_descriptor_counts = np.zeros(num_new_descriptors, dtype=np.uint64)


			# Append the new zero counts to the end of the existing descriptor match counts
			descriptor_match_counts = np.concatenate([descriptor_match_counts, new_descriptor_counts], axis=0)

			# Append the new descriptors to the end of the existing descriptor list
			prev_keypoints_descriptors = (
				np.concatenate([prev_keypoints_descriptors[0], unmatched_curr_keypoints_descriptors[0]], axis=0),
				np.concatenate([prev_keypoints_descriptors[1], unmatched_curr_keypoints_descriptors[1]], axis=0),
			)

			# Append new descriptor identifiers
			new_descriptor_identifiers = np.arange(num_new_descriptors, dtype=np.uint64) + descriptor_identifiers[-1]
			descriptor_identifiers = np.concatenate([descriptor_identifiers, new_descriptor_identifiers], axis=0)

			# Create and append the frame which these new descriptors where found (the current frame)
			new_descriptor_discovery_frame = np.zeros(num_new_descriptors, dtype=np.uint64) + frame
			descriptor_discovery_frame = np.concatenate([descriptor_discovery_frame, new_descriptor_discovery_frame], axis=0)

			# Create and append a new distance travelled for each of the new descriptors
			new_descriptor_distance_travelled = np.zeros(num_new_descriptors, dtype=np.float64)
			descriptor_distance_travelled = np.concatenate([descriptor_distance_travelled, new_descriptor_distance_travelled], axis=0)

			# Show an image
			# print(descriptor_match_counts[301])
			# print(descriptor_identifiers[301])
			for i in range(prev_keypoints.shape[0]):
				prev_keypoint = tuple(np.round(prev_keypoints[i].pt).astype(np.int64))
				curr_keypoint = tuple(np.round(curr_keypoints[i].pt).astype(np.int64))
				r = int((np.modf((np.sin(prev_descriptor_identifiers[i] * 768.233) + 1) * 43758.3456)[0] * 255).astype(np.int64))
				g = int((np.modf((np.sin(prev_descriptor_identifiers[i] * 894.113) + 1) * 54988.7912)[0] * 255).astype(np.int64))
				b = int((np.modf((np.sin(prev_descriptor_identifiers[i] * 875.564) + 1) * 32689.0352)[0] * 255).astype(np.int64))

				cv2.line(image, prev_keypoint, curr_keypoint, (b, g, r), 20)
			cv2.imshow('Akaze Matches', image[::4, ::4, :])
			cv2.waitKey(1)

		else:
			prev_keypoints_descriptors = curr_keypoints_descriptors

			# Initialize an array to store the number of times each descriptor has been matched
			descriptor_match_counts = np.zeros(prev_keypoints_descriptors[0].shape[0], dtype=np.uint64)

			# Initialize an array representing on which frame these descriptors were found
			descriptor_discovery_frame = np.zeros(prev_keypoints_descriptors[0].shape[0], dtype=np.uint64)

			# Initialize a set of unique identifiers so we can easily track each of these descriptors over time
			descriptor_identifiers = np.arange(prev_keypoints_descriptors[0].shape[0], dtype=np.uint64)

			# Initialize an array to store the distance travelled by a descriptor, which should vouche for its robustness
			descriptor_distance_travelled = np.zeros(prev_keypoints_descriptors[0].shape[0], dtype=np.float64)

		prev_image = image
		frame += 1
