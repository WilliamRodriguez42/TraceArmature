import pickle
import struct
import os
import cv2
import pdb
import math
import numpy as np

import tensorflow as tf

from deodr import LaplacianRigidEnergy
from deodr.tensorflow import CameraTensorflow, LaplacianRigidEnergyTensorflow, Scene3DTensorflow
from deodr.tensorflow.triangulated_mesh_tensorflow import ColoredTriMeshTensorflow as ColoredTriMesh
from deodr.tensorflow.triangulated_mesh_tensorflow import TriMeshTensorflow as TriMesh
from deodr.tensorflow.mesh_fitter_tensorflow import qrot

from helper_functions import print, start_video_at_frame, read_pickled, write_pickled, quaternion_to_mat4
def _quaternion_to_mat4(quaternion, translation, scale=None): # Create a wrapper function so I don't have to hand in tf all the time
	return quaternion_to_mat4(quaternion, translation, tf, scale=scale)

class ArmatureFitter:
	"""Class to fit a deformable mesh to a color image."""

	def __init__(
		self,
		vertices_input,
		vertex_colors,
		faces,

		bone_name_to_index,
		bone_scale_share_map,
		bone_parents,
		bone_quaternions,
		bone_translations,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		bone_vertex_indices,
		bone_vertex_weights,
	):
		self.vertices_input = tf.constant(vertices_input, dtype=tf.float64)
		self.vertex_colors = tf.constant(vertex_colors, dtype=tf.float64)
		self.faces = tf.constant(faces, dtype=tf.int64)
		self.bone_name_to_index = bone_name_to_index

		# Create the deodr mesh
		self.mesh = ColoredTriMesh(faces)
		self.mesh.set_vertices_colors(self.vertex_colors[:, :3]) # Doesn't support alpha channel

		self.scene = Scene3DTensorflow()
		self.scene.set_mesh(self.mesh)

		self.scene.light_directional = tf.constant((0, 0, 0), dtype=tf.float64) # No directional lights, we just want flat colors
		self.scene.light_ambient = tf.constant(1, dtype=tf.float64) # Ambient light to neutral

		# Gather information necessary for calculating / recalculating vertex poses during descent step
		bone_matrices_world = _quaternion_to_mat4(bone_quaternions, bone_translations)
		bone_matrices_world_inv = np.linalg.inv(bone_matrices_world)
		self.bone_matrices_world_inv = tf.constant(bone_matrices_world_inv, tf.float64)

		_, pose_bone_matrices_world = _quaternion_to_mat4(pose_bone_quaternions, pose_bone_translations, pose_bone_initial_scales)
		pose_bone_matrices_world_inv = np.linalg.inv(pose_bone_matrices_world)

		# Find the bone head position relative to the bone's parent
		num_bones = pose_bone_translations.shape[0]
		relative_pose_bone_translations = np.ones((num_bones, 4), dtype=np.float64)
		relative_pose_bone_translations[:, :3] = pose_bone_translations # Default to absolute world translation, this is used in the case a bone has no parent

		has_parent = bone_parents != -1 # Mask of which bones have parents
		valid_parent_indices = bone_parents[has_parent] # Indices of parents for bones that have parents

		parent_pose_bone_matrices_inv = pose_bone_matrices_world_inv[valid_parent_indices, :, :] # Bones that have parents need their translations transformed by the parent inverse

		temp = np.matmul(relative_pose_bone_translations[has_parent, np.newaxis, :], parent_pose_bone_matrices_inv) # Multiply each bone translation by the inverse of the parent matrix
		relative_pose_bone_translations[has_parent, :] = temp[:, 0, :] # Remove the broadcasting axis

		self.relative_pose_bone_translations = tf.constant(relative_pose_bone_translations, tf.float64)

		self.bone_parents = tf.constant(bone_parents, tf.int64)

		self.pose_bone_quaternions = tf.Variable(pose_bone_quaternions, dtype=tf.float64)
		self.pose_bone_initial_scales = tf.constant(pose_bone_initial_scales, dtype=tf.float64) # Each bone is allowed to have a unique scale prior to optimization, the pose_bone_scales will be applied on top of these

		# The vertices_input array is the location of the vertex after the bone matrix world was applied.
		# So if we apply bone_matrices_world_inv to a vertex, then apply the pose_bone_quaternion / translation,
		# we will be at the expected final vertex location.
		vertices_bone_inverted = np.zeros((bone_vertex_indices.shape[0], vertices_input.shape[0], 4), dtype=np.float64)
		for bone_index in range(bone_matrices_world.shape[0]):

			ones = np.ones((bone_vertex_indices[bone_index].shape[0], 1), dtype=np.float64)
			bone_vertices = vertices_input[bone_vertex_indices[bone_index], :]
			bone_vertices_4 = np.concatenate([bone_vertices, ones], axis=1)
			bone_vertices_weighed = bone_vertices_4 * bone_vertex_weights[bone_index][:, np.newaxis] # We can apply the scalar weight now since any following matrix multiplications are commutative with scalar multiplication

			inverted = bone_vertices_weighed.dot(self.bone_matrices_world_inv[bone_index])

			vertices_bone_inverted[bone_index, bone_vertex_indices[bone_index], :] = inverted

		# Normalize bone weights so they guarantee to sum to 1
		weights = vertices_bone_inverted[:, :, 3]
		vertex_total_weight = weights.sum(axis=0)
		vertices_bone_inverted /= vertex_total_weight[np.newaxis, :, np.newaxis]

		self.vertices_bone_inverted = tf.constant(vertices_bone_inverted, tf.float64)

		# Need to make up a variable to represent an offset for the entire armature
		self.offset = tf.Variable(tf.zeros(3, dtype=tf.float64))

		# Need to create bone scale variables, but only enough to cover the bone_scale_share_map

		# The bone scale share map maps bone names to shared scale indices, we need it to map bone indices to shared scale indices
		self.bone_index_to_scale_index = {bone_name_to_index[name]: bone_scale_share_map[name] for name in bone_name_to_index.keys()}
		num_shared_bones = np.max(list(self.bone_index_to_scale_index.values())) + 1
		self.pose_bone_scales = tf.Variable(tf.ones((num_shared_bones, 3), dtype=tf.float64))

		# Identify trainable variables
		self.trainable_variables = [
			self.offset,
			self.pose_bone_quaternions,
			self.pose_bone_scales,
		]

		# boundaries = [10, 20]
		# values = [1e-3, 1e-2, 1e-3]
		# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

		self.optimizer = tf.keras.optimizers.Adam(
			learning_rate=6e-3,
			beta_1=0.9,
			beta_2=0.9,
			epsilon=1e-07,
			amsgrad=False,
		)

	def set_background_color(self, background_color):
		self.scene.set_background_color(background_color)

	def set_projection_matrix(self, intrinsics, extrinsics, shape):
		self.camera = CameraTensorflow(
			extrinsic=extrinsics,
			intrinsic=intrinsics,
			width=shape[1],
			height=shape[0],
			distortion=None,
		)

	def step(self, combined_image, video_image, green_screen_image, chroma_mask):
		with tf.GradientTape() as tape:

			# Divide quaternions by their magnitude
			norm = tf.norm(self.pose_bone_quaternions, axis=1)
			pose_bone_quaternions = self.pose_bone_quaternions / norm[:, tf.newaxis]

			# Recursively modify translation, but use world rotation
			pose_bone_transforms = []
			pose_bone_transforms_scaled = []
			pose_bone_translations = [] # Assign this variable now for debugging, it will not be used in this calculation
			for i in range(pose_bone_quaternions.shape[0]):
				pose_bone_quaternion = pose_bone_quaternions[i]
				relative_pose_bone_translation = self.relative_pose_bone_translations[i]
				pose_bone_scale = self.pose_bone_scales[self.bone_index_to_scale_index[i]] * self.pose_bone_initial_scales[i]

				if self.bone_parents[i] == -1: # If there is no parent
					pose_bone_translation = relative_pose_bone_translation[:3] + self.offset # The relative translation is the same as the absolute world translation (but also include the offset)
					pose_bone_translations.append(tf.concat([pose_bone_translation, [1]], axis=0))
				else: # Otherwise convert translation relative to parent to absolute world translation
					parent_pose_bone_transform_scaled = pose_bone_transforms_scaled[self.bone_parents[i]] # Because pose bones are ordered by DFS, pose_bone_transforms[self.bone_parents[i]] is guaranteed to exist (if not -1)
					pose_bone_translation = tf.linalg.matmul(relative_pose_bone_translation[tf.newaxis, :], parent_pose_bone_transform_scaled)[0] # Index 0 to remove broadcasting axis
					pose_bone_translations.append(pose_bone_translation)

				pose_bone_transform, pose_bone_transform_scaled = _quaternion_to_mat4(pose_bone_quaternion[tf.newaxis, :], pose_bone_translation[tf.newaxis, :], pose_bone_scale[tf.newaxis, :])
				pose_bone_transforms.append(pose_bone_transform[0, :, :]) # Index 0 to remove broadcasting axis
				pose_bone_transforms_scaled.append(pose_bone_transform_scaled[0, :, :])

			vertices_output = tf.linalg.matmul(self.vertices_bone_inverted, pose_bone_transforms_scaled)
			vertices_output = tf.math.reduce_sum(vertices_output, axis=0)
			vertices_output = vertices_output[:, :3]

			# View from 90 degrees about z axis, so we can check if depth is correct
			# vertices_output -= self.relative_pose_bone_translations[0, :3]
			# vertices_output = tf.stack([vertices_output[:, 1], -vertices_output[:, 0], vertices_output[:, 2]])
			# vertices_output = tf.transpose(vertices_output)
			# vertices_output += self.relative_pose_bone_translations[0, :3]

			self.mesh.set_vertices(vertices_output)
			render = self.scene.render(self.camera)

			mse_image = tf.math.reduce_sum((render - tf.constant(combined_image)) ** 2, axis=2) # Mean squared error between render and image, doing reduce sum only on axis 2 so we can display this result to the user

			green_mask = (green_screen_image[:, :, 1] > 0.5) # Fuzzy compare if pixel is green
			should_be_green = tf.boolean_mask(render, green_mask)
			green_screen_loss = tf.math.reduce_sum((should_be_green - [[[0, 1, 0]]]) ** 2) # Add an additional penalty for overlapping the green screen

			loss = tf.reduce_sum(mse_image) + green_screen_loss * 2


			grads = tape.gradient(loss, self.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		return render, mse_image, loss, pose_bone_transforms, self.pose_bone_scales


def get_next_frame(video_capture, green_screen_capture, transpose, scale):

	_, video_image = video_capture.read()
	_, green_screen_image = green_screen_capture.read()

	if transpose[0]:
		video_image = cv2.transpose(video_image)
		green_screen_image = cv2.transpose(green_screen_image)
	if transpose[1]:
		video_image = cv2.flip(video_image, -1)
		green_screen_image = cv2.flip(green_screen_image, -1)
	if transpose[2]:
		video_image = cv2.flip(video_image, 0)
		green_screen_image = cv2.flip(green_screen_image, 0)

	# Downscale images
	image_shape = np.array(video_image.shape, dtype=np.float64)
	image_shape *= scale
	image_shape = image_shape.round().astype(np.int32)
	image_shape = (image_shape[1], image_shape[0])

	video_image = cv2.resize(video_image, image_shape, cv2.INTER_NEAREST)
	green_screen_image = cv2.resize(green_screen_image, image_shape, cv2.INTER_NEAREST)

	# Overlay video image with green screen contents where present
	combined_image = video_image.copy()
	chroma_mask = (green_screen_image > 128).any(axis=2) # Fuzzy compare for if any color is present
	combined_image[chroma_mask, :] = green_screen_image[chroma_mask, :]

	# Flip rgb, cast to float64
	video_image = video_image[:, :, ::-1].astype(np.float64) / 255
	green_screen_image = green_screen_image[:, :, ::-1].astype(np.float64) / 255
	combined_image = combined_image[:, :, ::-1].astype(np.float64) / 255

	return combined_image, video_image, green_screen_image, chroma_mask

def handle_stdin(subprocess_args):
	# Unpack subprocess args
	(
		bone_name_to_index,
		bone_scale_share_map,
		bone_parents,
		bone_quaternions,
		bone_translations,
		bone_vertex_indices,
		bone_vertex_weights,
		vertices,
		vertex_colors,
		polygons,
		video_path,
		video_green_screen_path,
		video_crop_transpose,
	) = subprocess_args

	# Get additional arguments for this frame
	read_result = read_pickled()
	if read_result is None: return False # Break from calling function loop

	(
		intrinsics,
		extrinsics,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		quality_scale,
		frame,
	) = read_result

	# Grab the current frame (TODO: No need to recapture and release the video every time, frame indices should be sequential)
	video_capture = start_video_at_frame(video_path, frame)
	green_screen_capture = start_video_at_frame(video_green_screen_path, frame)
	combined_image, video_image, green_screen_image, chroma_mask = get_next_frame(video_capture, green_screen_capture, video_crop_transpose, quality_scale)
	video_capture.release()
	green_screen_capture.release()

	# Specify default arguments for the ArmatureFitter
	euler_init = np.array([np.pi/2, 0, 0])

	fitter = ArmatureFitter(
		vertices,
		vertex_colors,
		polygons,

		bone_name_to_index,
		bone_scale_share_map,
		bone_parents,
		bone_quaternions,
		bone_translations,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_initial_scales,
		bone_vertex_indices,
		bone_vertex_weights,
	)

	fitter.set_projection_matrix(intrinsics, extrinsics, video_image.shape)
	fitter.set_background_color(np.array([0, 1.0, 0.0]))

	for niter in range(50): # TODO: Config this

		render, diff_image, loss, pose_bone_transforms, pose_bone_scales = fitter.step(combined_image, video_image, green_screen_image, chroma_mask)

		# Render the image
		render = render.numpy()
		diff_image = diff_image.numpy()
		display_image = np.column_stack(
			(combined_image, render, np.tile(diff_image[:, :, None], (1, 1, 3)))
		)

		width = display_image.shape[1]
		height = display_image.shape[0]
		target_width = 1000
		target_height = int(target_width / width * height)

		display_image = cv2.resize(display_image, (target_width, target_height))

		cv2.imshow(
			"animation",
			display_image[:, :, ::-1],
		)

		cv2.waitKey(1)
	# cv2.waitKey(0)

	# Convert pose bone transforms to pure numpy array instead of list of arrays
	pose_bone_transforms = np.array(pose_bone_transforms, dtype=np.float64)
	pose_bone_scales = pose_bone_scales.numpy()

	# Pickle generated matrices and send over stdout
	result = (
		pose_bone_transforms,
		pose_bone_scales,
		float(loss), # Convert from tensor float to python float
	)
	write_pickled(result)

	return True # Continue looping in calling function

if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	# # Use experimental gpu memory growth so we can run more than one process at a time
	# gpus = tf.config.list_physical_devices('GPU')
	# # Currently, memory growth needs to be the same across GPUs
	# for gpu in gpus:
	# 	tf.config.experimental.set_memory_growth(gpu, True)
	# 	logical_gpus = tf.config.list_logical_devices('GPU')

	# Read subprocess args
	subprocess_args = read_pickled()

	# Main loop
	while handle_stdin(subprocess_args):
		pass