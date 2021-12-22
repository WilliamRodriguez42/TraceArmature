import numpy as np

from helper_functions import print, quaternion_to_mat4, quaternion_mul, quaternion_invert

class CustomTransformModifiers:
	def __init__(self, bone_name_to_index, bone_index_to_name):
		# The custom transform loader is meant to modify bone transform properties right
		# after being sent to the optimize_armature_pytorch_deodr script. This allows
		# for sending transforms in a custom space (like relative to a parent) instead of
		# in world space by default. This is useful when (for instance) trying to limit
		# bone rotations in a parent's local axis. The operations applied here will most
		# likely require a matching operation to be defined in custom_transform_operations.
		# Otherwise the code will use the modified transformation as if it were a world
		# transformation by default.
		self.custom_transform_modifiers_dict = {}

		# Convert quaternions to parent-space for applicable bones
		self.custom_transform_modifiers_dict['lower.arm.L'] = self.quaternion_relative_to_parent
		self.custom_transform_modifiers_dict['lower.arm.R'] = self.quaternion_relative_to_parent
		self.custom_transform_modifiers_dict['elbow.L'] = self.quaternion_relative_to_parent
		self.custom_transform_modifiers_dict['elbow.R'] = self.quaternion_relative_to_parent
		self.custom_transform_modifiers_dict['shin.L'] = self.quaternion_relative_to_parent
		self.custom_transform_modifiers_dict['shin.R'] = self.quaternion_relative_to_parent

		# Store arguments
		self.bone_name_to_index = bone_name_to_index
		self.bone_index_to_name = bone_index_to_name

	def __getitem__(self, item):
		result = self.custom_transform_modifiers_dict.get(item)
		if result is None:
			result = self.passthrough
		return result

	def start_iteration(
		self,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_scales,
		pose_bone_transforms_scaled,
	):
		self.pose_bone_quaternions = pose_bone_quaternions
		self.pose_bone_translations = pose_bone_translations
		self.pose_bone_scales = pose_bone_scales
		self.pose_bone_transforms = pose_bone_transforms_scaled

	def passthrough(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		pose_bone_transform, pose_bone_transform_scaled = quaternion_to_mat4(pose_bone_quaternion[np.newaxis, :], pose_bone_translation[np.newaxis, :], pose_bone_scale[np.newaxis, :])

		return (
			pose_bone_quaternion,
			pose_bone_translation,
			pose_bone_scale,
			pose_bone_transform_scaled,
		)

	def quaternion_relative_to_parent(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		# Find the parent quaternion
		parent_quaternion = self.pose_bone_quaternions[parent_index]

		# Find the inverse of the parent quaternion
		parent_quaternion_inv = quaternion_invert(parent_quaternion)

		# Apply the inverse parent quaternion to get pose bone quaternion in parent space
		pose_bone_quaternion_relative_to_parent = quaternion_mul(parent_quaternion_inv, pose_bone_quaternion)

		# Find the bone transform in world space
		pose_bone_transform, pose_bone_transform_scaled = quaternion_to_mat4(pose_bone_quaternion[np.newaxis, :], pose_bone_translation[np.newaxis, :], pose_bone_scale[np.newaxis, :])

		# Override the world quaternion with the parent-space quaternion
		return (
			pose_bone_quaternion_relative_to_parent,
			pose_bone_translation,
			pose_bone_scale,
			pose_bone_transform_scaled,
		)
