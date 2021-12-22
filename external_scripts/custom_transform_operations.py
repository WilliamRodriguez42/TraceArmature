import torch
import numpy as np

from helper_functions import print, quaternion_to_mat4, quaternion_mul
def _quaternion_to_mat4(quaternion, translation, scale=None): # Create a wrapper function so I don't have to hand in torch all the time
	return quaternion_to_mat4(quaternion, translation, scale=scale, backend=torch)
def _quaternion_mul(a, b):
	return quaternion_mul(a, b, backend=torch)

class CustomTransformOperations:
	def __init__(self, bone_name_to_index, bone_index_to_name):
		# This is the main export of this class. This dictionary will map bone names
		# to transform functions, the goal is to take in some information about a bone
		# and output its transform. Additional modifications can be done such as limiting
		# the transform rotation to a certain axis or limiting a component to a certain
		# range via an activation function. The passthrough function is assigned by
		# default and will simply convert a quaternion, translation, and scale into a
		# transform. The resulting transform should be the application of scale, rotation,
		# and translation in that order (rather than the traditional rotation, then translation
		# then scale).
		self.custom_transform_operations_dict = {}

		# The following bones are not allowed to rotate in any axis other than their paren't z axis
		self.custom_transform_operations_dict['lower.arm.L'] = self.limit_to_parent_z
		self.custom_transform_operations_dict['lower.arm.R'] = self.limit_to_parent_z
		self.custom_transform_operations_dict['elbow.L'] = self.elbow_l_join_half_angle
		self.custom_transform_operations_dict['elbow.R'] = self.elbow_r_join_half_angle
		self.custom_transform_operations_dict['shin.L'] = self.limit_to_parent_z
		self.custom_transform_operations_dict['shin.R'] = self.limit_to_parent_z

		# Additional potentially useful information
		self.bone_name_to_index = bone_name_to_index
		self.bone_index_to_name = bone_index_to_name

		# Variable to cache the relative quaternions
		self.relative_quaternions = {}

	def __getitem__(self, item): # Allow for indexing the class variable directly instead of going through the extremely longly named dict variable
		result = self.custom_transform_operations_dict.get(item)
		if result is None:
			result = self.passthrough
		return result

	def start_iteration(
		self,
		pose_bone_quaternions,
		pose_bone_translations,
		pose_bone_scales,
		pose_bone_transforms,
		pose_bone_transforms_scaled,
	):
		# This function is intended to prevent every other function in this class from having a million
		# arguments by storing the pass-by-reference types for a given iteration.
		self.pose_bone_quaternions = pose_bone_quaternions
		self.pose_bone_translations = pose_bone_translations
		self.pose_bone_scales = pose_bone_scales
		self.pose_bone_transforms = pose_bone_transforms
		self.pose_bone_transforms_scaled = pose_bone_transforms_scaled

	def passthrough(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		pose_bone_transform, pose_bone_transform_scaled = _quaternion_to_mat4(pose_bone_quaternion[np.newaxis, :], pose_bone_translation[np.newaxis, :], pose_bone_scale[np.newaxis, :])
		return (
			pose_bone_quaternion,
			pose_bone_transform[0, :, :],
			pose_bone_transform_scaled[0, :, :],
		)

	def limit_to_parent_z(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion, # Assumed that the custom_transform_modifiers successfully wrote this value as local with parent
		pose_bone_translation,
		pose_bone_scale,
	):
		# Zero out the x and y axis components of the quaternion
		pose_bone_quaternion_limited = torch.zeros_like(pose_bone_quaternion)

		pose_bone_quaternion_limited[0] = pose_bone_quaternion[0]
		pose_bone_quaternion_limited[3] = pose_bone_quaternion[3]

		norm = torch.norm(pose_bone_quaternion_limited)
		pose_bone_quaternion = pose_bone_quaternion_limited / norm # Renormalize pose_bone_quaternion
		self.relative_quaternions[bone_index] = pose_bone_quaternion

		# Convert quaternion in parent space to world space
		pose_bone_quaternion = _quaternion_mul(self.pose_bone_quaternions[parent_index], pose_bone_quaternion)

		# Passthrough with component quaternion
		return self.passthrough(
			bone_index,
			parent_index,
			pose_bone_quaternion,
			pose_bone_translation,
			pose_bone_scale,
		)

	def joint_half_angle( # Finds the rotation of something like the elbow joint that lies between two bones and must point directly outwards
		self,
		bone_index,
		parent_index,
		pointing_index,
		angle_offset,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		# Find the pointing bone (If this were an elbow joint, the pointing bone would be the lower arm)
		pointing_quaternion = self.relative_quaternions[pointing_index] # This should also be relative to parent with zeroed x and y axis

		# Find the angle of the pointing quaternion
		pointing_angle = 2 * torch.acos(pointing_quaternion[0])

		# Create a new quaternion that is half of the pointing angle +- 90 degrees
		pose_bone_quaternion_half_angle = pointing_angle*0.5 + angle_offset
		pose_bone_quaternion_half_w = torch.cos(pose_bone_quaternion_half_angle / 2)
		pose_bone_quaternion_half_z = torch.sqrt(1 - pose_bone_quaternion_half_w*pose_bone_quaternion_half_w)

		zero = torch.tensor(0)
		pose_bone_quaternion_half = torch.stack([pose_bone_quaternion_half_w, zero, zero, pose_bone_quaternion_half_z])

		# Convert quaternion in parent space to world space
		pose_bone_quaternion = _quaternion_mul(self.pose_bone_quaternions[parent_index], pose_bone_quaternion_half)

		# Passthrough with component quaternion
		return self.passthrough(
			bone_index,
			parent_index,
			pose_bone_quaternion,
			pose_bone_translation,
			pose_bone_scale,
		)

	def elbow_l_join_half_angle(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		return self.joint_half_angle(
			bone_index,
			parent_index,
			self.bone_name_to_index['lower.arm.L'],
			3*torch.pi/2,
			pose_bone_quaternion,
			pose_bone_translation,
			pose_bone_scale,
		)

	def elbow_r_join_half_angle(
		self,
		bone_index,
		parent_index,
		pose_bone_quaternion,
		pose_bone_translation,
		pose_bone_scale,
	):
		return self.joint_half_angle(
			bone_index,
			parent_index,
			self.bone_name_to_index['lower.arm.R'],
			-torch.pi/2,
			pose_bone_quaternion,
			pose_bone_translation,
			pose_bone_scale,
		)