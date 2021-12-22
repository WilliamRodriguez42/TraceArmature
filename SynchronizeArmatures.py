import bpy
import pdb

# Copy bone parameters from source to target armature
def copy_armature_parameters(source, target):
	# Select and edit the source object
	bpy.context.view_layer.objects.active = source
	bpy.ops.object.mode_set(mode='EDIT')

	source_data = []
	for bone in source.data.edit_bones:
		source_data.append((
			bone.head,
			bone.tail,
			bone.roll,
		))

	# Select and edit the target object
	bpy.context.view_layer.objects.active = target
	bpy.ops.object.mode_set(mode='EDIT')

	for bone, sd in zip(target.data.edit_bones, source_data):
		bone.head = sd[0]
		bone.tail = sd[1]
		bone.roll = sd[2]

	# Return to object mode
	bpy.ops.object.mode_set(mode='OBJECT')

if __name__ == '__main__':
	# armature_temp = bpy.data.objects['Armature.optimized.001']
	armature = bpy.data.objects['Armature.metrabs']
	armature_optimized = bpy.data.objects['Armature.optimized']
	armature_retargeted = bpy.data.objects['Armature.retargeted']

	copy_armature_parameters(armature, armature_optimized)
	copy_armature_parameters(armature, armature_retargeted)
