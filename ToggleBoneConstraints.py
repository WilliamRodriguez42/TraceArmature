import bpy
import pdb

def set_bone_constraint_value(ao, value):
	for bone in ao.pose.bones:
		for constraint in bone.constraints:
			constraint.mute = value

if __name__ == '__main__':
	armature_object = bpy.data.objects['Armature.metrabs']

	is_enabled = armature_object.pose.bones[0].constraints[0].mute
	toggled_value = not is_enabled

	set_bone_constraint_value(armature_object, toggled_value)
