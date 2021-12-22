import bpy
import sys
import numpy as np
import math
import mathutils

oadr = bpy.data.texts["OptimizeArmatureDiffRender"].as_module()

config = bpy.data.texts["Config"].as_module()
cmvp = bpy.data.texts["CameraMVP"].as_module()

# The following nasty import is done because blender will not reload a library if it is edited
# after blender has already loaded it, even if it was loaded in a previous run of this script.
# Using just plain old from helper_functions import read_pickled would require a restart of blender
# every time that function is edited :(
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)
quaternion_to_mat4 = hf.quaternion_to_mat4

if __name__ == '__main__':
	print()

	bpy.ops.object.mode_set(mode='OBJECT') # Needs to be in object mode to get vertex colors

	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1

	armature_metrabs = bpy.data.objects['Armature.metrabs']
	armature_optimized = bpy.data.objects['Armature.optimized']

	bone_name_to_scale_index = {bone.name: i for i, bone in enumerate(armature_optimized.pose.bones)}

	pose_bone_scales = np.ones((len(armature_optimized.pose.bones), 3), dtype=np.float64)

	for frame in range(start_frame, end_frame, 5):
		bpy.context.scene.frame_set(frame+1)

		for pose_bone in armature_metrabs.pose.bones:
			pose_bone_transform = bpy.data.objects[pose_bone.name + '.transform']

			pose_bone_transform.matrix_world = pose_bone.matrix

			pose_bone_transform.keyframe_insert(data_path='location', frame=frame+1)
			pose_bone_transform.keyframe_insert(data_path='rotation_quaternion', frame=frame+1)
			pose_bone_transform.keyframe_insert(data_path='scale', frame=frame+1)

		if frame % 5 == 0:
			sys.stdout.write(f"Percent complete: {(frame - start_frame) / (end_frame - start_frame) * 100:.2f}\r")

	print()
	print("DONE")