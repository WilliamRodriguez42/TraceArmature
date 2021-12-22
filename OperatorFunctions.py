import bpy

def delete_keyframe_and_move(delta):
    scene = bpy.data.scenes['Scene']
    current_frame = scene.frame_current

    bpy.ops.ed.undo_push()

    active_object = bpy.context.active_object

    if 'Armature' in active_object.name:

        for pose_bone in bpy.context.selected_pose_bones_from_active_object:

            if pose_bone.rotation_mode == 'QUATERNION':
                pose_bone.keyframe_delete(data_path='rotation_quaternion', frame=current_frame)
            else:
                pose_bone.keyframe_delete(data_path='rotation_euler', frame=current_frame)

            pose_bone.keyframe_delete(data_path='location', frame=current_frame)
            pose_bone.keyframe_delete(data_path='scale', frame=current_frame)

    else:

        selected_objects = bpy.context.selected_objects

        for obj in selected_objects:

            if obj is not None:

                if obj.rotation_mode == 'QUATERNION':
                    obj.keyframe_delete(data_path='rotation_quaternion', frame=current_frame)
                else:
                    obj.keyframe_delete(data_path='rotation_euler', frame=current_frame)

                obj.keyframe_delete(data_path='location', frame=current_frame)
                obj.keyframe_delete(data_path='scale', frame=current_frame)

    bpy.context.scene.frame_set(current_frame+delta)


class DeleteKeyframeAndAdvance(bpy.types.Operator):
    bl_idname = "wm.delete_keyframe_and_advance"
    bl_label = "Delete keyframe and advance the playhead"

    def execute(self, context):
        delete_keyframe_and_move(1)
        return {'FINISHED'}

class DeleteKeyframeAndRetreat(bpy.types.Operator):
    bl_idname = "wm.delete_keyframe_and_retreat"
    bl_label = "Delete keyframe and retreat the playhead"

    def execute(self, context):
        delete_keyframe_and_move(-1)
        return {'FINISHED'}

class ResetMetrabsAnimationLayers(bpy.types.Operator):
    bl_idname = "wm.reset_metrabs_animation_layers"
    bl_label = "Delete animation layers for metrabs points and create new ones"

    def execute(self, context):
        collection = bpy.data.collections['MeTRAbs_points']

        for obj in collection.all_objects:
            context.view_layer.objects.active = obj

            while len(obj.Anim_Layers) > 0:
                bpy.ops.anim.remove_anim_layer()

            # if len(obj.Anim_Layers) < 2:
            #     bpy.ops.anim.add_anim_layer()

        return {'FINISHED'}

class InterpolateMarkers(bpy.types.Operator):
    bl_idname = "wm.interpolate_markers"
    bl_label = "Interpolate marker positions"

    def interpolate_markers(self, markers, first_marker, last_marker):
        bpy.ops.ed.undo_push()

        first_frame = first_marker.frame
        last_frame = last_marker.frame
        frame_range = last_frame - first_frame

        first_coordinates = first_marker.co.copy()
        last_coordinates = last_marker.co.copy()
        delta_coordinates = last_coordinates - first_coordinates

        delta_frame = 1
        for frame in range(first_frame + 1, last_frame):
            interpolation_percent = delta_frame / frame_range

            interpolated_coordinates = delta_coordinates * interpolation_percent + first_coordinates
            new_marker = markers.insert_frame(frame, co=interpolated_coordinates.copy())
            new_marker.pattern_corners = first_marker.pattern_corners
            new_marker.search_max = first_marker.search_max
            new_marker.search_min = first_marker.search_min
            new_marker.is_keyed = first_marker.is_keyed

            delta_frame += 1

    def execute(self, context):
        scene = bpy.data.scenes['Scene']
        current_frame = scene.frame_current

        config = bpy.data.texts["Config"].as_module()
        tracking = config.video.tracking
        track = tracking.tracks.active

        previous_marker = None
        for marker in track.markers:
            if not marker.mute:
                if marker.frame >= current_frame and previous_marker is not None:
                    self.interpolate_markers(track.markers, previous_marker, marker)
                    break
                previous_marker = marker

        return {'FINISHED'}

class ReplaceOptimizedReference(bpy.types.Operator):
    bl_idname = "wm.replace_optimized_reference"
    bl_label = "Replace optimized and reference"

    def delete_bone_constraints(self, ao):
        for bone in ao.pose.bones:
            for constraint in bone.constraints:
                bone.constraints.remove(constraint)

    def select_hierarchy(self, obj):
        obj.select_set(True)

        for child in obj.children:
            self.select_hierarchy(child)

    def execute(self, context):
        active_armature = bpy.context.view_layer.objects.active

        if active_armature is not None and active_armature.type == "ARMATURE":
            # Deselect all objects
            for obj in bpy.context.selected_objects:
                obj.select_set(False)

            # Select only the active object's hierarchy
            self.select_hierarchy(active_armature)

            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

            # Duplicate the selected objects
            bpy.ops.object.duplicate()

            # Replace the newly duplicated object names with .optimized
            for obj in bpy.context.selected_objects:
                obj.name = obj.name.replace('.metrabs.001', '.optimized')

            # Duplicate these duplicated objects
            bpy.ops.object.duplicate()

            # Replace the newly duplicated object names with .reference
            for obj in bpy.context.selected_objects:
                obj.name = obj.name.replace('.optimized.001', '.reference')

            # Get the new armatures
            optimized_armature_name = active_armature.name.replace('.metrabs', '.optimized')
            reference_armature_name = active_armature.name.replace('.metrabs', '.reference')
            optimized_armature = bpy.data.objects[optimized_armature_name]
            reference_armature = bpy.data.objects[reference_armature_name]

            # Delete all bone constraints on the new armatures
            self.delete_bone_constraints(optimized_armature)
            self.delete_bone_constraints(reference_armature)

            # Create new optimized transforms for each bone
            for pose_bone in optimized_armature.pose.bones:
                empty_object_name = pose_bone.name + '.transform'
                if bpy.data.objects[empty_object_name] is not None: # Skip creating the object if it already exists
                    continue

                bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.1)

                empty_object = bpy.context.active_object
                empty_object.name = empty_object_name
                empty_object.parent = optimized_armature
                empty_object.rotation_mode = 'QUATERNION'
                empty_object.hide_set(True)

                copy_transform_constraint = pose_bone.constraints.new('COPY_TRANSFORMS')
                copy_transform_constraint.target = empty_object
                copy_transform_constraint.mute = True

        return {'FINISHED'}

class SetTiePointTarget(bpy.types.Operator):
    bl_idname = "wm.set_tie_point_target"
    bl_label = "Set Tie Point Targets"

    def execute(self, context):
        active_model = bpy.context.view_layer.objects.active

        collection = bpy.data.collections['TiePoints']

        for obj in collection.all_objects:
            obj.constraints['Shrinkwrap'].target = active_model

        return {'FINISHED'}

class ToggleBoneConstraints(bpy.types.Operator):
    bl_idname = "wm.toggle_bone_constraints"
    bl_label = "Toggle All Bone Constraints"

    def execute(self, context):
        active_armature = bpy.context.view_layer.objects.active

        if active_armature is not None and active_armature.type == "ARMATURE":
            is_enabled = active_armature.pose.bones[0].constraints[0].mute
            toggled_value = not is_enabled

            for bone in active_armature.pose.bones:
                for constraint in bone.constraints:
                    constraint.mute = toggled_value

        return {'FINISHED'}

class FixQuaternions(bpy.types.Operator):
    bl_idname = "wm.fix_quaternions"
    bl_label = "Fix Quaternions"

    def execute(self, context):
        start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
        end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
        assert start_frame >= 0 # Assume blender starts at at least frame 1


        armature_metrabs = bpy.data.objects['Armature.metrabs']

        prev_quaternions = {}
        for frame in range(start_frame, end_frame, 5):
            bpy.context.scene.frame_set(frame+1)

            for pose_bone in armature_metrabs.pose.bones:
                pose_bone_transform = bpy.data.objects[pose_bone.name + '.transform']

                curr_quaternion = pose_bone_transform.rotation_quaternion.copy()

                prev_quaternion = prev_quaternions.get(pose_bone.name)

                if prev_quaternion is not None:
                    quaternion_diff = prev_quaternion.inverted() @ curr_quaternion
                    if quaternion_diff.w < 0:
                        curr_quaternion.negate()

                pose_bone_transform.rotation_quaternion = curr_quaternion

                pose_bone_transform.keyframe_insert(data_path='rotation_quaternion', frame=frame+1)

                prev_quaternions[pose_bone.name] = curr_quaternion

        return {'FINISHED'}

def main():
    bpy.utils.register_class(DeleteKeyframeAndAdvance)
    bpy.utils.register_class(DeleteKeyframeAndRetreat)
    bpy.utils.register_class(ResetMetrabsAnimationLayers)
    bpy.utils.register_class(InterpolateMarkers)
    bpy.utils.register_class(ReplaceOptimizedReference)
    bpy.utils.register_class(SetTiePointTarget)
    bpy.utils.register_class(ToggleBoneConstraints)
    bpy.utils.register_class(FixQuaternions)

if __name__ == '__main__':
    main()