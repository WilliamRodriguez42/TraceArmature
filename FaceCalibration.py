import glob
import os
import bpy
import importlib
import time
import numpy as np
import pdb
import tensorflow as tf

fr = bpy.data.texts["FaceRegression"].as_module()
config = bpy.data.texts["Config"].as_module()
cmvp = bpy.data.texts["CameraMVP"].as_module()


def calibrate():
    with tf.device('/cpu:0'):

        desired_points = np.arange(468)

        face_points = np.zeros((desired_points.size, 4), dtype=np.float32)
        face_points[:, 3] = 1

        canonical_face = bpy.data.objects['canonical_face']

        # j = 0
        # for i in desired_points:
        #     face_points[j, :3] = canonical_face.data.vertices[i].co
        #     j += 1

        start_time = time.time()

        start_frame = bpy.context.scene.frame_start
        end_frame = bpy.context.scene.frame_end
        num_frames = end_frame - start_frame

        rtsl = fr.regress_face(
            face_points,
            start_frame,
            end_frame,
            desired_points=desired_points,
            calibration_offsets=desired_points.size,
            learning_rate=0.03,
            iterations=1000,
        )

        stop_time = time.time()

        print(f"Time taken: {stop_time - start_time}")

        # pdb.set_trace()
        face_points[:, :3] += rtsl.point_offset

        transforms = rtsl.generate_transforms()
        face_points = face_points.dot(transforms[0])

        print(face_points)

        # camera = bpy.data.objects['Camera']
        # MVP = cmvp.projection_matrix(camera)
        # MVP_i = np.linalg.inv(MVP)

        # translations = rtsl.translation.numpy()[:, 0, :] # Remove 2nd axis which was only there for broadcasting
        # w = np.ones((num_frames, 1), dtype=np.float32)
        # world_xyzw = np.concatenate([translations, w], axis=1) # Concatenate w of 1

        # screen_xyzw = world_xyzw.dot(MVP) # Convert to screen coordinates
        # xy = screen_xyzw[:, :2] / screen_xyzw[:, 3, np.newaxis] # Normalize the first two axis (x, y) by w
        # depth = np.linalg.norm(translations - camera.matrix_world.translation, axis=1)

        # rotation_x, rotation_y, rotation_z = rtsl.generate_raw_rotation_xyz()
        # pitch = rotation_x.numpy()
        # roll = rotation_y.numpy()
        # yaw = rotation_z.numpy()


        face_cage = bpy.data.objects['face_cage']
        j = 0
        for i in desired_points:
            face_cage.data.vertices[i].co = face_points[j, :3]
            j += 1

if __name__ == '__main__':
    calibrate()
