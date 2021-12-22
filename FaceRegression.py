import json
import numpy as np
import tensorflow as tf
import sys
import bpy
import pdb
import mathutils
import math

mpr = bpy.data.texts["MultiPointRegression"].as_module()
cmvp = bpy.data.texts["CameraMVP"].as_module()
gyt = bpy.data.texts["GenerateYTrue"].as_module()
config = bpy.data.texts["Config"].as_module()

def mean_distance_loss(y_true, pred):
    square_distance = tf.math.reduce_sum(tf.math.square(y_true - pred), axis=2)
    mean = tf.math.reduce_mean(square_distance, axis=1)

    return mean

def regress_face(
    face_points,
    start_frame,
    end_frame,
    desired_points=None,
    initialization_features=None,
    rtsl=None,
    use_rotation=True,
    learning_rate=0.1,
    calibration_offsets=None,
    iterations=1000,
):
    # Get camera model view projection matrix
    camera = bpy.data.objects["Camera"]
    MVP = cmvp.projection_matrix(camera)
    MVP_i = np.linalg.inv(MVP)
    MVP_tf = tf.constant(MVP)

    # Generate truth data
    y_true, valid_frames = gyt.generate_y_true(config.VIDEO_PATH)
    num_frames = y_true.shape[0]

    # Convert to tf constants
    print(desired_points.shape)
    print(y_true.shape)
    y_true = tf.constant(y_true[:, desired_points, :], dtype=tf.float32)

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Format initialization features
    world_translation = None
    yaw = None
    pitch = None
    roll = None
    if initialization_features is not None:
        head_features, head_depth = initialization_features

        # First we must compute xyz coordinates in world space based on xy camera position and depth from camera
        world_translation = cmvp.camera_xy_depth_to_world(camera.matrix_world.translation, head_features[:, :2], head_depth[:, np.newaxis], MVP_i)

        # Now get yaw pitch roll
        yaw = head_features[:, 2]
        pitch = head_features[:, 3]
        roll = head_features[:, 4]

    # Training
    if rtsl is None:
        rtsl = mpr.RotationTranslationLayer(
            MVP_tf,
            num_frames,
            translation=world_translation,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            calibration_offsets=calibration_offsets,
            use_rotation=use_rotation,
            rotation_limits=False,
        )

    # Training
    for i in range(iterations):

        with tf.GradientTape() as tape:
            res = rtsl(face_points)
            loss = tf.keras.losses.MSE(y_true, res)

        gradients = tape.gradient(loss, rtsl.trainable_variables)
        grad_vars = [(grad, var) for (grad, var) in zip(gradients, rtsl.trainable_variables) if grad is not None]

        optimizer.apply_gradients(grad_vars)

    return rtsl
