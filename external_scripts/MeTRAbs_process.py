
import tensorflow as tf
import numpy as np
import pickle
import cv2
import sys

from helper_functions import print, start_video_at_frame, get_next_frame, read_pickled, write_pickled

def MeTRAbs_predict(model, frames, intrinsics):
	detections, poses3d, poses2d = model.predict_multi_image(frames, intrinsics) # Run MeTRAbs prediction using YOLO detector

	# Find if a person was detected on each frame, if there is, assert only 1 person was detected
	detections_cpu = []
	poses3d_cpu = []
	poses2d_cpu = []
	for i in range(detections.shape[0]): # Iterate over elements in batch
		if len(detections[i]) != 1: # YOLO did not detect any people, or did not detect exaclty one person, there is currently no support for multi person tracking in TraceArmature
			detections_cpu.append(None)
			poses3d_cpu.append(None)
			poses2d_cpu.append(None)
		else:
			detections_cpu.append(detections[i, 0].numpy())

			# Swap y and z axis to match blender coordinates
			poses3d_temp = poses3d[i, 0].numpy()
			temp = poses3d_temp[:, 1].copy()
			poses3d_temp[:, 1] = poses3d_temp[:, 2]
			poses3d_temp[:, 2] = -temp

			poses3d_cpu.append(poses3d_temp)
			poses2d_cpu.append(poses2d[i, 0].numpy())

	return detections_cpu, poses3d_cpu, poses2d_cpu

if __name__ == '__main__':
	(
		start_frame,
		end_frame,
		quality_scale,
		batch_size,
		video_path,
		transpose,
		intrinsics,
		model_path,
	) = read_pickled() # Assuming first pickled object is not end command

	print("Loading model...")
	model = tf.saved_model.load(model_path)
	print("Model loaded")

	cap = start_video_at_frame(video_path, start_frame)
	for i in range(start_frame, end_frame, batch_size):
		cpu_frames = []
		for j in range(batch_size):
			ret, frame = get_next_frame(cap, transpose, scale=quality_scale)
			if not ret:
				break

			cpu_frames.append(frame)
		cpu_frames = np.stack(cpu_frames, axis=0)
		frames = tf.constant(cpu_frames)

		detections, poses3d, poses2d = MeTRAbs_predict(model, frames, intrinsics)
		write_pickled((detections, poses3d, poses2d)) # Write model results to parent process

		# Show preview of first detection in batch
		for j in range(batch_size):
			if detections[j] is not None: # Skip over frames where nothing was detected
				frame = cpu_frames[j, :, :] / 255 # TODO: Make this configurable? Downscale preview by 4 since camera is 4k and monitor is 1080p, convert to float color

				bbox = detections[j]
				bbox = bbox.round().astype(np.int32)

				start_point = (bbox[0], bbox[1])
				end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])

				cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 3) # Draw YOLO human detection

				for k in range(24): # Draw MeTRAbs 2d joint detections
					point = poses2d[j][k, :2]
					point = point.round().astype(np.int32)
					point = (point[0], point[1]) # Cast to tuple

					cv2.circle(frame, point, 5, (255, 0, 255), 3)

				cv2.imshow('Frame', frame)
				cv2.waitKey(1)

		# No need to signal completion because parent process should be iterating over the same frames and batch size

		print(f"Completed {i+1 - start_frame} of {end_frame - start_frame}")

	cap.release()
	print("Complete")