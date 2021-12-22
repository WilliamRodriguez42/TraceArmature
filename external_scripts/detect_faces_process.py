import cv2
import pdb
import face_recognition
import pathlib
import os
import numpy as np
import argparse

# TODO: Only detect faces for frames in blender frame_start frame_end range

def batch_face_detection(frames, end_frame_count, bboxes):
	batch_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0) # TODO: Config this

	start_frame_count = end_frame_count - len(batch_face_locations)

	for i, face_locations in enumerate(batch_face_locations):
		if len(face_locations) != 1: # Assume there is exactly one face in the frame, otherwise report -1's
			bboxes.append([-1]*4)
			print(f"Did not find a face in frame {start_frame_count + i}")

		bboxes.append(face_locations[0])
		top, right, bottom, left = face_locations[0]

		frame = frames[i][top:bottom, left:right]
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		print(f"Completed frame {start_frame_count + i}")
		cv2.imshow(f'Frame', frame)
		cv2.waitKey(1)


def main(video_path):
	cap = cv2.VideoCapture(video_path)

	bboxes = []
	frames = []
	frame_count = 0

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			break
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		frames.append(image)
		frame_count += 1

		if frame_count % 4 == 0:
			batch_face_detection(frames, frame_count, bboxes)
			frames.clear()

	cap.release()

	os.chdir(pathlib.Path(__file__).parent.absolute())

	with open('tmp/bboxes.npz', 'wb+') as f:
		np.save(f, bboxes)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Creates a numpy file with the top, right, bottom, left dimensions of the face boxes')
	parser.add_argument('video_path', help='Path to the video file')
	args = parser.parse_args()

	main(args.video_path)

	print('Completed parsing faces')