import bpy
import cv2

config = bpy.data.texts["Config"].as_module()

# The following nasty import is done because blender will not reload a library if it is edited
# after blender has already loaded it, even if it was loaded in a previous run of this script.
# Using just plain old from helper_functions import read_pickled would require a restart of blender
# every time that function is edited :(
import importlib.util
spec = importlib.util.spec_from_file_location("helper_functions", bpy.path.abspath('//external_scripts/helper_functions.py'))
hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf)

def get_keypoints_and_descriptors(im, detector):
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	for i in range(2):
		blurred1 = cv2.GaussianBlur(gray, (0, 0), 2, 2)
		gray = gray * 2 - blurred1

	(kps, descs) = detector.detectAndCompute(gray, None)

	print("keypoints: {}, descriptors: {}".format(len(kps), descs.shape))
	# cv2.imshow("AKAZE matching", im)
	# cv2.waitKey(20)

	return kps, descs

def match_keypoints(kps_descs):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	num_images = len(kps_descs)


	k = 0
	for i in range(num_images):
		for j in range(num_images):
			if i == j:
				continue

			descs1 = kps_descs[i][1]
			descs2 = kps_descs[j][1]

			matches = bf.knnMatch(descs1, descs2, k=2)

			# Apply ratio test
			good = []
			for m, n in matches:
				if m.distance < 0.6 * n.distance:
					good.append(m)


			print(f"{len(good):<32}  {k} of {num_images**2}")

			k += 1


def kaze_match(im1, im2):
	# load the image and convert it to grayscale
	gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

	# blurred1 = cv2.GaussianBlur(gray1, (0, 0), 2, 2)
	# blurred2 = cv2.GaussianBlur(gray2, (0, 0), 2, 2)

	# for i in range(2):
	# 	gray1 = gray1 * 2 - blurred1
	# 	gray2 = gray2 * 2 - blurred2

	# initialize the AKAZE descriptor, then detect keypoints and extract
	# local invariant descriptors from the image
	detector = cv2.AKAZE_create()
	(kps1, descs1) = detector.detectAndCompute(gray1, None)
	(kps2, descs2) = detector.detectAndCompute(gray2, None)

	print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
	print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

	# Match the features
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.6*n.distance:
			good.append(m)

	for match in good:
		query_point = kps1[match.queryIdx].pt
		train_point = kps2[match.trainIdx].pt

		query_point = (int(round(query_point[0])), int(round(query_point[1])))
		train_point = (int(round(train_point[0])), int(round(train_point[1])))

		cv2.line(im2, query_point, train_point, (255, 0, 0), 2)

	cv2.imshow("AKAZE matching", im2)
	cv2.waitKey(20)

if __name__ == '__main__':
	start_frame = bpy.context.scene.frame_start - 1 # Subtract one from frames because blender starts at frame 1 and we are indexing from 0
	end_frame = bpy.context.scene.frame_end # Don't subtract one because blender is inclusive while python range is exclusive
	assert start_frame >= 0 # Assume blender starts at at least frame 1

	detector = cv2.AKAZE_create()

	cap = hf.start_video_at_frame(config.VIDEO_PATH, start_frame)

	# Get first frame keypoints and descriptors
	ret, image = hf.get_next_frame(cap, config.VIDEO_CROP_TRANSPOSE, scale=0.25)
	get_keypoints_and_descriptors(

	)

	prev_image = None
	kps_descs = []
	for i in range(start_frame, end_frame):
		ret, image = hf.get_next_frame(cap, config.VIDEO_CROP_TRANSPOSE, scale=0.25)
		if not ret:
			break

		if prev_image is not None:
			kaze_match(
				prev_image,
				image,
			)
		else:
			prev_image = image
	# 	result = get_keypoints_and_descriptors(image, detector)
	# 	kps_descs.append(result)

	# # Perform matching
	# match_keypoints(kps_descs)
