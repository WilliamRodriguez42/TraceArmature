import bpy
import lxml.etree
import numpy as np
import mathutils

if __name__ == '__main__':
	with open(bpy.path.abspath('//external_scripts/tmp/cameras.xml'), 'rb') as f:
		content = f.read()
		xml = lxml.etree.fromstring(content)

	cameras_xml = xml.xpath('//cameras/camera')
	cube = bpy.data.objects['Cube']

	for camera_xml in cameras_xml:
		label = camera_xml.get('label') # Corresponds to the file name (e.g. head_0.png gives label head_0)
		frame = int(label.lstrip('head_')) # Remove head and parse int

		transform_xml = camera_xml.xpath('transform') # Get first transform (there only should be one)

		if len(transform_xml) == 1:
			transform_xml = transform_xml[0]

			transform = np.fromstring(transform_xml.text, dtype=np.float64, sep=' ')
			transform = transform.reshape((4, 4))

			transform = mathutils.Matrix(transform).inverted()
			translation = transform.to_translation()
			translation.y *= -1

			cube.location = translation
			cube.rotation_quaternion = transform.to_quaternion()
			cube.keyframe_insert(data_path='location', frame=frame+1)
			cube.keyframe_insert(data_path='rotation_quaternion', frame=frame+1)
