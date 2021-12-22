from distutils.core import setup, Extension

extension = Extension(
	'aabb_filter',
	['aabb_filter.cpp'],
	define_macros=[('NUM_BITS', 8)],
)

extension.export_symbols = ['aabb_filter'] # This is required or else aabb_filter function is not visible to ctypes

setup(
	ext_modules=[extension],
)
