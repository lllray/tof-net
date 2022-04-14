#### references ####
# https://blender.stackexchange.com/questions/52328/render-depth-maps-with-world-space-z-distance-with-respect-the-camera/52348
# https://blender.stackexchange.com/questions/56967/how-to-get-depth-data-using-python-api
# https://blender.stackexchange.com/questions/14910/output-node-with-python
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
####

import bpy
import mathutils
import numpy as np
import math
from math import radians
from os.path import join
from numpy.linalg import inv


#### code from https://www.learnopencv.com/rotation-matrix-to-euler-angles/ ####
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
	# assert(isRotationMatrix(R))
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if not singular:
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	return np.array([x, y, z])
#### end of code from https://www.learnopencv.com/rotation-matrix-to-euler-angles/ ####


#### config for scenes ####
#bathroom
# scale_factor = 0.1
# m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
# 							[0, 0, 1, 0],
# 							[0, -1, 0, 0],
# 							[0, 0, 0, 1]])
#breakfast
# scale_factor = 0.2
# m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
# 							 [0, 1, 0, 0],
# 							 [0, 0, 1, 0],
# 							 [0, 0, 0, 1]])
#contemporary-bathroom
scale_factor = 1
m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
							 [0, 0, 1, 0],
							 [0, -1, 0, 0],
							 [0, 0, 0, 1]])
#pavilion
# scale_factor = 0.5
# m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
# 							[0, 0, 1, 0],
# 							[0, -1, 0, 0],
# 							[0, 0, 0, 1]])
#white-room
# scale_factor = 1
# m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
# 							[0, 0, 1, 0],
# 							[0, -1, 0, 0],
# 							[0, 0, 0, 1]])
#### end of config for scenes ####


#### setup blender properties
scene = bpy.context.scene
camera = bpy.context.active_object
mw = camera.matrix_world
#switch on nodes
scene.use_nodes = True
tree = scene.node_tree
links = tree.links
# clear default nodes
for n in tree.nodes:
	tree.nodes.remove(n)
# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')
# create output node
v = tree.nodes.new('CompositorNodeOutputFile')   
v.base_path = "//render/"
# Links
links.new(rl.outputs['Z'], v.inputs[0]) # link Z to output
#### end of setup blender properties ####


#### in/out path ####
with open('/Users/ssu/GitHub/pbrt-v3-scenes/contemporary-bathroom/camerapath_z.txt') as f:
	lines=f.readlines()
renderFolder = "//render"
#### end of in/out path ####


#### processing each frame ####
for f in range(1,250+1):
	scene.frame_set(f)

	myarray = np.fromstring(lines[f-1], dtype=float, sep=' ').reshape(3,3)
	pos = myarray[0]/scale_factor
	target = myarray[1]/scale_factor
	up = myarray[2]/scale_factor
	up = up/np.linalg.norm(up)
	ndir = pos-target
	ndir = ndir/np.linalg.norm(ndir)
	nleft = np.cross(ndir,up)
	nup = np.cross(nleft,ndir)

	mpbrt = np.zeros((4,4))
	mpbrt[3,:3] = pos
	mpbrt[2,:3] = ndir
	mpbrt[1,:3] = nup
	mpbrt[0,:3] = nleft
	mpbrt[3,3] = 1
	mw = inv(m_blender2pbrt) * mpbrt.transpose()

	## method 1: modify camera.matrix_world directly
	mw_ = mathutils.Matrix()
	mw_[0] = (mw[0,0],mw[0,1],mw[0,2],mw[0,3])
	mw_[1] = (mw[1,0],mw[1,1],mw[1,2],mw[1,3])
	mw_[2] = (mw[2,0],mw[2,1],mw[2,2],mw[2,3])
	mw_[3] = (mw[3,0],mw[3,1],mw[3,2],mw[3,3])
	camera.matrix_world = mw_

	## method 2: modify location and rotation
	# camera.location = (mw[0,3],mw[1,3],mw[2,3])
	# xyz = rotationMatrixToEulerAngles(mw[:3,:3])
	# camera.rotation_euler = (xyz[0], xyz[1], xyz[2])

	frmNum = str(f).zfill(3)
	fileName = "{f}".format(f = frmNum)
	fileName += scene.render.file_extension
	bpy.context.scene.render.filepath = join(renderFolder, fileName)
	bpy.ops.render.render(write_still = True)
