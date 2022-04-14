# load this file to .blend -> Text Editor -> Text (Open text block) -> Text (Run script)
# acording to the camera path animated, this will output a sequence of LookAt arrays, 
# which can then be read by pbrt_tof_batch.sh to alternate the camera view.

import bpy
import numpy as np


def write_some_data(context, filepath, use_some_setting):
    print("running write_some_data...")
    camera = context.active_object
    mw = camera.matrix_world
    scene = context.scene
    frame = scene.frame_start

    f = open(filepath, 'w', encoding='utf-8')
    while frame <= scene.frame_end:
        scene.frame_set(frame)
#        x, y, z = mw.to_translation()
#        rx, ry, rz = mw.to_euler('XYZ')
#        sx, sy, sz = mw.to_scale()
#        rotx = Matrix(((1, 0, 0, 0), 
#                       (0, cos(rx), -sin(rx), 0), 
#                       (0, sin(rx), cos(rx), 0), 
#                       (0, 0, 0, 1)))
#        roty = Matrix(((cos(ry), 0, sin(ry), 0), 
#                       (0, 1, 0, 0), 
#                       (-sin(ry), 0, cos(ry), 0), 
#                       (0, 0, 0, 1)))
#        rotz = Matrix(((cos(rz), -sin(rz), 0, 0), 
#                       (sin(rz), cos(rz), 0, 0), 
#                       (0, 0, 1, 0), 
#                       (0, 0, 0, 1)))
#        t    = Matrix(((0, 0, 0, x), 
#                       (0, 0, 0, y), 
#                       (0, 0, 0, z), 
#                       (0, 0, 0, 1)))
#        s    = Matrix(((sx, 0, 0, 0), 
#                       (0, sy, 0, 0), 
#                       (0, 0, sz, 0), 
#                       (0, 0, 0, 1)))
        #for bathroom, white-room, pavilion
        # m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
        #                             [0, 0, 1, 0],
        #                             [0, -1, 0, 0],
        #                             [0, 0, 0, 1]])
        #for breakfast
        m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        #for cont bathroom
        # m_blender2pbrt = np.matrix([[-1, 0, 0, 0],
        #                             [0, 0, 1, 0],
        #                             [0, -1, 0, 0],
        #                             [0, 0, 0, 1]])
        mpbrt = np.matrix(m_blender2pbrt * np.matrix(mw)).transpose()
        pos = mpbrt[3]
        forwards = -mpbrt[2]
        target = pos + forwards
        up = mpbrt[1]
        pos = np.squeeze(np.asarray(pos))
        target = np.squeeze(np.asarray(target))
        up = np.squeeze(np.asarray(up))
        # additional scale
        scale_factor = 0.2 #0.1 for bathroom, 1 for white-room, 0.5 for barcelona-pavilion, 0.2 for breakfast, 1 for cont bathroom
        pos = pos*scale_factor
        target = target*scale_factor
        up = up*scale_factor
#        f.write("%d\n" % frame)
        f.write("LookAt ")
        f.write("%5.3f %5.3f %5.3f" % (pos[0], pos[1], pos[2]))
        f.write(" ")
        f.write("%5.3f %5.3f %5.3f" % (target[0], target[1], target[2]))
        f.write(" ")
        f.write("%5.3f %5.3f %5.3f" % (up[0], up[1], up[2]))
        f.write("\n")
        frame += 1
    f.close()

    return {'FINISHED'}


# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator


class ExportSomeData(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_test.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Some Data"

    # ExportHelper mixin class uses this
    filename_ext = ".txt"

    filter_glob = StringProperty(
            default="*.txt",
            options={'HIDDEN'},
            )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    use_setting = BoolProperty(
            name="Example Boolean",
            description="Example Tooltip",
            default=True,
            )

    type = EnumProperty(
            name="Example Enum",
            description="Choose between two items",
            items=(('OPT_A', "First Option", "Description one"),
                   ('OPT_B', "Second Option", "Description two")),
            default='OPT_A',
            )

    def execute(self, context):
        return write_some_data(context, self.filepath, self.use_setting)


# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(ExportSomeData.bl_idname, text="Text Export Operator")


def register():
    bpy.utils.register_class(ExportSomeData)
    bpy.types.INFO_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportSomeData)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.export_test.some_data('INVOKE_DEFAULT')
