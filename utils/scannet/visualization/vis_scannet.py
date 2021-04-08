'''
Visualization tools for Scannet.
author: ynie
date: July, 2020
'''
import sys

sys.path.append('.')
import os
from configs.path_config import PathConfig
from utils.scannet.load_scannet_data import export
import seaborn as sns
import pickle
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
import math
from configs.path_config import SHAPENETCLASSES

path_config = PathConfig('scannet')


class Vis_Scannet(object):
    '''
    visualization class for scannet frames.
    '''

    def __init__(self, gt_dirname: str = None):
        scene_name = os.path.basename(gt_dirname)
        self._mesh_file = os.path.join(gt_dirname, scene_name + '_vh_clean_2.ply')
        agg_file = os.path.join(gt_dirname, scene_name + '.aggregation.json')
        seg_file = os.path.join(gt_dirname, scene_name + '_vh_clean_2.0.010000.segs.json')
        meta_file = os.path.join(gt_dirname, scene_name + '.txt')  # includes axis

        with open(path_config.raw_label_map_file, 'rb') as file:
            label_map = pickle.load(file)

        self._vertices, self.semantic_labels, self.instance_labels, self.instance_bboxes, self.instance2semantic = \
            export(self._mesh_file, agg_file, seg_file, meta_file, label_map, None)

        self.palette_cls = np.array([(0., 0., 0.), *sns.color_palette("hls", len(SHAPENETCLASSES))])
        self.palette_inst = np.array([(0., 0., 0.), *sns.color_palette("hls", max(self.instance_labels) + 1)])

    @property
    def mesh_file(self):
        return self._mesh_file

    @property
    def vertices(self):
        return self._vertices

    def set_mapper(self, prop, mode):

        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

        else:
            raise IOError('No Mapper mode found.')

        return mapper

    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()

        '''replace aligned points'''
        polydata = plydata.GetOutput()
        points_array = numpy_to_vtk(self.vertices[..., :3], deep=True)
        # Update the point information of vtk
        polydata.GetPoints().SetData(points_array)
        # update changes
        plydata.Update()

        return plydata

    def set_points_property(self, point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point

    def set_axes_actor(self):
        '''
        Set camera coordinate system
        '''
        transform = vtk.vtkTransform()
        transform.Translate(0., 0., 0.)
        # self defined
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.SetTotalLength(1, 1, 1)

        axes.SetTipTypeToCone()
        axes.SetConeRadius(50e-2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(40e-3)

        vtk_textproperty = vtk.vtkTextProperty()
        vtk_textproperty.SetFontSize(1)
        vtk_textproperty.SetBold(True)
        vtk_textproperty.SetItalic(False)
        vtk_textproperty.SetShadow(True)

        for label in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            label.SetCaptionTextProperty(vtk_textproperty)

        return axes

    def get_box_corners(self, center, vectors):
        '''
        Convert box center and vectors to the corner-form
        :param center:
        :param vectors:
        :return: corner points and faces related to the box
        '''
        corner_pnts = [None] * 8
        corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
        corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
        corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
        corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

        corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
        corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
        corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
        corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)]

        return corner_pnts, faces

    def mkVtkIdList(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def set_cube_prop(self, corners, faces, color):

        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        color = np.uint8(color)

        for i in range(8):
            points.InsertPoint(i, corners[i])

        for i in range(6):
            polys.InsertNextCell(self.mkVtkIdList(faces[i]))

        for i in range(8):
            colors.InsertNextTuple3(*color)

        # Assign the pieces to the vtkPolyData
        cube.SetPoints(points)
        del points
        cube.SetPolys(polys)
        del polys
        cube.GetPointData().SetScalars(colors)
        cube.GetPointData().SetActiveScalars('Color')
        del colors

        return cube

    def make_bands(self, dR, numberOfBands, nearestInteger):
        '''
        Divide a range into bands
        :param: dR - [min, max] the range that is to be covered by the bands.
        :param: numberOfBands - the number of bands, a positive integer.
        :param: nearestInteger - if True then [floor(min), ceil(max)] is used.
        :return: A List consisting of [min, midpoint, max] for each band.
        '''
        bands = list()
        if (dR[1] < dR[0]) or (numberOfBands <= 0):
            return bands
        x = list(dR)
        if nearestInteger:
            x[0] = math.floor(x[0])
            x[1] = math.ceil(x[1])
        dx = (x[1] - x[0]) / float(numberOfBands)
        b = [x[0], x[0] + dx / 2.0, x[0] + dx]
        i = 0
        while i < numberOfBands:
            bands.append(b)
            b = [b[0] + dx, b[1] + dx, b[2] + dx]
            i += 1
        return bands

    def setLUT(self):

        class_Id_list = np.unique(self.instance_bboxes[:, -1].astype(np.int32))
        class_Id_list = np.array([item for item in class_Id_list if item in path_config.OBJ_CLASS_IDS])
        class_names = [SHAPENETCLASSES[i] for i in class_Id_list]
        color_list = self.palette_cls[class_Id_list]

        num_class = len(class_names)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(num_class)
        lut.SetTableRange(0, num_class)

        number_of_bands = lut.GetNumberOfTableValues()

        # we will use the midpoint of the band as label
        bands = self.make_bands([0, num_class], number_of_bands, False)
        labels = []

        for i in range(len(bands)):
            labels.append('{:4.2f}'.format(bands[i][1]))

        # annotate
        for i in range(num_class):
            lut.SetAnnotation(labels[i], str(class_names[i]))
            lut.SetTableValue(i, color_list[i][0], color_list[i][1], color_list[i][2], 1)
        lut.Build()

        return lut

    def set_scalar_bar_actor(self):

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetOrientationToVertical()
        scalar_bar.SetTitle('Category')
        scalar_bar.SetLookupTable(self.setLUT())
        scalar_bar.SetNumberOfLabels(0)
        # scalar_bar.GetLabelTextProperty().SetFontSize(80)
        scalar_bar.GetTitleTextProperty().SetFontSize(40)
        scalar_bar.GetAnnotationTextProperty().SetFontSize(20)
        scalar_bar.SetMaximumWidthInPixels(150)
        scalar_bar.SetMaximumHeightInPixels(800)
        scalar_bar.GetPositionCoordinate().SetValue(0.8, 0.15)

        return scalar_bar

    def set_camera(self, position, focal_point, cam_K):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point[0])
        camera.SetViewUp(*focal_point[1])
        camera.SetViewAngle((2*np.arctan(cam_K[1][2]/cam_K[0][0]))/np.pi*180)
        return camera

    def set_render(self, detection=True):
        renderer = vtk.vtkRenderer()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        if detection:
            '''load ply mesh file'''
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property(self.mesh_file), 'model'))
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

            '''load lookup table'''
            scalar_bar_actor = self.set_scalar_bar_actor()
            renderer.AddActor(scalar_bar_actor)

            '''load bounding boxes'''
            for instance_bbox in self.instance_bboxes:
                cls_id = int(instance_bbox[-1])
                if cls_id not in path_config.OBJ_CLASS_IDS:
                    continue
                center = instance_bbox[:3]
                vectors = np.diag(instance_bbox[3:6]) / 2.
                corners, faces = self.get_box_corners(center, vectors)
                color = self.palette_cls[cls_id] * 255
                bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
                bbox_actor.GetProperty().SetOpacity(0.5)
                bbox_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(bbox_actor)
        else:
            '''load point actor'''
            instance_labels = [inst_label if self.semantic_labels[inst_id] in path_config.OBJ_CLASS_IDS else -1 for
                               inst_id, inst_label in enumerate(self.instance_labels)]
            instance_labels = np.array(instance_labels)
            colors = self.palette_inst[instance_labels + 1] * 255
            point_actor = self.set_actor(self.set_mapper(self.set_points_property(self.vertices[:, :3], colors), 'box'))
            point_actor.GetProperty().SetPointSize(3)
            point_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(point_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, detection=True):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(detection)
        renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(1366, 768)

        return render_window

    def visualize(self, detection=True):
        '''
        Visualize a 3D scene.
        '''

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window = self.set_render_window(detection)
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Start()


if __name__ == '__main__':
    scene_dirname = 'scene0001_00'
    gt_dirname = os.path.join(path_config.metadata_root, 'scans', scene_dirname)
    assert os.path.exists(gt_dirname)

    scene = Vis_Scannet(gt_dirname=gt_dirname)
    scene.visualize(detection=True)
