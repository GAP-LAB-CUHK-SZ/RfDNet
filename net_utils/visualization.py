# Visualization functions
# author: ynie
# date: Feb, 2020

import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image

def visualize_voxels(voxels, out_file=None, show=False):
    '''
    Visualizes voxel data.
    :param voxels (tensor): voxel data
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    '''
    Visualizes point cloud data.
    :param points (tensor): point data
    :param normals (tensor): normal data (if existing)
    :param out_file (string): output file
    :param show (bool): whether the plot should be shown
    :return:
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    if normals is not None:
        ax.quiver(
            points[:, 0], points[:, 1], points[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            length=0.1, color='k'
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def visualize_data(data, data_type, out_file):
    '''
    Visualizes the data with regard to its type.
    :param data (tensor): batch of data
    :param data_type (string): data type (img, voxels or pointcloud)
    :param out_file (string): output file
    :return:
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)