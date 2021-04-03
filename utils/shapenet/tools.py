import numpy as np

def pc_from_dep(depth_maps, cam_Ks, cam_Rs, znf, store_camera=False):
    '''
    get point cloud from depth maps
    :param depth_maps: depth map list
    :param cam_Ks: corresponding camera intrinsics
    :param cam_RTs: corresponding camera rotations and translations
    :param znf: [nearest camera distance, furthest distance]
    :param store_camera: if calculate camera position and orientations.
    :return: aligned point clouds in the canonical system with color intensities.
    '''
    point_list_canonical = []
    camera_positions = []

    for depth_map, cam_K, cam_R in zip(depth_maps, cam_Ks, cam_Rs):
        cam_RT = np.hstack([cam_R, np.array([[0, 0, 1]]).T])

        u, v = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]

        z = depth_map[v, u]

        # remove infinitive pixels
        non_inf_indices = np.argwhere(z < znf[1]).T[0]

        z = z[non_inf_indices]
        u = u[non_inf_indices]
        v = v[non_inf_indices]

        # calculate coordinates
        x = (u - cam_K[0][2]) * z / cam_K[0][0]
        y = (v - cam_K[1][2]) * z / cam_K[1][1]

        point_cam = np.vstack([x, y, z]).T

        point_canonical = (point_cam - cam_RT[:, -1]).dot(cam_RT[:, :-1])

        if store_camera:
            cam_pos = - cam_RT[:, -1].dot(cam_RT[:, :-1])
            focal_point = ([0, 0, 1] - cam_RT[:, -1]).dot(cam_RT[:, :-1])
            up = np.array([0, -1, 0]).dot(cam_RT[:, :-1])
            cam_pos = {'pos': cam_pos, 'fp': focal_point, 'up': up}
        else:
            cam_pos = {}

        point_list_canonical.append(point_canonical)
        camera_positions.append(cam_pos)

    output = {'pc': point_list_canonical}

    if store_camera:
        output['cam'] = camera_positions

    return output