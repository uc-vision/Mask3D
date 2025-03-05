import torch
import open3d as o3d
import numpy as np
from pathlib import Path
import plyfile
import torch.nn.functional as F


def read_gaussians(filename:Path | str):
    filename = Path(filename) 
    plydata = plyfile.PlyData.read(str(filename))
    vertex = plydata['vertex']

    def get_keys(ks):
        values = [torch.from_numpy(vertex[k].copy()) for k in ks]
        return torch.stack(values, dim=-1)

    
    positions = torch.stack(
        [ torch.from_numpy(vertex[i].copy()) for i in ['x', 'y', 'z']], dim=-1)

    attrs = sorted(plydata['vertex'].data.dtype.names)
    sh_attrs = [k for k in attrs if k.startswith('f_rest_') or k.startswith('f_dc_')]
    
    n_sh = len(sh_attrs) // 3
    deg = int(np.sqrt(n_sh))

    assert deg * deg == n_sh, f"SH feature count must be square ({deg} * {deg} != {n_sh}), got {len(sh_attrs)}"
    log_scaling = get_keys([f'scale_{k}' for k in range(3)])


    sh_dc = get_keys([f'f_dc_{k}' for k in range(3)]).view(positions.shape[0], 3, 1)
    sh_rest = get_keys([f'f_rest_{k}' for k in range(3 * (deg * deg - 1))])
    sh_rest = sh_rest.view(positions.shape[0], 3, n_sh - 1)

    sh_features = torch.cat([sh_dc, sh_rest], dim=2)  

    rotation = get_keys([f'rot_{k}' for k in range(4)])
    # convert from wxyz to xyzw quaternion and normalize
    rotation = torch.roll(F.normalize(rotation), -1, dims=(1,))
    
    alpha_logit = get_keys(['opacity'])

    foreground = (get_keys(['foreground']).to(torch.bool) 
        if 'foreground' in plydata['vertex'].data.dtype.names else None)
    
    label = get_keys(['label']) if 'label' in vertex.data.dtype.names else None
    instance_label = get_keys(['instance_label']) if 'instance_label' in vertex.data.dtype.names else None

    if foreground is not None:
        positions = positions[(foreground > 0).squeeze()].cpu().numpy()
        sh_features = sh_features[(foreground > 0).squeeze()].cpu().numpy()
        label = label[(foreground > 0).squeeze()].cpu().numpy()
        instance_label = instance_label[(foreground > 0).squeeze()].cpu().numpy()
    
    return positions, sh_features, label, instance_label

sh0 = 0.282094791773878

def get_rgb(sh):
    return (sh * sh0 + 0.5) * 255

def process_file(filepath):
    position, sh_feature, label, instance_label = read_gaussians(filepath)
    rgb = get_rgb(sh_feature[:, :, 0])
    file_len = len(position)

    pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(position)
    # search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
    # pcl.estimate_normals(search_param=search_param)
    # normals = np.asarray(pcl.normals)
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=9)
    
    # # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # # mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # # mesh.paint_uniform_color([0.6, 0.6, 0.6])
    
    # # o3d.visualization.draw_geometries([pcl], point_show_normal=True)
    # # o3d.visualization.draw_geometries([mesh, pcl])

    # pcl_tree = o3d.geometry.KDTreeFlann(pcl)
    # mesh_vertex_colors = []
    # for vertex in mesh.vertices:
    #     _, idx, _ = pcl_tree.search_knn_vector_3d(vertex, 1)
    #     mesh_vertex_colors.append(pcl.colors[idx[0]])
    # mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    # # Visualize the mesh and the original point cloud together
    # o3d.visualization.draw_geometries([mesh, pcl])
    
    # Assign the points and colors to the PointCloud object
    pcl.points = o3d.utility.Vector3dVector(position)
    pcl.colors = o3d.utility.Vector3dVector(rgb)

    # Estimate normals
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
    pcl.estimate_normals(search_param=search_param)

    # Ball Pivoting Algorithm for surface reconstruction
    distances = pcl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcl,
        o3d.utility.DoubleVector([radius, radius * 2, radius * 3])
    )

    # Transfer colors from point cloud to the mesh vertices
    pcl_tree = o3d.geometry.KDTreeFlann(pcl)
    mesh_vertex_colors = []
    for vertex in mesh.vertices:
        _, idx, _ = pcl_tree.search_knn_vector_3d(vertex, 1)
        mesh_vertex_colors.append(pcl.colors[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    # Visualize the mesh and the original point cloud together
    o3d.visualization.draw_geometries([mesh, pcl])
    


if __name__ == "__main__":
    process_file("/local/scan_09/point_cloud/iteration_30000/point_cloud.ply")