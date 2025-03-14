import os
import numpy as np
from fire import Fire
from natsort import natsorted
from datasets.preprocessing.base_preprocessing import BasePreprocessing
import torch
import open3d as o3d
import torch.nn.functional as F
from pathlib import Path
import plyfile

# from splat_viewer.gaussians import read_gaussians


class SplatPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/gaussian",
        save_dir: str = "./data/processed/splat",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        self.class_map = {
            "background": 0,
            "fruit": 1,
        }

        self.color_map = [
            [0, 255, 0],  # background
            [0, 0, 255],  # fruit
        ]
        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_path in [
                Path(f.path) for f in os.scandir(self.data_dir / mode)
            ]:
                filepaths.append(str(scene_path / 'transfer'/ 'clustered.ply'))
            self.files[mode] = natsorted(filepaths)


    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    
    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the main file
            mode: typically train, test or validation

        Returns:
            filebase: info about file
        """
        file_base = {
            "filepath": filepath,
            "scene": filepath.split("/")[-3],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        position, sh_feature, label, instance_label = read_gaussians(filepath)
        rgb = get_rgb(sh_feature[:, :, 0])
        file_len = len(position)
        file_base["file_len"] = file_len

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(position)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
        pcl.estimate_normals(search_param=search_param)
        normals = np.asarray(pcl.normals)

        points = np.hstack((position, rgb, normals))

        if mode in ["train", "validation"]:
            points = np.hstack((points, label, instance_label))
            points = np.hstack((points, np.ones(points.shape[0])[..., None]))
            points[:, [-3, -2, -1]] = points[:, [-1, -3, -2]]

            file_base["raw_segmentation_filepath"] = ""
            gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1

        else:
            points = np.hstack((points, np.ones(points.shape[0])[..., None]))

        processed_filepath = (
            self.save_dir / mode / f"{file_base['scene']}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        file_base["filepath"] = str(processed_filepath)

        if mode == "test":
            return file_base

        processed_gt_filepath = (
            self.save_dir
            / "instance_gt"
            / mode
            / f"{file_base['scene']}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        file_base["instance_gt_filepath"] = str(processed_gt_filepath)

        file_base["color_mean"] = [
            float((rgb[:, 0] / 255).mean()),
            float((rgb[:, 1] / 255).mean()),
            float((rgb[:, 2] / 255).mean()),
        ]
        file_base["color_std"] = [
            float(((rgb[:, 0] / 255) ** 2).mean()),
            float(((rgb[:, 1] / 255) ** 2).mean()),
            float(((rgb[:, 2] / 255) ** 2).mean()),
        ]
        return file_base

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/splat/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)






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



# def get_normals(gaussians: Gaussians) -> torch.Tensor:
#     # from rotatoin matrix get shortest axis direction
#     rotmax = roma.unitquat_to_rotmat(gaussians.rotation)
#     row_indices = torch.arange(gaussians.shape[0])
#     shortest_axis_vector = rotmax[row_indices, :, gaussians.log_scaling.min(dim=1).indices]

#     # adjust the shortest axis vectors to point to the concave side
#     k = 7

#     position_np = np.array(gaussians.position.cpu())
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(position_np)

#     distances, indices_list = nbrs.kneighbors(position_np)

#     for indices in torch.from_numpy(indices_list):
#         base_vector = shortest_axis_vector[0]
#         neighbor_vectors = shortest_axis_vector[indices[1:]]

#         dot_products = torch.sum(base_vector * neighbor_vectors, dim=1)




if __name__ == "__main__":
    Fire(SplatPreprocessing)