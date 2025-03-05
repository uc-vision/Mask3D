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
import csv
import roma

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
        position, sh_feature, label, instance_label, normals = read_gaussians(filepath)
        rgb = get_rgb(sh_feature[:, :, 0])
        file_len = len(position)
        file_base["file_len"] = file_len

        # pcl = o3d.geometry.PointCloud()
        # pcl.points = o3d.utility.Vector3dVector(position)
        # search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=20)
        # pcl.estimate_normals(search_param=search_param)
        # normals = np.asarray(pcl.normals)
        


        # points = np.hstack((position, 
        #                     rgb, 
        #                     np.ones(position.shape[0])[..., None],  # normal 1
        #                     np.ones(position.shape[0])[..., None],  # normal 2
        #                     np.ones(position.shape[0])[..., None],))  # normal 3

        points = np.hstack((position,
                            rgb,
                            normals,))

        points[:, :3] = points[:, :3] - points[:, :3].min(0)
        points = points.astype(np.float32)

        if mode in ["test", "validation"]:
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

        if mode in ["validation", "test"]:
            blocks = self.splitPointCloud(points)
            if mode in ["validation", "train"]:
                for index, block in enumerate(blocks):
                    instance_ids = np.unique(block[:, -1])
                    # print("instance_ids: ", instance_ids)
                    if len(instance_ids) == 1 and instance_ids[0] == -1:
                        # print("index: ", index)
                        blocks.pop(index)

            file_base["instance_gt_filepath"] = []
            file_base["filepath_crop"] = []
            # output_csv_path = "./xxxx.csv"
            # with open(output_csv_path, 'a', newline='') as csvfile:
                # csvwriter = csv.writer(csvfile)
            for block_id, block in enumerate(blocks):
                if len(block) > 0:
                    # if mode in ["train", "validation"]:
                    new_instance_ids = np.unique(
                        block[:, -1], return_inverse=True
                    )[1]

                    assert new_instance_ids.shape[0] == block.shape[0]
                    # == 0 means -1 == no instance
                    # new_instance_ids[new_instance_ids == 0]
                    assert (
                        new_instance_ids.max() < 1000
                    ), "we cannot encode when there are more than 999 instances in a block"

                    gt_data = (block[:, -2]) * 1000 + new_instance_ids

                    processed_gt_filepath = (
                        self.save_dir
                        / "instance_gt"
                        / mode
                        / f"{file_base['scene'].replace('.txt', '')}_{block_id}.txt"
                    )

                    # scene_name = file_base['scene'].replace('.txt', '').replace('scan_', '')
                    # block_id = block_id  # Make sure block_id is defined within your loop
                    # num_points = len(block)                      
                    # Write the data to the CSV file
                    # file_name = f"{scene_name}_{block_id}"
                    # csvwriter.writerow([file_name, num_points, points.shape[0]])
                    # print("file and number of points: ", f"{file_base['scene'].replace('.txt', '')}_{block_id}.txt", len(block), np.any(gt_data))

                    if not processed_gt_filepath.parent.exists():
                        processed_gt_filepath.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                    np.savetxt(
                        processed_gt_filepath,
                        gt_data.astype(np.int32),
                        fmt="%d",
                    )
                    file_base["instance_gt_filepath"].append(
                        str(processed_gt_filepath)
                    )

                    processed_filepath = (
                        self.save_dir
                        / mode
                        / f"{file_base['scene'].replace('.txt', '')}_{block_id}.npy"
                    )
                    if not processed_filepath.parent.exists():
                        processed_filepath.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                    np.save(processed_filepath, block.astype(np.float32))
                    file_base["filepath_crop"].append(str(processed_filepath))
                else:
                    # print("block: ", block)
                    # print("block was smaller than 100 points")
                    assert False
                    # continue

            # processed_gt_filepath = (
            #     self.save_dir
            #     / "instance_gt"
            #     / mode
            #     / f"{file_base['scene']}.txt"
            # )
            # if not processed_gt_filepath.parent.exists():
            #     processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
            # np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
            # file_base["instance_gt_filepath"] = str(processed_gt_filepath)

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


    def splitPointCloud(self, cloud, size=0.3, stride=0.25):
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - size) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
        cells = [
            (x * stride, y * stride)
            for x in range(width)
            for y in range(depth)
        ]
        blocks = []
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            if len(block) > 500:
                blocks.append(block)
        return blocks



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
        rotation = rotation[(foreground > 0).squeeze()]
        log_scaling = log_scaling[(foreground > 0).squeeze()]

    normals = get_normals(rotation, log_scaling).cpu().numpy()
    
    return positions, sh_features, label, instance_label, normals

sh0 = 0.282094791773878

def get_rgb(sh):
    return (sh * sh0 + 0.5) * 255



def get_normals(rotation, log_scaling):
    rotmax = roma.unitquat_to_rotmat(rotation)
    row_indices = torch.arange(rotation.shape[0])
    shortest_axis_vector = rotmax[row_indices, :, log_scaling.min(dim=1).indices]
    return shortest_axis_vector


# def get_normals(gaussians) -> torch.Tensor:
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