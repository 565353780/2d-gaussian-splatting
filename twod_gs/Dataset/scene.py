import os
import json
import random

from arguments import ModelParams
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    def __init__(self, args : ModelParams, shuffle=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path

        self.cameras = []

        scene_info = readColmapSceneInfo(args.source_path, args.images)

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.cameras:
            camlist.extend(scene_info.cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.cameras = cameraList_from_camInfos(scene_info.cameras, 1.0, args)

        self.scene_info = scene_info
        return

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, index: int) -> dict:
        index = index % len(self.cameras)
        return self.cameras[index]
