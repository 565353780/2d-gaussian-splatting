import os
import torch
import open3d as o3d
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, post_process_mesh

from twod_gs.Dataset.scene import Scene

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(
        dataset.model_path,
        "point_cloud/iteration_" + str(args.iteration),
        "point_cloud.ply"
    ))

    scene = Scene(dataset, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(args.iteration))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(args.iteration))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    print("export mesh ...")
    os.makedirs(train_dir, exist_ok=True)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.cameras)
    # extract the mesh and save
    name = 'fuse.ply'
    depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(train_dir, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
