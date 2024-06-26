pip install -U torch torchvision torchaudio

pip install -U ffmpeg pillow open3d mediapy lpips \
	scikit-image tqdm trimesh plyfile opencv-python

cd ./submodules/diff-surfel-rasterization
pip install -e .

cd ../simple-knn
pip install -e .
