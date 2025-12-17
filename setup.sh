cd ..
git clone https://github.com/565353780/sibr-core.git

cd sibr-core
./setup.sh

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124

pip install ffmpeg pillow open3d mediapy lpips \
  scikit-image tqdm trimesh plyfile opencv-python \
  tensorboard ninja

cd ./submodules/diff-surfel-rasterization
python setup.py install

cd ../simple-knn
python setup.py install
