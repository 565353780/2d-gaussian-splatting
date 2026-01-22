cd ..
git clone https://github.com/565353780/base-trainer.git
git clone git@github.com:565353780/sibr-core.git
git clone --depth 1 https://github.com/camenduru/simple-knn.git

cd base-trainer
./setup.sh

cd ../sibr-core
./setup.sh

cd ../simple-knn
python setup.py install

pip install ffmpeg pillow mediapy lpips scikit-image \
  plyfile opencv-python ninja

cd ../2d-gaussian-splatting/submodules/diff-surfel-rasterization
python setup.py install
