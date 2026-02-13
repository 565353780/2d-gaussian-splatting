cd ..
git clone https://github.com/565353780/base-gs-trainer.git
#git clone git@github.com:565353780/sibr-core.git

cd base-gs-trainer
./setup.sh

#cd ../sibr-core
#./setup.sh

pip install ffmpeg pillow mediapy lpips scikit-image

cd ../2d-gaussian-splatting/twod_gs/Lib/diff-surfel-rasterization
python setup.py install
