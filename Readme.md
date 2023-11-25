
Proste programy do sprawdzenia właściwości karty GPU na Linux 

1. INSTALL NVIDIA CUDA Toolkit on Linux
	https://developer.nvidia.com/cuda-downloads?target_os=Linux 
# Debian12/Proxmox 8:
apt-get update
apt-get dist-upgrade
apt install build-essential gcc dirmngr ca-certificates software-properties-common apt-transport-https dkms curl pve-headers
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
add-apt-repository contrib
apt-get install cuda-toolkit 
apt show cuda-drivers -a
apt policy cuda
apt-get clean
apt-get update
apt-get install -y python3 python3-venv python3-pip
apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev
apt-get install cuda-drivers
apt-get install cuda
apt-get update
apt-get dist-upgrade
apt-get clean
set |grep -e cuda
set |grep -e PATH -e LD_LIBRARY_PATH
cat cat /etc/profile
# NV-CUDA
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# logout/login
set |grep -e cuda
nvidia-smi

