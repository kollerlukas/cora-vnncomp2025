#!/bin/bash

echo post_install.sh running..

# -------------------------------------------------------------------------
# SETTINGS

USER_NAME=ubuntu
LICENSE_URL='https://drive.google.com/uc?export=download&id=<file-id>'

MATLAB_RELEASE=2024a
EXISTING_MATLAB_LOCATION=$(dirname $(dirname $(readlink -f $(which matlab))))

# define required products (remove already installed products..)
ADDITIONAL_PRODUCTS="Deep_Learning_Toolbox_Converter_for_ONNX_Model_Format"

CURR_DIR=$(pwd)

# -------------------------------------------------------------------------
# ECHO
echo ${USER_NAME}
echo ${LICENSE_URL}
echo ${EXISTING_MATLAB_LOCATION}
echo ${CURR_DIR}
ls -al
# -------------------------------------------------------------------------
# INITIAL GENERAL INSTALLATION
# check if everything is up to date
export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
    wget \
    unzip \
    ca-certificates \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*
	
# -------------------------------------------------------------------------
# MATLAB PACKAGE INSTALLATION
wget -q https://www.mathworks.com/mpm/glnxa64/mpm \
    && chmod +x mpm \
    && ./mpm install \
        --destination=${EXISTING_MATLAB_LOCATION} \
        --release=r${MATLAB_RELEASE} \
        --products ${ADDITIONAL_PRODUCTS}	
	
# -------------------------------------------------------------------------
# CORA INSTALLATION
# download license file
curl --retry 100 --retry-connrefused  -L ${LICENSE_URL} -o license.lic
# copy to license folder and delete other license info
cp -f license.lic "${EXISTING_MATLAB_LOCATION}/licenses"
# run installCORA non-interactively
matlab -nodisplay -r "cd ${CURR_DIR}; addpath(genpath('.')); installCORA(false,true,'${CURR_DIR}/code'); savepath"

# -------------------------------------------------------------------------
# APPEND SSH KEY
# echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICNsvgOYRzzuh3BH6Hslv3g8Ro4bG5dZoQbN4QixS1fd lukas.koller@tum.de" >> ~/.ssh/authorized_keys

# -------------------------------------------------------------------------
# FIX GPU DRIVER ISSUES
#!/bin/bash
set -e  # Exit on error

# -------------------------
# 1. Remove any existing NVIDIA drivers and CUDA
# -------------------------
sudo apt-get purge -y '*nvidia*'
sudo apt-get autoremove -y
sudo rm -rf /usr/local/cuda*
sudo rm -rf /var/lib/dkms/nvidia

# -------------------------
# 2. Install dependencies for building kernel modules
# -------------------------
sudo apt-get update
sudo apt-get install -y build-essential dkms linux-headers-$(uname -r) gcc make

# -------------------------
# 3. Download the correct NVIDIA driver (535.154.05)
#    Compatible with A10G + MATLAB R2024a (CUDA 12.2)
# -------------------------
cd /tmp
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.154.05/NVIDIA-Linux-x86_64-535.154.05.run
chmod +x NVIDIA-Linux-x86_64-535.154.05.run

# -------------------------
# 4. Run the installer in silent mode (no GUI)
#    --dkms builds the kernel module
# -------------------------
sudo ./NVIDIA-Linux-x86_64-535.154.05.run --silent --dkms

# -------------------------
# 5. All done — reboot ONCE manually after this script completes
# -------------------------

# -------------------------------------------------------------------------
# DONE
echo post_install.sh done
