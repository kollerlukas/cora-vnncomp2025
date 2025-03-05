#!/bin/bash

echo post_install.sh running..

# -------------------------------------------------------------------------
# SETTINGS

USER_NAME=ubuntu
LICENSE_URL='https://drive.google.com/uc?export=download&id=<file-id>'

MATLAB_RELEASE=2024a
EXISTING_MATLAB_LOCATION=$(dirname $(dirname $(readlink -f $(which matlab))))

# define required products (remove already installed products..)
# ADDITIONAL_PRODUCTS="Symbolic_Math_Toolbox Optimization_Toolbox Statistics_and_Machine_Learning_Toolbox Deep_Learning_Toolbox Deep_Learning_Toolbox_Converter_for_ONNX_Model_Format Parallel_Computing_Toolbox"
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
# DONE
echo post_install.sh done
