#!/bin/bash
# Setup of NanoDiP venv with Python 3.7.x
# J. Hench, IfP Basel, 2021-2023
# for use with Ubuntu 20.04 x86_64
# jupyter notebook and minknow_api won't work togehter in python 3.8.8
# Ubuntu 18.04 / jetpack requirements:
# sudo apt install python3.7 python3.7-dev python3.7-venv python3-pip

venvname=nanodipVenv02
venvpath=/applications
mypython=python3.7
startDir=`pwd`
mkdir -p $venvpath/$venvname

cd $venvpath
$mypython -m venv $venvname
source $venvpath/$venvname/bin/activate

# essential components
# 20230424: jupyter notebook 7 is announced which might break some dependencies - pin to versions that are known to work
pip install --upgrade pip
pip install wheel
pip install notebook==6.5.3
pip install jupyter==1.0.0

# convenience components for jupyter notebook
pip install jupyter-contrib-nbextensions==0.7.0
jupyter contrib nbextension install --user
pip install jupyter-nbextensions-configurator==0.6.1
jupyter nbextensions_configurator enable --user

# remaining components in alphabetical order
pip install cupy
pip install cherrypy
pip install grpcio # required for MinKNOW API, replaces old "pip install grpc" which does not work any longer (2021-10-11)
pip install h5py
pip install -U kaleido
pip install matplotlib
pip install numpy
pip install numba
pip install openpyxl
pip install pandas
pip install plotly
pip install psutil
pip install pysam
pip install reportlab==3.6.1 # 20220308 discovered bug in current reportlab version, hence the version pinning
pip install seaborn
pip install scikit-learn
pip install neuralnetwork
pip install tqdm
pip install umap-learn==0.5.3
pip install xhtml2pdf==0.2.5 # 20220308 discovered bug in current reportlab version, hence the version pinning

pip uninstall minknow_api
# install a patched version of the minkow API "minknow_api-2"
cd $venvpath/$venvname
git clone https://github.com/neuropathbasel/minknow_api-3
cd minknow_api-3/python
python setup.py install
python -m pip install .
history

# launch the jupyter notebook
# cd /applications/nanodip
# python -m notebook
