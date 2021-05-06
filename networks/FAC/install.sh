#!/bin/bash
cd ./kernelconv2d
python setup.py clean
# python setup.py install --single-version-externally-managed --user \
# python setup.py install --prefix=/usr/local/ \
# python setup.py install --prefix=/mnt/lustre/wangzhouxia/project/fast-rcnn/venv/
# python setup.py install --prefix=/mnt/lustre/wangzhouxia/miniconda3/envs/pt1.3v1/
python setup.py install --prefix=/data1/env/miniconda3/envs/py37/
