import os
import argparse
import numpy as np
import open3d as o3d

from config import get_config
from urllib.request import urlretrieve
from core.deep_global_registration import DeepGlobalRegistration

BASE_URL = "http://node2.chrischoy.org/data/"
DOWNLOAD_LIST = [ (BASE_URL + 'projects/DGR/', 'ResUNetBN2C-feat32-3dmatch-v0.05.pth') ]

def main():

    # Initialize DGR
    config = get_config()
    if config.weights is None:
        config.weights = DOWNLOAD_LIST[-1][-1]
    input_dir = './pcd_registration/van-gogh'
    dgr = DeepGlobalRegistration(config)

    # Main loop
    pcds = os.listdir(input_dir)
    global_pcd = o3d.geometry.PointCloud()
    for i, p in enumerate(pcds):
        pcd = o3d.io.read_point_cloud(os.path.join(input_dir, p))
        if i == 0:
            global_pcd = pcd
        else:
            global_pcd.estimate_normals()
            pcd.estimate_normals()
            T = dgr.register(global_pcd, pcd)
            pcd = pcd.transform(T)
            global_pcd += pcd

    # global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
    o3d.io.write_point_cloud('out.ply', global_pcd)

if __name__ == '__main__':

    # Download weights if necessary
    if not os.path.isfile('ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
        print('Downloading weights...')
        for f in DOWNLOAD_LIST:
            print(f'Downloading {f}')
            urlretrieve(f[0] + f[1], f[1])

    main()
