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
    pcds, pcd_inds = os.listdir(input_dir), []
    for pcd in pcds:
        pcd_inds.append(int(pcd.split('_')[1].split('.')[0]))
    pcd_inds = np.argsort(pcd_inds)

    for i, _ in enumerate(pcds):
        pcd = o3d.io.read_point_cloud(os.path.join(input_dir, pcds[pcd_inds[i]]))
        if i == 0:
            pcd_0 = o3d.geometry.PointCloud()
            pcd_0.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 13000)
            pcd_0.estimate_normals()
        else:
            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 13000)
            pcd_1.estimate_normals()
            T = dgr.register(pcd_0, pcd_1)

            print('-----------------------')
            print('TRANSFORMATION:')
            print(T)
            print('-----------------------')

            pcd_0 = pcd_1

            np.save(pcds[pcd_inds[i]].split('.')[0], T)

if __name__ == '__main__':

    # Download weights if necessary
    if not os.path.isfile('ResUNetBN2C-feat32-3dmatch-v0.05.pth'):
        print('Downloading weights...')
        for f in DOWNLOAD_LIST:
            print(f'Downloading {f}')
            urlretrieve(f[0] + f[1], f[1])

    main()
