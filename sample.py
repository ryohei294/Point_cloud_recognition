import numpy as np
import open3d as o3d
import os
import argparse

#球状に分布した点群の生成
def generate_sphere(save_dir, num_data, num_pc):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_data):
        phi = np.random.uniform(0, np.pi*2, num_pc)
        costheta = np.random.uniform(-1, 1, num_pc)
        u = np.random.uniform(0, 1, num_pc)

        theta = np.arccos(costheta)
        r = (u**(1/3))*np.random.uniform(0.8, 1.2)

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        points = np.stack([x,y,z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        filename = f"random_sphere_{i+1}.ply"
        save_path = os.path.join(save_dir, filename)
        o3d.io.write_point_cloud(save_path, pcd)

"""
#xy平面上に分布した点群の生成
def generate_plane(save_dir, num_pc):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_pc):
        x = np.random.uniform(-1, 1, 1024)
        y = np.random.uniform(-1, 1, 1024)
        z = np.random.normal(-0.02, 0.02, 1024)
        points = np.stack([x,y,z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        filename = f"random_plane_{i+1}.ply"
        save_path = os.path.join(save_dir, filename)
        o3d.io.write_point_cloud(save_path, pcd)
        """
#楕円形に分布した点群の生成
def generate_ellipsoid(save_dir, num_data, num_pc, scale=(1.0, 0.7, 0.4)):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_data):
        phi = np.random.uniform(0, 2 * np.pi, num_pc)
        costheta = np.random.uniform(-1, 1, num_pc)
        u = np.random.uniform(0, 1, num_pc)

        theta = np.arccos(costheta)
        r = u ** (1/3)

        x = r * np.sin(theta) * np.cos(phi) * scale[0]
        y = r * np.sin(theta) * np.sin(phi) * scale[1]
        z = r * np.cos(theta) * scale[2]
        points = np.stack([x, y, z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        filename = f"random_ellipsoid_{i+1}.ply"
        save_path = os.path.join(save_dir, filename)
        o3d.io.write_point_cloud(save_path, pcd)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_data", type=int, default=100
)
parser.add_argument(
    "--num_pc", type=int, default=256
)
args = parser.parse_args()

num_data = args.num_data
num_pc = args.num_pc
data1 = "data1/"
data2 = "data2/"

generate_sphere(data1, num_data, num_pc)
#generate_plane(data2, num_pc)
generate_ellipsoid(data2, num_data, num_pc)
