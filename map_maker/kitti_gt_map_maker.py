#!/usr/bin/env python3
import os, sys, signal

from tqdm import tqdm
import rospy
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from geometry_msgs.msg import Pose, PoseArray # PoseArray, Pose
from open3d_ros_helper import open3d_ros_helper as orh
import copy

import csv
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description='A simple kitti publisher')
parser.add_argument('--dir', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', metavar='DIR', help='path to dataset')
parser.add_argument('--hz', type=int, default=10, help='Hz of dataset')
parser.add_argument('--scan_dir', type=str, default='00', help='seq_num')
args = parser.parse_args()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1 + np.trace(R)) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return qw, qx, qy, qz

def transform_point_cloud(point_cloud, pose_matrix):
    # 확장된 형태로 변환
    points_homogeneous = np.hstack((point_cloud, np.ones((len(point_cloud), 1))))

    # 포즈 행렬과의 행렬 곱
    transformed_points = np.dot(pose_matrix, points_homogeneous.T).T

    # 마지막 열 제거하여 3차원 공간으로 되돌림
    transformed_points = transformed_points[:, :3]

    return transformed_points

class Gt_dataset():
    def __init__(self, args) -> None:
        self.pub_pc_map = rospy.Publisher('/gt_map', PointCloud2, queue_size=100)
        self.pub_poses = rospy.Publisher('/gt_odom', PoseArray, queue_size=100)

        self.dir_path = args.dir
        self.scan_dir = args.scan_dir
        self.scan_names = os.listdir(self.scan_dir)
        self.scan_names.sort()
        self.num_frames = self.scan_names.__len__()
        self.poses = []
        self.times = []
        self.scans = []
        self.key_idx = []

        self.icp_keypose = []

        self._get_poses()
        self._get_times()
        self._get_scans()
        self._get_key_idx()

    def transfrom_cam2velo(self, Tcam):
        R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                    -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                    ]).reshape(3, 3)
        t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
        cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
        return Tcam @ cam2velo
    
    def _get_poses(self):
        with open(os.path.join(os.path.join(self.dir_path + seqence_num), 'poses.txt')) as csvfile:
            poses_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            T1 = np.eye(4,4)
            T1[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            T2 = np.eye(4,4)
            T2[:3,:3] = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()

            for line in poses_reader:
                pose_raw = np.array(line).reshape(3, 4).astype(np.float64)
                pose_raw = np.concatenate((pose_raw, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
                pose_to_velo = self.transfrom_cam2velo(pose_raw)
                pose = np.matmul(T2, np.matmul(T1, pose_to_velo))
                self.poses.append(pose)
        print("_get_poses End")

    def _get_times(self):
        with open(os.path.join(os.path.join(self.dir_path + seqence_num), 'times.txt')) as csvfile:
            times_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for line in times_reader:
                self.times.append(float(line[0]))
        print("_get_times End")

    def _get_scans(self):
        for frame_idx in range(self.num_frames):
            scan_path = os.path.join(self.scan_dir, self.scan_names[frame_idx])
            xyzi = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
            self.scans.append(xyzi[np.linalg.norm(xyzi[:,:3], axis=1) > 3.0])
        print("_get_scans End")

    def _get_key_idx(self):
        anc_pose = self.poses[0]
        translation_accumulated = 0.0
        rotation_accumulated = 0.0

        for i, pose in enumerate(self.poses):
            if i == 0:
                self.key_idx.append(i)
                continue
            relative_pose = np.linalg.inv(anc_pose) @ pose
            translation = relative_pose[:3, 3]
            translation_accumulated += np.sum(translation)

            rotation_matrix = relative_pose[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            rotation_accumulated += sum(euler_angles)

            if ((translation_accumulated > 3.0) or (rotation_accumulated > 10.0)):
                translation_accumulated = 0.0
                rotation_accumulated = 0.0
                self.key_idx.append(i)
                anc_pose = pose
        print("_get_key_idx End")
    
    def make_map(self):
        self.pcMsg = PointCloud2()
        pcMsg_pc = o3d.geometry.PointCloud()
        self.odom_msg = PoseArray()

        self.icp_keypose.append(self.poses[0])
        prev_pose = self.icp_keypose[0]

        for i, idx in tqdm(enumerate(self.key_idx[1:])):
            if i < 100:
                icp_source = o3d.geometry.PointCloud()
                icp_target = o3d.geometry.PointCloud()
                
                icp_source.points = o3d.utility.Vector3dVector(transform_point_cloud(self.scans[idx][:,:3], self.poses[idx]))
                # icp_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
                icp_source.voxel_down_sample(voxel_size=0.1)
                icp_target.points = o3d.utility.Vector3dVector(transform_point_cloud(self.scans[self.key_idx[i]][:,:3], prev_pose))
                # icp_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
                icp_target.voxel_down_sample(voxel_size=0.1)

                voxel_size = 0.1
                threshold = 8.0 * voxel_size
                reg_p2p = o3d.pipelines.registration.registration_icp(icp_source, icp_target, threshold, np.eye(4,4), o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=400))
                
                # draw_registration_result(icp_source, icp_target, reg_p2p.transformation)
                self.icp_keypose.append(np.matmul(reg_p2p.transformation, self.poses[idx]))
                prev_pose = self.icp_keypose[-1]
            
        for i, idx in tqdm(enumerate(self.icp_keypose)):
            if idx < 100:
                PCChach = o3d.geometry.PointCloud()
                PCChach.points= o3d.utility.Vector3dVector(transform_point_cloud(self.scans[idx][:,:3], self.icp_keypose[i]))
                pcMsg_pc += PCChach.voxel_down_sample(voxel_size=0.2)
            if idx < 100:
                odom_msg_chach = Pose()
                odom_msg_chach.position.x = self.icp_keypose[idx][0,3]
                odom_msg_chach.position.y = self.icp_keypose[idx][1,3]
                odom_msg_chach.position.z = self.icp_keypose[idx][2,3]
                qw, qx, qy, qz = rotation_matrix_to_quaternion(self.icp_keypose[i][:3, :3])
                odom_msg_chach.orientation.w = qw
                odom_msg_chach.orientation.x = qx
                odom_msg_chach.orientation.y = qy
                odom_msg_chach.orientation.z = qz
                self.odom_msg.poses.append(odom_msg_chach)
        # ------------------------------------------
        self.odom_msg.header.frame_id = "/camera_init"
        self.pcMsg = orh.o3dpc_to_rospc(pcMsg_pc.voxel_down_sample(voxel_size=0.3))
        self.pcMsg.header.frame_id = "/camera_init"
        self.pub_pc_map.publish(self.pcMsg)
        self.pub_poses.publish(self.odom_msg)
        
    def pub_map(self):
        self.pub_pc_map.publish(self.pcMsg)
        self.pub_poses.publish(self.odom_msg)
        # print("pub complite")

def handle_sigint(signal, frame):
    print("\n ---cancel by user---")
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('kitti_gt_map_maker')
    signal.signal(signal.SIGINT, handle_sigint)

    # seqence_num = input("seq? >> ")
    seqence_num = "00"
    args.scan_dir = os.path.join(os.path.join(args.dir + seqence_num), 'velodyne')

    Data = Gt_dataset(args=args)
    Data.make_map()
    while True:
        input("map massage publist again?")
        Data.pub_map()