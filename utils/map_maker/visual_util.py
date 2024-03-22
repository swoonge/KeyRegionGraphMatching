#!/usr/bin/env python3
import os, sys, signal,rospy, argparse, csv

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import copy, pickle
import open3d as o3d

from open3d_ros_helper import open3d_ros_helper as orh
from geometry_msgs.msg import Pose, PoseArray, Point # PoseArray, Pose
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

from shot_fpfh import compute_normals

parser = argparse.ArgumentParser(description='A simple kitti publisher')
# parser.add_argument('--gt_dir', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', metavar='DIR', help='path to dataset')
parser.add_argument('--slam_dir', type=str, default='/media/vision/Seagate/DataSets/KRGM/kitti/', metavar='DIR', help='path to SLAM dataset')
parser.add_argument('--local_global', type=bool, default=False, help='')
parser.add_argument('--seq_num', type=str, default='00', help='seq_num')
args = parser.parse_args()

class dataset():
    def __init__(self, args) -> None:
        self.gt_seq = args.seq_num

        self.dir_SLAM_path = args.slam_dir + self.gt_seq + "/"
        self.pub_SLAM_map = rospy.Publisher('/slam_map', PointCloud2, queue_size=100)
        self.pub_SLAM_keypoints = rospy.Publisher('/slam_keypoints', PointCloud2, queue_size=100)
        self.pub_SLAM_keypoints_local = rospy.Publisher('/slam_keypoints_local', PointCloud2, queue_size=100)
        self.pub_SLAM_poses = rospy.Publisher('/slam_odom', PoseArray, queue_size=100)
        # self.pub_matching_line = rospy.Publisher('/matching_line', Marker, queue_size=100)

        self.poses = []
        self.dense_scans = []
        self.keypoints = []
        self.descriptors = []
        self.local_graph_range = [0, 0]

        self._get_SLAM_poses()
        self._get_dense_frames()
        self._get_keypoints()
        self._get_descriptors()
        print("[Load] SLAM data complite")

    def _get_SLAM_poses(self):
        with open(file=os.path.join(self.dir_SLAM_path, "Poses_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.poses = pickle.load(f)

    def _get_dense_frames(self):
        with open(file=os.path.join(self.dir_SLAM_path, "DenseFrames_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.dense_scans = pickle.load(f)

    def _get_keypoints(self):
        with open(file=os.path.join(self.dir_SLAM_path, "keyPoints_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.keypoints = pickle.load(f)

    def _get_descriptors(self):
        with open(file=os.path.join(self.dir_SLAM_path, "Descriptors_FPFH_kitti_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.descriptors = pickle.load(f)

    def set_current_pose_idx(self, idx):
        if idx >= len(self.poses):
            idx = len(self.poses) - 1
        self.local_graph_range[1] = idx

        kp_num = 0
        for pc_idx in range(self.local_graph_range[1], 0, -1):
            kp_num += len(self.keypoints[pc_idx])
            if kp_num > 150:
                self.local_graph_range[0] = pc_idx
                break

    def _keypoints_l2_matching(self):
        local_keypoints = np.zeros((1,3))
        local_descriptors = np.zeros((1,33))
        global_keypoints = np.zeros((1,3))
        global_descriptors = np.zeros((1,33))

        for idx in range(self.local_graph_range[0], self.local_graph_range[1]):
            if self.keypoints[idx].shape[0]: local_keypoints = np.vstack((local_keypoints, self.keypoints[idx]))
            if len(self.descriptors[idx]): local_descriptors = np.vstack((local_descriptors, self.descriptors[idx]))
        for idx in range(self.local_graph_range[0]):
            if self.keypoints[idx].shape[0]: global_keypoints = np.vstack((global_keypoints, self.keypoints[idx]))
            if len(self.descriptors[idx]): global_descriptors = np.vstack((global_descriptors, self.descriptors[idx]))
        
        local_keypoints = local_keypoints[1:]
        local_descriptors = local_descriptors[1:]
        global_keypoints = global_keypoints[1:]
        global_descriptors = global_descriptors[1:]

        threshold = 30.0
        distance_matrix = cdist(local_descriptors, global_descriptors)
        matched_indices = np.where((distance_matrix <= threshold))
        # matched_indices = np.where((distance_matrix <= threshold) & (distance_matrix != 0.0))
        matched_keypoints1 = matched_indices[0]
        matched_keypoints2 = matched_indices[1]

        self.matching_line = Marker()
        for i in range(len(matched_keypoints1)):
            self.matching_line.header.frame_id = "/camera_init"
            self.matching_line.type = Marker.LINE_LIST
            self.matching_line.action = Marker.ADD
            line = Point(x=local_keypoints[matched_keypoints1[i]][0], y=local_keypoints[matched_keypoints1[i]][1], z=local_keypoints[matched_keypoints1[i]][2])
            self.matching_line.points.append(line)
            line = Point(x=global_keypoints[matched_keypoints2[i]][0], y=global_keypoints[matched_keypoints2[i]][1], z=global_keypoints[matched_keypoints2[i]][2])
            self.matching_line.points.append(line)
            self.matching_line.colors.append(ColorRGBA(1.0,0,0,1.0))
            self.matching_line.colors.append(ColorRGBA(1.0,0,0,1.0))
            self.matching_line.scale.x = 0.01

        # self.pub_matching_line.publish(self.matching_line)

        print(matched_keypoints1)
        print(matched_keypoints2)
    
    def make_map(self):        
        #slam_keypoints
        self.keypoints_msg = PointCloud2()
        self.keypoints_local_msg = PointCloud2()
        keypoints_msg_pc = o3d.geometry.PointCloud()
        keypoints_msg_pc_local = o3d.geometry.PointCloud()
        
        for pc_idx in range(self.local_graph_range[1]):
            pc_add = o3d.geometry.PointCloud()
            pc_add.points = o3d.utility.Vector3dVector(self.keypoints[pc_idx])

            if pc_idx >= self.local_graph_range[0]: keypoints_msg_pc_local += pc_add
            else: keypoints_msg_pc += pc_add
               
        self.keypoints_msg = orh.o3dpc_to_rospc(keypoints_msg_pc)
        self.keypoints_msg.header.frame_id = "/camera_init"
        self.keypoints_local_msg = orh.o3dpc_to_rospc(keypoints_msg_pc_local)
        self.keypoints_local_msg.header.frame_id = "/camera_init"
        
        self.pub_SLAM_keypoints.publish(self.keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.keypoints_local_msg)
        
        # slam_poses
        self.keyposes_msg = PoseArray()
        for pose in self.poses:
            odom_msg_chach = Pose()
            odom_msg_chach.position.x = pose[0]
            odom_msg_chach.position.y = pose[1]
            odom_msg_chach.position.z = pose[2]
            odom_msg_chach.orientation.x = pose[3]
            odom_msg_chach.orientation.y = pose[4]
            odom_msg_chach.orientation.z = pose[5]
            odom_msg_chach.orientation.w = pose[6]
            self.keyposes_msg.poses.append(odom_msg_chach)
        self.keyposes_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_poses.publish(self.keyposes_msg)

        # slam_map
        self.map_msg = PointCloud2()
        map_msg_pc = o3d.geometry.PointCloud()
        for pc in self.dense_scans:
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(pc))
            map_msg_pc += pc_chach
        self.map_msg = orh.o3dpc_to_rospc(map_msg_pc.voxel_down_sample(voxel_size=0.3))
        self.map_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_map.publish(self.map_msg)
        
    def pub_map(self):
        self.pub_SLAM_keypoints.publish(self.keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.keypoints_local_msg)
        self.pub_SLAM_poses.publish(self.keyposes_msg)
        self.pub_SLAM_map.publish(self.map_msg)
        print("pub complite")

    def fpfh_descriptor_test(self):
        
        for idx, kps in enumerate(self.keypoints[330:350]):
            for kp in kps:
                print("      [keypoint]: ",kp)
                scan_np = self.dense_scans[idx+330]
                
                normals = compute_normals(scan_np, scan_np, k=50, radius=0.4)

                print(normals)
                
                # scan_o3d = o3d.geometry.PointCloud()
                # scan_o3d.points = o3d.utility.Vector3dVector(np.array(scan_np))
                # scan_o3d_kp = o3d.geometry.PointCloud()
                # kp = 
                # scan_o3d_kp.points = o3d.utility.Vector3dVector(np.array([kp, kp+0.01]))
                # scan_o3d += scan_o3d_kp

                # scan_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
                # o3d.visualization.draw_geometries([scan_o3d],
                #                   zoom=0.3412,
                #                   front=[0.4257, -0.2125, -0.8795],
                #                   lookat=[2.6172, 2.0475, 1.532],
                #                   up=[-0.0694, -0.9768, 0.2024],
                #                   point_show_normal=True)

                # descriptors_test = extract_fpfh(scan_o3d, 0.2)

                # for i in descriptors_test[-1]:
                #     print(i, end=" ")
                # print("")
                # for i in descriptors_test[-2]:
                #     print(i, end=" ")
                # print("dist: ")
                # print(np.linalg.norm(descriptors_test[-1]-descriptors_test[-2]))
                # for i in [20, 50, 200, 405, 800, 2000, 10000, 12200]:
                #     print(np.linalg.norm(descriptors_test[-1]-descriptors_test[i]), end=" ")

                # input("go to next? >> ")


def handle_sigint(signal, frame):
    print("\n ---cancel by user---")
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('kitti_dataset_setter')
    signal.signal(signal.SIGINT, handle_sigint)

    Data = dataset(args=args)

    pose = input("set new pose idx >> ")
    Data.set_current_pose_idx(int(pose))
    Data.make_map()
    Data.fpfh_descriptor_test()
    # Data._keypoints_l2_matching()
    while True:
        pose = input("map massage publist again? or set new pose idx >> ")
        if pose != "":
            Data.set_current_pose_idx(int(pose))
            Data.make_map()
            # Data._keypoints_l2_matching()
        Data.pub_map()