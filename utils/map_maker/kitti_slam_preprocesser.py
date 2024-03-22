#!/usr/bin/env python3
import os, sys, signal,rospy, argparse, csv

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy, pickle
import open3d as o3d

from open3d_ros_helper import open3d_ros_helper as orh
from geometry_msgs.msg import Pose, PoseArray # PoseArray, Pose
from sensor_msgs.msg import PointCloud2

parser = argparse.ArgumentParser(description='A simple kitti publisher')
parser.add_argument('--dataset', type=str, default='slam', help='select dataset [slam/gt/all]')
parser.add_argument('--gt_dir', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', metavar='DIR', help='path to dataset')
parser.add_argument('--slam_dir', type=str, default='/media/vision/Seagate/DataSets/KRGM/test_data/', metavar='DIR', help='path to SLAM dataset')
parser.add_argument('--local_global', type=bool, default=False, help='')
parser.add_argument('--seq_num', type=str, default='00', help='seq_num')
args = parser.parse_args()

class dataset():
    def __init__(self, args) -> None:
        self.gt_seq = args.seq_num
        if args.dataset != "slam":
            self.pub_GT_map = rospy.Publisher('/gt_map', PointCloud2, queue_size=100)
            self.pub_GT_poses = rospy.Publisher('/gt_odom', PoseArray, queue_size=100)

            self.gt_dir_path = args.gt_dir
            self.scan_dir = os.path.join(os.path.join(args.dir + args.seq_num), 'velodyne')
            self.scan_names = os.listdir(self.scan_dir)
            self.scan_names.sort()
            self.num_frames = self.scan_names.__len__()

            self.gt_poses = []
            self.gt_times = []
            self.gt_scans = []
            self.key_idx = []
            # self.icp_keypose = []

            self._get_GT_poses()
            self._get_GT_times()
            self._get_GT_scans()
            self._get_key_idx()
            print("[Load] GT data complite")

        if args.dataset != "gt":
            self.dir_SLAM_path = args.slam_dir
            self.pub_SLAM_map = rospy.Publisher('/slam_map', PointCloud2, queue_size=100)
            self.pub_SLAM_keypoints = rospy.Publisher('/slam_keypoints', PointCloud2, queue_size=100)
            self.pub_SLAM_keypoints_local = rospy.Publisher('/slam_keypoints_local', PointCloud2, queue_size=100)
            self.pub_SLAM_poses = rospy.Publisher('/slam_odom', PoseArray, queue_size=100)

            self.slam_poses = []
            self.dense_scans = []
            self.keypoints = []
            self.merged_keypoints = []

            self._get_SLAM_poses()
            self._get_dense_frames()
            self._get_keypoints()
            self._mergging_keypoints()
            print("[Load] SLAM data complite")

    def _transfrom_cam2velo(self, Tcam):
        R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                    -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02]).reshape(3, 3)
        t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
        cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
        return Tcam @ cam2velo
    
    def _get_GT_poses(self):
        with open(os.path.join(os.path.join(self.gt_dir_path + self.gt_seq), 'poses.txt')) as csvfile:
            poses_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            T1 = T2 = np.eye(4,4)
            T1[:3,:3] = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
            T2[:3,:3] = R.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()

            for line in poses_reader:
                pose_raw = np.concatenate((np.array(line).reshape(3, 4).astype(np.float64), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
                pose_to_velo = self._transfrom_cam2velo(pose_raw)
                pose = np.matmul(T2, np.matmul(T1, pose_to_velo))
                self.gt_poses.append(pose)

    def _get_GT_times(self):
        with open(os.path.join(os.path.join(self.dir_path + self.gt_seq), 'times.txt')) as csvfile:
            times_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for line in times_reader:
                self.gt_times.append(float(line[0]))

    def _get_GT_scans(self):
        for frame_idx in range(self.num_frames):
            scan_path = os.path.join(self.scan_dir, self.scan_names[frame_idx])
            xyzi = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
            self.scans.append(xyzi[np.linalg.norm(xyzi[:,:3], axis=1) > 3.0])
    
    def _get_key_idx(self):
        anc_pose = self.gt_poses[0]
        translation_accumulated = rotation_accumulated = 0.0
        self.key_idx.append(i)

        for i, pose in enumerate(self.gt_poses[1:]):
            relative_pose = np.linalg.inv(anc_pose) @ pose
            translation = relative_pose[:3, 3]
            translation_accumulated += np.sum(translation)

            rotation_matrix = relative_pose[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            rotation_accumulated += sum(euler_angles)

            if ((translation_accumulated > 3.0) or (rotation_accumulated > 10.0)):
                translation_accumulated = rotation_accumulated = 0.0
                self.key_idx.append(i-1)
                anc_pose = pose

    def _get_SLAM_poses(self):
        with open(file=os.path.join(self.dir_SLAM_path, "Poses_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.slam_poses = pickle.load(f)
        self.merged_keypoints = [o3d.geometry.PointCloud() for _ in range(len(self.slam_poses)+1)]
        self.keypoints_descriptors = [[] for _ in range(len(self.slam_poses)+1)]

    def _get_dense_frames(self):
        with open(file=os.path.join(self.dir_SLAM_path, "DenseFrames_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.dense_scans = pickle.load(f)

    def _get_keypoints(self):
        with open(file=os.path.join(self.dir_SLAM_path, "keyPoints_" + self.gt_seq + ".pickle"), mode='rb') as f:
            self.keypoints = pickle.load(f)

    def _mergging_keypoints(self, merge_range = 4):
        for i in tqdm(range(merge_range, len(self.keypoints) - merge_range, merge_range)):
            local_keypoints = o3d.geometry.PointCloud()
            
            # merge_range 범위의 ketpoints들을 합친다.
            for r in range(i - merge_range, i + merge_range):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.keypoints[r])
                local_keypoints += pcd

            # dbscan 수행
            labels = np.array(local_keypoints.cluster_dbscan(eps=0.15, min_points=4, print_progress=False))
            if len(labels) < 1: pass
            else: 
                max_label = labels.max() # max_label == 클러스터 개수
                # mergedKeypoints에 각 label(클러스터)끼리 배열
                merged_keypoints_chach = [[] for _ in range(max_label+1)]
                for idx, label in enumerate(labels):
                    if label >= 0:
                        merged_keypoints_chach[label].append(np.array(local_keypoints.points[idx]))

                # keypointChach에 각 클러스터의 평균점을 추가
                keypoint_chach = o3d.geometry.PointCloud()
                for pi in merged_keypoints_chach:
                    p = np.mean(pi, axis=0)
                    keypoint_chach.points.append(p)
                # self.localKeyPointPCList.append(keypointChach)

                # keypointChach의 점들을 가장 가까운 keypose에 assign 및 merge
                self.assign_keypoints_to_closest_pose(keypoint_chach, i - 2*merge_range, i + 2*merge_range)

    def assign_keypoints_to_closest_pose(self, keypoint_chach, range_1, range_2):
        if range_1 < 0: range_1 = 0
        if range_2 >= len(self.slam_poses): range_2 = len(self.slam_poses) - 1

        # keypose중 최근의 keypose만 탐색하기 위해 범위의 keypose를 pointcloud형식으로 추출(후에 거리계산이 편하도록)
        poseChach = o3d.geometry.PointCloud()
        poseChach.points= o3d.utility.Vector3dVector(np.array(self.slam_poses)[:,:3][range_1:range_2])

        for p in keypoint_chach.points:
            # 최근 범위의 poseChach와의 거리 계산
            distances = np.linalg.norm(poseChach.points - p, axis=1)
            if (np.min(np.linalg.norm(poseChach.points - p, axis=1))) > 25: # 가끔 위로 튀는 점들 제거
                continue
            idx = np.argmin(distances)+range_1

            if len(self.merged_keypoints[idx].points) == 0:
                self.merged_keypoints[idx].points.append(p)
                self.keypoints_descriptors[idx].append(self._create_descriptor(idx, p, 0.2))
            else:
                if (np.min(np.linalg.norm(self.merged_keypoints[idx].points - p, axis=1)) > 0.5):
                    self.merged_keypoints[idx].points.append(p)
                    self.keypoints_descriptors[idx].append(self._create_descriptor(idx, p, 0.2))
                else:
                    continue
    
    def _create_descriptor(self, map_idx, point, voxel_size):
        pc = self.dense_scans[map_idx]
        # 일정 거리 설정
        distance_threshold = voxel_size * 7
        # 거리 계산
        distances = np.linalg.norm(pc - point, axis=1)
        # 거리가 일정 값 이하인 점 선택
        selected_points = np.array(pc)[distances < distance_threshold]
        # 선택된 점들로 새로운 PointCloud 생성
        pc_cropped = o3d.geometry.PointCloud()
        pc_cropped.points = o3d.utility.Vector3dVector(selected_points)
        pc_cropped.points.append(point)
        # fpfh = self._compute_FPFH_descriptor(pc_cropped, voxel_size) # fpfh -> numpy.ndarray
        return self._compute_FPFH_descriptor(pc_cropped, voxel_size).data[:, -1]

    def _compute_FPFH_descriptor(self, pcd, voxel_size):
        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_fpfh

    def save_final_keypoints(self):
        #slam_keypoints
        self.slam_keypoints_msg = PointCloud2()
        slam_keypoints_msg_pc = o3d.geometry.PointCloud()
        for pc in self.merged_keypoints:
            slam_keypoints_msg_pc += pc
        self.slam_keypoints_msg = orh.o3dpc_to_rospc(slam_keypoints_msg_pc)
        self.slam_keypoints_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_keypoints.publish(self.slam_keypoints_msg)
        
        # slam_poses
        self.slam_keyposes_msg = PoseArray()
        for pose in tqdm(self.slam_poses):
            odom_msg_chach = Pose()
            odom_msg_chach.position.x = pose[0]
            odom_msg_chach.position.y = pose[1]
            odom_msg_chach.position.z = pose[2]
            odom_msg_chach.orientation.x = pose[3]
            odom_msg_chach.orientation.y = pose[4]
            odom_msg_chach.orientation.z = pose[5]
            odom_msg_chach.orientation.w = pose[6]
            self.slam_keyposes_msg.poses.append(odom_msg_chach)
        self.slam_keyposes_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_poses.publish(self.slam_keyposes_msg)

        # slam_map
        self.slam_map_msg = PointCloud2()
        slam_map_msg_pc = o3d.geometry.PointCloud()
        for pc in self.dense_scans:
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(pc))
            slam_map_msg_pc += pc_chach
        self.slam_map_msg = orh.o3dpc_to_rospc(slam_map_msg_pc.voxel_down_sample(voxel_size=0.3))
        self.slam_map_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_map.publish(self.slam_map_msg)

    #     for idx in tqdm(self.key_idx):
    #         # pose
    #         odom_msg_chach = Pose()
    #         odom_msg_chach.position.x = self.gt_poses[idx][0,3]
    #         odom_msg_chach.position.y = self.gt_poses[idx][1,3]
    #         odom_msg_chach.position.z = self.gt_poses[idx][2,3]
    #         qw, qx, qy, qz = rotation_matrix_to_quaternion(self.gt_poses[idx][:3, :3])
    #         odom_msg_chach.orientation.w = qw
    #         odom_msg_chach.orientation.x = qx
    #         odom_msg_chach.orientation.y = qy
    #         odom_msg_chach.orientation.z = qz
    #         self.odom_gt_msg.poses.append(odom_msg_chach)
    #     # ------------------------------------------
            
    #     self.odom_gt_msg.header.frame_id = "/camera_init"
    #     self.odom_slam_msg.header.frame_id = "/camera_init"

    #     self.pub_SLAM_poses.publish(self.odom_slam_msg)
    #     self.pub_GT_poses.publish(self.odom_gt_msg)
        
    def make_certain_map(self, pose_idx):
        #slam_keypoints
        self.slam_keypoints_msg = PointCloud2()
        self.slam_keypoints_local_msg = PointCloud2()
        slam_keypoints_msg_pc = o3d.geometry.PointCloud()
        slam_keypoints_msg_pc_local = o3d.geometry.PointCloud()
        
        for pc_idx in range(pose_idx, 0, -1):
            slam_keypoints_msg_pc_local += self.merged_keypoints[pc_idx]
            if len(slam_keypoints_msg_pc_local.points) > 40:
                slam_keypoints_msg_pc += self.merged_keypoints[pc_idx]

        self.slam_keypoints_msg = orh.o3dpc_to_rospc(slam_keypoints_msg_pc)
        self.slam_keypoints_msg.header.frame_id = "/camera_init"
        self.slam_keypoints_local_msg = orh.o3dpc_to_rospc(slam_keypoints_msg_pc_local)
        self.slam_keypoints_local_msg.header.frame_id = "/camera_init"
        
        self.pub_SLAM_keypoints.publish(self.slam_keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.slam_keypoints_local_msg)
        
        # slam_poses
        self.slam_keyposes_msg = PoseArray()
        for pose in tqdm(self.slam_poses):
            odom_msg_chach = Pose()
            odom_msg_chach.position.x = pose[0]
            odom_msg_chach.position.y = pose[1]
            odom_msg_chach.position.z = pose[2]
            odom_msg_chach.orientation.x = pose[3]
            odom_msg_chach.orientation.y = pose[4]
            odom_msg_chach.orientation.z = pose[5]
            odom_msg_chach.orientation.w = pose[6]
            self.slam_keyposes_msg.poses.append(odom_msg_chach)
        self.slam_keyposes_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_poses.publish(self.slam_keyposes_msg)

        # slam_map
        self.slam_map_msg = PointCloud2()
        slam_map_msg_pc = o3d.geometry.PointCloud()
        for pc in self.dense_scans:
            pc_chach = o3d.geometry.PointCloud()
            pc_chach.points = o3d.utility.Vector3dVector(np.array(pc))
            slam_map_msg_pc += pc_chach
        self.slam_map_msg = orh.o3dpc_to_rospc(slam_map_msg_pc.voxel_down_sample(voxel_size=0.3))
        self.slam_map_msg.header.frame_id = "/camera_init"
        self.pub_SLAM_map.publish(self.slam_map_msg)
        
    def pub_map(self):
        self.pub_SLAM_keypoints.publish(self.slam_keypoints_msg)
        self.pub_SLAM_poses.publish(self.slam_keyposes_msg)
        self.pub_SLAM_map.publish(self.slam_map_msg)
        print("pub complite")

    def pub_certain_map(self):
        self.pub_SLAM_keypoints.publish(self.slam_keypoints_msg)
        self.pub_SLAM_keypoints_local.publish(self.slam_keypoints_local_msg)
        self.pub_SLAM_poses.publish(self.slam_keyposes_msg)
        self.pub_SLAM_map.publish(self.slam_map_msg)
        print("pub complite")


def handle_sigint(signal, frame):
    print("\n ---cancel by user---")
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('kitti_dataset_setter')
    signal.signal(signal.SIGINT, handle_sigint)

    Data = dataset(args=args)

    if args.local_global == True:
        pose = input("set new pose idx >> ")
        Data.make_certain_map(int(pose))
        while True:
            pose = input("map massage publist again? or set new pose idx >> ")
            if pose != "":
                Data.make_certain_map(int(pose))
            Data.pub_certain_map()
    else:
        Data.make__map()
        while True:
            input("map massage publist again?")
            Data.pub_map()