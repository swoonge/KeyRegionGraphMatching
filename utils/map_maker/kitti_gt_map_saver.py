# !/usr/bin/env python3
# -- coding: utf-8 --
# run on scaloam
import argparse, time, sys, signal, rospy

import numpy as np
import open3d as o3d
import pickle
from tqdm import tqdm

from open3d_ros_helper import open3d_ros_helper as orh
from aloam_velodyne.msg import PointCloud2PoseIdx

import threading

from utils import extract_fpfh

LocalMapBoundary = 25.0

SaveDir = "/media/vision/Seagate/DataSets/KRGM/kitti/00(02,05)/"

class KeyPointStorage():
    # 여기서 denseFramePC도 모아두고, keyPointPC는 받아서 잘 머지해서 통합해서 모아두고(keyPointPC에), get함수를 통해 여러 pc에 접근하는 방식으로 구성
    def __init__(self, opt) -> None:
        rospy.Subscriber("/denseFrameForSaver", PointCloud2PoseIdx, self._denseFrameHandler)

        self.slam_poses = []
        self.dense_scans = []
        self.keypoints = []
        self.merged_keypoints = []
        self.descriptors = [[]]

        self.denseFrameCount = 0

        self.mutexDenseFrame = threading.Lock()

        self.sub_flag = True

    def _denseFrameHandler(self, denseFrameMsg):
        self.sub_flag = True
        # denseFrameMsg로부터 sub한 데이터들 저장
        with self.mutexDenseFrame:
            self.dense_scans.append(np.asarray(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud1).points)) # localDensemap 저장
            self.keypoints.append(np.asarray(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud2).points)) # localDensemap에 대한 keypoint들 저장
            self.slam_poses.append(np.array([denseFrameMsg.pose.position.x, denseFrameMsg.pose.position.y, denseFrameMsg.pose.position.z, denseFrameMsg.pose.orientation.x, denseFrameMsg.pose.orientation.y, denseFrameMsg.pose.orientation.z, denseFrameMsg.pose.orientation.w])) # 해당 map의 pose 저장
        self.descriptors.append([])
        self.merged_keypoints.append(o3d.geometry.PointCloud())
        self.denseFrameCount+=1

    def _mergging_keypoints(self, merge_range = 4):
        for i in tqdm(range(merge_range, self.denseFrameCount - merge_range, merge_range)):
            local_keypoints = o3d.geometry.PointCloud()
            
            # merge_range 범위의 ketpoints들을 합친다.
            for r in range(i - merge_range, i + merge_range):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.keypoints[r])
                local_keypoints += pcd

            # dbscan 수행
            labels = np.array(local_keypoints.cluster_dbscan(eps=0.10, min_points=4, print_progress=False))
            if len(labels) > 0:
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
                self._assign_keypoints_to_closest_pose(keypoint_chach, i - 2*merge_range, i + 2*merge_range)
    
    def _assign_keypoints_to_closest_pose(self, keypoint_chach, range_1, range_2):
        if range_1 < 0: range_1 = 0
        if range_2 >= len(self.slam_poses): range_2 = len(self.slam_poses) - 1

        # keypose중 최근의 keypose만 탐색하기 위해 범위의 keypose를 pointcloud형식으로 추출(후에 거리계산이 편하도록)
        poseChach = o3d.geometry.PointCloud()
        poseChach.points = o3d.utility.Vector3dVector(np.array(self.slam_poses)[:,:3][range_1:range_2])

        for p in keypoint_chach.points:
            # 최근 범위의 poseChach와의 거리 계산
            distances = np.linalg.norm(np.asarray(poseChach.points)[:,:2] - np.asarray(p[:2]), axis=1)
            md = np.min(distances)
            if md > 23 or md < 1.5: # 가끔 위로 튀는 점들 제거 and 너무 가장자리 점 제거 # 차 경로상에 있는 점 제거
                continue
            idx = np.argmin(distances)+range_1

            self.merged_keypoints[idx].points.append(p)
            self.descriptors[idx].append(self._create_descriptor(idx, p, 0.2))

            # if len(self.merged_keypoints[idx].points) == 0:
            #     self.merged_keypoints[idx].points.append(p)
            #     self.descriptors[idx].append(self._create_descriptor(idx, p, 0.2))
            # else:
            #     if (np.min(np.linalg.norm(self.merged_keypoints[idx].points - p, axis=1)) > 0.4):
            #         self.merged_keypoints[idx].points.append(p)
            #         self.descriptors[idx].append(self._create_descriptor(idx, p, 0.2))
            #     else:
            #         continue

    def _create_descriptor(self, map_idx, point, voxel_size):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.dense_scans[map_idx])
        pc_k = o3d.geometry.PointCloud()
        pc_k.points = o3d.utility.Vector3dVector(np.array([point]))
        pc += pc_k

        radius_normal = voxel_size * 2
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        fpfh_kp = np.array(fpfh.data).T[-1]/201.0
        return fpfh_kp

    def _pickle_list_save(self, file_name, list_data):
        with open(file_name, 'wb') as f:
            pickle.dump(list_data, f)
    
    def _pickle_o3d_list_save(self, file_name, list_data):
        with open(file_name, 'wb') as f:
            for l in list_data:
                pickle.dump(np.array(l.points), f)
    
    def save_map_descriptor_for_mdgat(self):
        self._pickle_list_save(SaveDir + "DenseFrames_kitti_00.pickle", self.dense_scans)
        self._pickle_list_save(SaveDir + "Poses_kitti_00.pickle", self.slam_poses)
        self._pickle_list_save(SaveDir + "keyPoints_kitti_00.pickle", [np.asarray(pc.points) for pc in self.merged_keypoints])
        self._pickle_list_save(SaveDir + "Descriptors_FPFH_kitti_00.pickle", self.descriptors)
        
    def data_processing(self):
        self._mergging_keypoints(merge_range = 4)

def main(opt):
    Storage = KeyPointStorage(opt)
    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        
        if len(Storage.dense_scans) > 5:
            if(Storage.sub_flag == True):
                print("yet [save]")
                Storage.sub_flag = False
            else:
                print("[data processing] start")
                Storage.data_processing()
                print("[data processing] end \n[save] start")
                Storage.save_map_descriptor_for_mdgat()
                print("[save] complite")
                exit()
        else:
            print("SLAM not end")
        rate.sleep()            

if __name__ == "__main__":
    rospy.init_node('KRGMLoopDetector',anonymous=True)
    parser = argparse.ArgumentParser(
        description='Point cloud matching and pose evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--visualize', type=bool, default=True,
        help='Visualize the matches')

    parser.add_argument(
        '--localMapRange', type=int, default=25,
        help='the width of the match line open3d visualization')

    parser.add_argument(
        '--calculate_pose', type=bool, default=True,
        help='Registrate the point cloud using the matched point pairs and calculate the pose')

    parser.add_argument(
        '--learning_rate', type=int, default=0.0001,
        help='Learning rate')
    opt = parser.parse_args()

    main(opt)