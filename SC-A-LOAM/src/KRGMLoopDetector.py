# !/usr/bin/env python3
# -- coding: utf-8 --

import argparse
import time
import sys
import signal

import rospy
import numpy as np
import open3d as o3d
import pickle
# import math
# import random
# from scipy.spatial.distance import cdist, pdist, squareform
# import pygmtools as pygm
# import matplotlib
# import matplotlib.pyplot as plt # for plotting

# from matplotlib.patches import ConnectionPatch # for plotting matching result
# import networkx as nx # for plotting graphs

from open3d_ros_helper import open3d_ros_helper as orh
# from ros_np_multiarray import ros_np_multiarray as rnm
# from sklearn.decomposition import PCA

from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Float64MultiArray 
from aloam_velodyne.msg import PointCloud2PoseIdx
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose

import threading

LocalMapBoundary = 25.0

SaveDir = "/media/vision/Seagate/DataSets/KRGM/test_data/"

class KeyPointStorage():
    # 여기서 denseFramePC도 모아두고, keyPointPC는 받아서 잘 머지해서 통합해서 모아두고(keyPointPC에), get함수를 통해 여러 pc에 접근하는 방식으로 구성
    def __init__(self, opt) -> None:
        rospy.Subscriber("/denseFrameForLC", PointCloud2PoseIdx, self._denseFrameHandler)

        self.pubPCtest = rospy.Publisher('/PCtest_inKP', PointCloud2, queue_size=100)
        self.pubPCtest2 = rospy.Publisher('/PCtest_inKP2', PointCloud2, queue_size=100)

        self.rawDenseFramePCList = []
        self.rawKeyPointPCList = []
        self.keyPointPCList = []
        self.keypoint_descriptors_list = []

        self.denseFramePoseList = [] # geometry_msgs.msg.Pose
        self.denseFrameCount = 0
        # self.localMapRange = opt.localMapRange
        
        self.cumulative_distance = 0.0
        self.keypoint_merge_range = [0, 0, 0, 0, 0]

        self.mutexDenseFrame = threading.Lock()

        self.sub_flag = False

    def _denseFrameHandler(self, denseFrameMsg):
        self.sub_flag = True
        # denseFrameMsg로부터 sub한 데이터들 저장
        with self.mutexDenseFrame:
            self.rawDenseFramePCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud1)) # localDensemap 저장
            self.rawKeyPointPCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud2)) # localDensemap에 대한 keypoint들 저장
            self.denseFramePoseList.append(np.array([denseFrameMsg.pose.position.x, denseFrameMsg.pose.position.y, denseFrameMsg.pose.position.z])) # 해당 map의 pose 저장
            self.keyPointPCList.append(o3d.geometry.PointCloud()) # 차후 _keyPointMerge에서 merge에서 최종 keypoint가 들어갈 빈 list 생성
            self.keypoint_descriptors_list.append([])
            self.denseFrameCount += 1

        # keypointMerge가 수행되는 조건
            # LocalMapBoundary만큼의 경로가 지나갔을 때 한번 씩 수행하면 합리적일 것으로 생각 됨.
            # 따라서 denseFramePose의 변위를 누적하여 누적 이동 거리가 LocalMapBoundary를 넘어설 때 마다 수행.
        if self.denseFrameCount > 1:
            self.cumulative_distance += np.linalg.norm(self.denseFramePoseList[-2] - self.denseFramePoseList[-1])
            
            if self.cumulative_distance > LocalMapBoundary:
                print("[cumulative_distance] ", self.cumulative_distance, ", perform _keyPointMerge()")
                self.cumulative_distance = 0.0
                self.keypoint_merge_range[0:4] = self.keypoint_merge_range[1:]
                self.keypoint_merge_range[4] = self.denseFrameCount - 1 # 길이가 아닌 idx저장

                # 충분히 _keyPointMerge가 실행 되어 keypoint가 누적 되었을 경우에만 실행
                if self.keypoint_merge_range[3] != 0:
                    self._keyPointMerge()
            
    def _keyPointMerge(self):
        localDensePcd = o3d.geometry.PointCloud()
        
        # merge하는 범위는 LocalMapBoundary범위의 두배를 커버하는 범위에 대해 수행
        for i in range(self.keypoint_merge_range[1], self.keypoint_merge_range[3]):
            localDensePcd += self.rawKeyPointPCList[i]
        # dbscan 수행
        labels = np.array(localDensePcd.cluster_dbscan(eps=0.15, min_points=4, print_progress=False))
        if len(labels) < 1: pass
        else: 
            max_label = labels.max() # max_label == 클러스터 개수

            # mergedKeypoints에 각 label(클러스터)끼리 배열
            mergedKeypoints = [[] for _ in range(max_label+1)]
            for idx, label in enumerate(labels):
                if label >= 0:
                    mergedKeypoints[label].append(np.array(localDensePcd.points[idx]))

            # keypointChach에 각 클러스터의 평균점을 추가
            keypointChach = o3d.geometry.PointCloud()
            for i in mergedKeypoints:
                p = np.mean(i, axis=0)
                keypointChach.points.append(p)
            # self.localKeyPointPCList.append(keypointChach)

            # keypointChach의 점들을 가장 가까운 keypose에 assign 및 merge
            self._optimalKeyPointPCList(keypointChach)
                
    def _optimalKeyPointPCList(self, keypointChach):
        # keypose중 최근의 keypose만 탐색하기 위해 범위의 keypose를 pointcloud형식으로 추출(후에 거리계산이 편하도록)
        poseChach = o3d.geometry.PointCloud()
        poseChach.points= o3d.utility.Vector3dVector(np.array(self.denseFramePoseList)[self.keypoint_merge_range[0]:self.keypoint_merge_range[4]])

        for p in keypointChach.points:
            # 최근 범위의 poseChach와의 거리 계산
            distances = np.linalg.norm(poseChach.points - p, axis=1)
            if (np.min(np.linalg.norm(poseChach.points - p, axis=1))) > LocalMapBoundary: # 가끔 위로 튀는 점들 제거
                continue
            idx = np.argmin(distances)+self.keypoint_merge_range[0]

            if len(self.keyPointPCList[idx].points) == 0:
                self.keyPointPCList[idx].points.append(p)
                self.keypoint_descriptors_list[idx].append(self._create_descriptor(idx, p, 0.2))
            else:
                if (np.min(np.linalg.norm(self.keyPointPCList[idx].points - p, axis=1)) > 0.5):
                    self.keyPointPCList[idx].points.append(p)
                    self.keypoint_descriptors_list[idx].append(self._create_descriptor(idx, p, 0.2))
                else:
                    continue

    def _create_descriptor(self, map_idx, point, voxel_size):
        pc = self.rawDenseFramePCList[map_idx]
        
        # 일정 거리 설정
        distance_threshold = voxel_size * 7
        # 거리 계산
        distances = np.linalg.norm(pc.points - point, axis=1)
        # 거리가 일정 값 이하인 점 선택
        selected_points = np.array(pc.points)[distances < distance_threshold]
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
    
    def _pickle_list_save(self, file_name, list_data):
        with open(file_name, 'wb') as f:
            for l in list_data:
                pickle.dump(l, f)
    
    def _pickle_o3d_list_save(self, file_name, list_data):
        with open(file_name, 'wb') as f:
            for l in list_data:
                pickle.dump(np.array(l.points), f)
    
    def save_map_descriptor_for_mdgat(self):
        self._pickle_o3d_list_save(SaveDir + "DenseFrames.pickle", self.rawDenseFramePCList)
        self._pickle_list_save(SaveDir + "Poses.pickle", self.denseFramePoseList)
        self._pickle_o3d_list_save(SaveDir + "keyPoints.pickle", self.keyPointPCList)
        self._pickle_list_save(SaveDir + "descriptors.pickle", self.keypoint_descriptors_list)
        print("[save] complite")
        exit()

    def testDisplay(self):
        # pcMsg = PointCloud2
        PCChach = o3d.geometry.PointCloud()
        for i in self.keyPointPCList[0:self.keypoint_merge_range[0]]:
            PCChach+=i
        # print("[KeyPoints 길이] : ", len(PCChach.points))
        pcMsg = orh.o3dpc_to_rospc(PCChach)
        pcMsg.header.frame_id = "/camera_init"
        self.pubPCtest.publish(pcMsg)

        PCChach2 = o3d.geometry.PointCloud()
        for i in self.keyPointPCList[self.keypoint_merge_range[0]:self.keypoint_merge_range[4]]:
            PCChach2+=i
        pcMsg2 = orh.o3dpc_to_rospc(PCChach2)
        pcMsg2.header.frame_id = "/camera_init"
        self.pubPCtest2.publish(pcMsg2)

def main(opt):
    KeyPoints = KeyPointStorage(opt)
    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        if (opt.visualize) and (KeyPoints.denseFrameCount > 0):
            KeyPoints.testDisplay()
            print(KeyPoints.sub_flag)
            if KeyPoints.sub_flag == True:
                KeyPoints.sub_flag = False
            else:
                print("[end]")
                
                KeyPoints.keypoint_merge_range[0:4] = KeyPoints.keypoint_merge_range[1:]
                KeyPoints.keypoint_merge_range[4] = KeyPoints.denseFrameCount - 1 # 길이가 아닌 idx저장
                KeyPoints._keyPointMerge()

                KeyPoints.save_map_descriptor_for_mdgat()
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