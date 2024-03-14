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
        rospy.Subscriber("/denseFrameForSaver", PointCloud2PoseIdx, self._denseFrameHandler)

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

        self.sub_flag = True

    def _denseFrameHandler(self, denseFrameMsg):
        self.sub_flag = True
        # denseFrameMsg로부터 sub한 데이터들 저장
        with self.mutexDenseFrame:
            self.rawDenseFramePCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud1)) # localDensemap 저장
            self.rawKeyPointPCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud2)) # localDensemap에 대한 keypoint들 저장
            self.denseFramePoseList.append(np.array([denseFrameMsg.pose.position.x, denseFrameMsg.pose.position.y, denseFrameMsg.pose.position.z])) # 해당 map의 pose 저장

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
        # self._pickle_list_save(SaveDir + "descriptors.pickle", self.keypoint_descriptors_list)
        print("[save] complite")
        exit()

def main(opt):
    KeyPoints = KeyPointStorage(opt)
    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        if len(KeyPoints.rawDenseFramePCList) > 5:
            if(KeyPoints.sub_flag == True):
                print("yet [save]")
                KeyPoints.sub_flag = False
            else:
                print("[save] start")
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