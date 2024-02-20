# !/usr/bin/env python3
# -- coding: utf-8 --

import argparse
# import sys
import rospy
import numpy as np
import open3d as o3d
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

class KeyPointStorage():
    # 여기서 denseFramePC도 모아두고, keyPointPC는 받아서 잘 머지해서 통합해서 모아두고(keyPointPC에), get함수를 통해 여러 pc에 접근하는 방식으로 구성
    def __init__(self, opt) -> None:
        rospy.Subscriber("/denseFrameForLC", PointCloud2PoseIdx, self._denseFrameHandler)

        self.pubPCtest = rospy.Publisher('/PCtest_inKP', PointCloud2, queue_size=100)
        self.pubPCtest2 = rospy.Publisher('/PCtest_inKP2', PointCloud2, queue_size=100)

        self.rawDenseFramePCList = []
        self.rawKeyPointPCList = []
        self.keyPointPCList = []
        self.denseFramePoseList = [] # geometry_msgs.msg.Pose
        self.denseFrameCount = self._keyPointMergeCount = 0
        self.localKeyPointPCList = [] # = o3d.geometry.PointCloud()
        self.localMapRange = opt.localMapRange
        
        # self.keyPointPC.points = o3d.utility.Vector3dVector(np.array([]))

        self.mutexDenseFrame = threading.Lock()

    def _denseFrameHandler(self, denseFrameMsg):
        with self.mutexDenseFrame:
            self.rawDenseFramePCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud1))
            self.rawKeyPointPCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud2))
            # self.keyPointPC += self.keyPointPCList[-1]
            self.denseFramePoseList.append([denseFrameMsg.pose.position.x, denseFrameMsg.pose.position.y, denseFrameMsg.pose.position.z])
            self.keyPointPCList.append(o3d.geometry.PointCloud())
            self.denseFrameCount += 1
        if self._keyPointMergeCount != (self.denseFrameCount//5):
            self._keyPointMerge()
            self._keyPointMergeCount = self.denseFrameCount//5
            print(self._keyPointMergeCount, self.denseFrameCount)
            
    def _keyPointMerge(self):
        localDensePcd = o3d.geometry.PointCloud()
        for i in range(self.denseFrameCount-10, self.denseFrameCount):
            localDensePcd += self.rawKeyPointPCList[i]
        # print("_keyPointMerge 포인트 개수: ", len(localDensePcd.points))
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            # labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
        labels = np.array(localDensePcd.cluster_dbscan(eps=0.015, min_points=3, print_progress=False))
        max_label = labels.max()
        
        mergedKeypoints = [[] for _ in range(max_label+1)]
        for idx, label in enumerate(labels):
            if label >= 0:
                mergedKeypoints[label].append(np.array(localDensePcd.points[idx]))
        # print(mergedKeypoints)
        keypointChach = o3d.geometry.PointCloud()
        for i in mergedKeypoints:
            p = np.mean(i, axis=0)
            keypointChach.points.append(p)
        self.localKeyPointPCList.append(keypointChach)

        self.optimalKeyPointPCList()
            
    def optimalKeyPointPCList(self):
        # print(len(self.localKeyPointPCList))
        if len(self.localKeyPointPCList) >= self.localMapRange//5:
            if (self.denseFrameCount > self.localMapRange*2):
                poseRange = self.denseFrameCount-self.localMapRange*2
            else:
                poseRange = 0
            keypointChach = self.localKeyPointPCList[0]
            self.localKeyPointPCList = self.localKeyPointPCList[1:]
            poseChach = o3d.geometry.PointCloud()
            poseChach.points= o3d.utility.Vector3dVector(np.array(self.denseFramePoseList)[poseRange:])
            for p in keypointChach.points:
                distances = np.linalg.norm(poseChach.points - p, axis=1)
                # print(np.min(np.linalg.norm(poseChach.points - p, axis=1)))
                if (np.min(np.linalg.norm(poseChach.points - p, axis=1))) > 30:
                    continue
                self.keyPointPCList[np.argmin(distances)+poseRange].points.append(p)

                # 여기서 과조건이 걸리는지 이거 하면 좀 많은 point가 저장이 되지 않고 날아가 버린다. 여기 수정하면 keypoint쪽은 마무리 될듯.
                # if len(self.keyPointPCList[np.argmin(distances)+poseRange].points) > 0:
                #     if (np.argmin(np.linalg.norm(self.keyPointPCList[np.argmin(distances)+poseRange].points - p, axis=1)) > 0.15):
                #         self.keyPointPCList[np.argmin(distances)+poseRange].points.append(p)
                # else:
                #     self.keyPointPCList[np.argmin(distances)+poseRange].points.append(p)
                    
            
        # rebatchRange = range(len(self.keyPointPCList)-2*self.localMapRange, len(self.keyPointPCList)-self.localMapRange)
        # for i in rebatchRange:
        #     for point in self.keyPointPCList[i]:
        #         for p in rebatchRange:
        #             self.denseFramePoseList[]

    def testDisplay(self):
        # pcMsg = PointCloud2
        PCChach = o3d.geometry.PointCloud()
        for i in self.keyPointPCList:
            PCChach+=i
        print("[KeyPoints 길이] : ", len(PCChach.points))
        pcMsg = orh.o3dpc_to_rospc(PCChach)
        pcMsg.header.frame_id = "/camera_init"
        self.pubPCtest.publish(pcMsg)

        PCChach2 = o3d.geometry.PointCloud()
        for i in self.localKeyPointPCList:
            PCChach2+=i
        pcMsg2 = orh.o3dpc_to_rospc(PCChach2)
        pcMsg2.header.frame_id = "/camera_init"
        self.pubPCtest2.publish(pcMsg2)



def main(opt):
    KeyPoints = KeyPointStorage(opt)
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        if (opt.visualize) and (KeyPoints.denseFrameCount > 0):
            
            KeyPoints.testDisplay()
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