# !/usr/bin/env python3
# -- coding: utf-8 --

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
    def __init__(self) -> None:
        rospy.Subscriber("/denseFrameForLC", PointCloud2PoseIdx, self.denseFrameHandler)

        self.pubPCtest = rospy.Publisher('/PCtest_inKP', PointCloud2, queue_size=100)

        self.denseFramePCListPCList = []
        self.keyPointPCList = []
        self.denseFramePoseList = [] # geometry_msgs.msg.Pose
        self.denseFrameCount = 0
        self.keyPointPC = o3d.geometry.PointCloud()
        # self.keyPointPC.points = o3d.utility.Vector3dVector(np.array([]))

        self.mutexDenseFrame = threading.Lock()

    def denseFrameHandler(self, denseFrameMsg):
        with self.mutexDenseFrame:
            self.denseFramePCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud1))
            self.keyPointPCList.append(orh.rospc_to_o3dpc(denseFrameMsg.point_cloud2))
            self.keyPointPC += self.keyPointPCList[-1]
            self.denseFramePoseList.append(denseFrameMsg.pose)
            self.denseFrameCount = denseFrameMsg.idx

    def testDisplay(self):
        # pcMsg = PointCloud2
        pcMsg = orh.o3dpc_to_rospc(self.keyPointPC)
        pcMsg.header.frame_id = "/camera_init"
        self.pubPCtest.publish(pcMsg)


def main():
    KeyPoints = KeyPointStorage()
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        if KeyPoints.denseFrameCount > 0:
            print("[KeyPoints 길이] : ", KeyPoints.denseFrameCount)
            KeyPoints.testDisplay()
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('KRGMLoopDetector',anonymous=True)
    main()