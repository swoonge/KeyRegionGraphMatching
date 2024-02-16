#!/usr/bin/env python3
import os, sys, signal

from tqdm import tqdm
import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs

import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='A simple kitti publisher')
parser.add_argument('--dir', type=str, default='/media/vision/Seagate/DataSets/kitti/dataset/sequences/', metavar='DIR', help='path to dataset')
parser.add_argument('--hz', type=int, default=10, help='Hz of dataset')
args = parser.parse_args()

def makePointCloud2Msg(points, frame_time, parent_frame, pcd_format):

    ros_dtype = sensor_msgs.PointField.FLOAT32

    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(pcd_format)]

    # header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.from_sec(frame_time))
    
    num_field = len(pcd_format)
    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * num_field),
        row_step=(itemsize * num_field * points.shape[0]),
        data=data
    )

def handle_sigint(signal, frame):
    print("\n publish end.")
    sys.exit(0)

if __name__ == '__main__':
    rospy.init_node('KittiPublisher')
    scan_publisher = rospy.Publisher('velodyne_points', sensor_msgs.PointCloud2, queue_size=10)

    signal.signal(signal.SIGINT, handle_sigint)

    dir_path = args.dir
    r = rospy.Rate(args.hz)
    
    seqence_num = input("seq? >> ")

    scan_dir = os.path.join(os.path.join(dir_path + seqence_num), 'velodyne')
    scan_names = os.listdir(scan_dir)
    scan_names.sort()
    num_frames = scan_names.__len__()

    # parse gt times 
    times = []
    with open(os.path.join(os.path.join(dir_path + seqence_num), 'times.txt')) as csvfile:
        times_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in times_reader:
            times.append(float(line[0]))

    # parse gt poses 
    poses = []
    with open(os.path.join(os.path.join(dir_path + seqence_num), 'poses.txt')) as csvfile:
        poses_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in poses_reader:
            poses.append(float(line[0]))
    
    input("press enter to start publish >> ")
    print("pub start")

    for frame_idx in tqdm(range(num_frames)):
        frame_num_str = scan_names[frame_idx][:-4]

        # pub velodyne scan 
        scan_path = os.path.join(scan_dir, scan_names[frame_idx])
        xyzi = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
        scan_publisher.publish(makePointCloud2Msg(xyzi, times[frame_idx], "KITTI", ['x', 'y', 'z', 'intensity']))

        r.sleep()
    
    print("pub end")
