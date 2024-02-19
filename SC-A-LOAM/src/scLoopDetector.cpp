#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <tuple>
#include <pcl/io/vtk_lib_io.h>

#include <tf/transform_datatypes.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/normal_3d.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/point_types_conversion.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/rops_estimation.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <aloam_velodyne/LCPair.h>
#include <aloam_velodyne/LocalMapAndPose.h>
#include <aloam_velodyne/PointCloud2List.h>
#include <aloam_velodyne/PointCloud2PoseIdx.h>


#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

typedef pcl::SHOT352 ShotFeature;

std::mutex mKeyFrameBuf;
std::mutex mLocalMapBuf;
std::queue<pcl::PointCloud<PointType>::Ptr> keyFrameQue;
std::queue<pcl::PointCloud<PointType>::Ptr> localMapPclQue;
std::queue<geometry_msgs::Pose> localMapPoseQue;

SCManager scManager;
double scDistThres, scMaximumRadius;
std::string sc_source = "keyframe";

ros::Publisher pubLCdetectResult, pubKeyPointResult, pubKeyPointDisplay, pubPCtest;

int KeyFrameNum = 0;
int DenseFrameNum = 0;

Eigen::Affine3f rosPoseToEigenAffine(const geometry_msgs::Pose& rosPose) {
    Eigen::Affine3f eigenAffine;

    // Translation
    eigenAffine.translation() << rosPose.position.x, rosPose.position.y, rosPose.position.z;

    // Quaternion rotation
    Eigen::Quaternionf quaternion(
        rosPose.orientation.w,
        rosPose.orientation.x,
        rosPose.orientation.y,
        rosPose.orientation.z
    );
    eigenAffine.linear() = quaternion.toRotationMatrix();

    return eigenAffine;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const geometry_msgs::Pose& rosPose)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = rosPoseToEigenAffine(rosPose);

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr &cloudIn, const geometry_msgs::Pose& rosPose)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = rosPoseToEigenAffine(rosPose);
    transCur = transCur.inverse();

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

PointType global2localPoint(const PointType pointIn, const geometry_msgs::Pose& rosPose)
{
    PointType pointOut;

    Eigen::Affine3f transCur = rosPoseToEigenAffine(rosPose);
    transCur = transCur.inverse();

    pointOut.x = transCur(0,0) * pointIn.x + transCur(0,1) * pointIn.y + transCur(0,2) * pointIn.z + transCur(0,3);
    pointOut.y = transCur(1,0) * pointIn.x + transCur(1,1) * pointIn.y + transCur(1,2) * pointIn.z + transCur(1,3);
    pointOut.z = transCur(2,0) * pointIn.x + transCur(2,1) * pointIn.y + transCur(2,2) * pointIn.z + transCur(2,3);
    pointOut.intensity = pointIn.intensity;

    return pointOut;
}

void keyframeHandler(const sensor_msgs::PointCloud2::ConstPtr &_thisKeyFrame) {
    // ROSmsg 타입의 pointcloud를 pcl::PointCloud 로 변환
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*_thisKeyFrame, *thisKeyFrameDS);

    // 들어온 keyFrame을 keyFrameQue에 push
    mKeyFrameBuf.lock();
    keyFrameQue.push(thisKeyFrameDS);
    mKeyFrameBuf.unlock();
    KeyFrameNum++;
}

void denseFrameHandler(const aloam_velodyne::PointCloud2PoseIdx::ConstPtr &_LocalMapAndPose){
    // ROSmsg 타입의 pointcloud를 pcl::PointCloud 로 변환
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(_LocalMapAndPose->point_cloud1, *thisKeyFrameDS);

    mLocalMapBuf.lock();
    localMapPoseQue.push(_LocalMapAndPose->pose);
    // 들어온 keyFrame을 keyFrameQue에 push
    localMapPclQue.push(thisKeyFrameDS);
    mLocalMapBuf.unlock();
    DenseFrameNum++;
}

void ScancontextProcess(void) {
    float loopClosureFrequency = 30.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()) {
        if (sc_source == "denseFrame") {
            mLocalMapBuf.lock();
            if (localMapPclQue.size() > 0) {
                auto frontData = localMapPclQue.front();
                auto frontPoseData = localMapPoseQue.front();
                localMapPclQue.pop();
                localMapPoseQue.pop();
                mLocalMapBuf.unlock();

                // Make SC.
                // pcl::PointCloud<PointType>::Ptr frontData
                *frontData = *global2local(frontData, frontPoseData);
                scManager.makeAndSaveScancontextAndKeys(*frontData);

                sensor_msgs::PointCloud2 localMapMsg;
                pcl::toROSMsg(*frontData, localMapMsg);
                localMapMsg.header.frame_id = "/camera_init";
                pubPCtest.publish(localMapMsg);

                // Search Loop by SC.
                auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
                int SCclosestHistoryFrameID = detectResult.first;
                if( SCclosestHistoryFrameID != -1 ) { 
                    const int prev_node_idx = SCclosestHistoryFrameID;
                    const int curr_node_idx = DenseFrameNum - 1; // because cpp starts 0 and ends n-1
                    // cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

                    aloam_velodyne::LCPair pair;
                    pair.a_int = prev_node_idx;
                    pair.b_int = curr_node_idx;
                    pubLCdetectResult.publish(pair);
                }
            }
            else{
                mLocalMapBuf.unlock();
            }
        }
        else {
            // If keyFrameQue have keyFrame data, pop out keyFrame.
            mKeyFrameBuf.lock();
            if (keyFrameQue.size() > 0) {
                auto frontData = keyFrameQue.front();
                keyFrameQue.pop();
                mKeyFrameBuf.unlock();

                // Make SC.
                scManager.makeAndSaveScancontextAndKeys(*frontData);

                // Search Loop by SC.
                auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
                int SCclosestHistoryFrameID = detectResult.first;
                if( SCclosestHistoryFrameID != -1 ) { 
                    const int prev_node_idx = SCclosestHistoryFrameID;
                    const int curr_node_idx = KeyFrameNum - 1; // because cpp starts 0 and ends n-1
                    // cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

                    aloam_velodyne::LCPair pair;
                    pair.a_int = prev_node_idx;
                    pair.b_int = curr_node_idx;
                    pubLCdetectResult.publish(pair);
                }
            }
            else{
                mKeyFrameBuf.unlock();
            }
        }

        
        rate.sleep();
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "alaserSC");
	ros::NodeHandle nh;

    nh.param<std::string>("sc_source", sc_source, "keyframe");

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

	ros::Subscriber subKeyFrameDS = nh.subscribe<sensor_msgs::PointCloud2>("/keyframeForLC", 100, keyframeHandler);
    ros::Subscriber subKeyLocalMap = nh.subscribe<aloam_velodyne::PointCloud2PoseIdx>("/denseFrameForLC", 100, denseFrameHandler);

	pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);
    pubPCtest = nh.advertise<sensor_msgs::PointCloud2>("/PCtest_inSC", 100);
    // pubKeyPointResult = nh.advertise<aloam_velodyne::PointCloud2List>("/keyPointResult", 100);

    std::thread threadSC(ScancontextProcess);

 	ros::spin();

	return 0;
}
