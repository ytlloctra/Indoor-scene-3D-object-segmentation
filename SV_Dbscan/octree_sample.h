#pragma once
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

class OctreeSample {
    //���ݳ�Ա
private:
    float m_resolution;
public:
    OctreeSample(float resolution = 0.002) :
        m_resolution(resolution) {}
    ~OctreeSample() {}

    void OctreeCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octree_filter_cloud);
    void OctreeCenterKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octknn_filter_cloud);
    void visualize_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filter_cloud);

};