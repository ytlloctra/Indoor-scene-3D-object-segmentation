#include"octree_sample.h"
#include<vector>
#include<pcl/octree/octree.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef Eigen::aligned_allocator<pcl::PointXYZ> AlignedPointT;
//使用八叉树的体素中心点来精简原始点云（此中心点可能不是点云中的点）
void OctreeSample::OctreeCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octree_filter_cloud)
{
	pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(m_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<pcl::PointXYZ, AlignedPointT> voxel_centers;
	octree.getOccupiedVoxelCenters(voxel_centers);

	octree_filter_cloud->width = voxel_centers.size();
	octree_filter_cloud->height = 1;
	octree_filter_cloud->points.resize(octree_filter_cloud->height * octree_filter_cloud->width);
	for (size_t i = 0; i < voxel_centers.size() - 1; i++)
	{
		octree_filter_cloud->points[i].x = voxel_centers[i].x;
		octree_filter_cloud->points[i].y = voxel_centers[i].y;
		octree_filter_cloud->points[i].z = voxel_centers[i].z;
	}
	//std::cout << "体素中心点滤波后点云个数为：" << voxel_centers.size() << std::endl;
};
//使用八叉树的体素中心点的最近邻点来精简原始点云（最终获取的点云都还是原始点云数据中的点）
void OctreeSample::OctreeCenterKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octknn_filter_cloud)
{
	pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(m_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<pcl::PointXYZ, AlignedPointT> voxel_centers;
	octree.getOccupiedVoxelCenters(voxel_centers);
	//-----------K最近邻搜索------------
		//根据下采样的结果，选择距离采样点最近的点作为最终的下采样点
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointIndicesPtr inds = std::shared_ptr<pcl::PointIndices>(new pcl::PointIndices());//采样后根据最邻近点提取的样本点下标索引
	for (size_t i = 0; i < voxel_centers.size(); ++i) {
		pcl::PointXYZ searchPoint;
		searchPoint.x = voxel_centers[i].x;
		searchPoint.y = voxel_centers[i].y;
		searchPoint.z = voxel_centers[i].z;

		int K = 1;//最近邻搜索
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {

			inds->indices.push_back(pointIdxNKNSearch[0]);

		}

	}
	pcl::copyPointCloud(*cloud, inds->indices, *octknn_filter_cloud);

	//std::cout << "体素中心最近邻点滤波后点云个数为：" << octknn_filter_cloud->points.size() << std::endl;
};


void OctreeSample::visualize_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& filter_cloud) {
	//-------------------显示点云-----------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("显示点云"));

	int v1(0), v2(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("point clouds", 10, 10, "v1_text", v1);
	viewer->createViewPort(0.5, 0.0, 1, 1.0, v2);
	viewer->setBackgroundColor(0.1, 0.1, 0.1, v2);
	viewer->addText("filtered point clouds", 10, 10, "v2_text", v2);

	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud", v1);
	viewer->addPointCloud<pcl::PointXYZ>(filter_cloud, "cloud_filtered", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "sample cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloud_filtered", v2);
	//viewer->addCoordinateSystem(1.0);
	//viewer->initCameraParameters();
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


