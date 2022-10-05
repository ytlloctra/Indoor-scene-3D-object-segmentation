#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_search.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h> //条件滤波

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <boost/thread/thread.hpp>

#include <time.h>
#include <iostream>
#include <vector>

#include "DBSCAN_simple.h"
#include "DBSCAN_precomp.h"
#include "DBSCAN_kdtree.h"

#include "octree_sample.h"
#include "GuildFilter.h"


#include "testing.h";

using namespace std;


// Visualization, [The CloudViewer](https://pcl.readthedocs.io/projects/tutorials/en/latest/cloud_viewer.html#cloud-viewer)
template <typename PointCloudPtrType>
void show_point_cloud(PointCloudPtrType cloud, std::string display_name) {
  pcl::visualization::CloudViewer viewer(display_name);
  viewer.showCloud(cloud);
 while (!viewer.wasStopped())
  {
   }
}

int main(int argc, char** argv) 
{
    //创建存储点云指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_conditional(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2XYZ(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Octree(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Dbscan(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_segmentation(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_segmentation_Part(new pcl::PointCloud<pcl::PointXYZL>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud_cluster_colour(new pcl::PointCloud<pcl::PointXYZRGB>);
    //-----------------------------------------------------------读入点云数据-----------------------------------------------------------------------------------------------------------
    pcl::PCDReader reader;
    reader.read("floor.pcd", *cloud);
    std::cout << "Raw Point is : " << cloud->points.size () << " data points" << std::endl;
    //show_point_cloud(cloud, "Original point cloud");

    //pcl::PCDReader reader1; // 读入ground truth 模板
    //reader1.read("cloud_Octree_label.pcd", *cloud_segmentation);
    //std::cout << "Truth_ground PointXYZL is : " << cloud_segmentation->points.size() << " data points" << std::endl;
   
   //------------------------------------------------------------去除背景---------------------------------------------------------------------------------------------------------------   
    clock_t start_ransac = clock();
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100); // 最大迭代次数
    seg.setDistanceThreshold(0.012); // 去除地面阈值
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) 
    {
        cout << "Could not estimate a planar anymore." << endl;
    }
    else 
    {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        // filter planar
        extract.setNegative(true);// 保留地面以外的
        extract.filter(*cloud_f);
       // show_point_cloud(cloud_f, "plane filtered point cloud");
        *cloud_filtered = *cloud_f;
    }
   
    cout << "after remove plane size is: " << cloud_filtered->points.size() << endl;
    //show_point_cloud(cloud_filtered, "after remove plane");
    //pcl::io::savePCDFileBinary("cloud_filtered1.pcd", *cloud_filtered);
    
    //--------------去除孤立点--------------------------------------------------------------------------------------------------------------------------------
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>);//实例化条件指针
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
    pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, -0.5))); //大于
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem; // 创建条件滤波器
    condrem.setCondition(range_cond);               // 并用条件定义对象初始化     
    condrem.setInputCloud(cloud_filtered);                // 输入点云
    condrem.setKeepOrganized(true);                 // 设置true则保持点云的结构，保存原有点云结结构就是点的数目没有减少，采用nan代替了。 
    condrem.filter(*cloud_conditional);             // 不在条件范围内的点　被替换为　nan
    // 4、去除nan点
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*cloud_conditional, *cloud_conditional, mapping);
    //show_point_cloud(cloud_conditional, "after Conditional filtered");
    cout << "after Conditional filtered size is: " << cloud_conditional->points.size() << endl;
    //pcl::io::savePCDFileBinary("cloud_conditional2.pcd", *cloud_conditional);

    //--------------超体素过分割------------------------------------------------------------------------------------------------
    float voxel_resultion = 0.001f;   // 设置体素大小，该设置决定底层八叉树的叶子尺寸
    float seed_resultion = 0.01f;     // 设置种子大小，该设置决定超体素的大小

    pcl::SupervoxelClustering<pcl::PointXYZ> super(voxel_resultion, seed_resultion);
    super.setInputCloud(cloud_conditional);      // 输入点云
    super.setNormalImportance(4);    // 设置法向量的权重，即表面法向量影响超体素分割结果的比重。
    super.setColorImportance(0);     // 设置颜色在距离测试公式中的权重，即颜色影响超体素分割结果的比重。
    super.setSpatialImportance(2); // 设置空间距离在距离测试公式中的权重，较高的值会构建非常规则的超体素，较低的值产生的体素会按照法线
    std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr >supervoxl_clustering;
    super.extract(supervoxl_clustering);
    cout << "SV number is：" << supervoxl_clustering.size() << endl;
    // 获取点云对应的超体素分割标签
    pcl::PointCloud<pcl::PointXYZL>::Ptr supervoxel_cloud = super.getLabeledCloud();
    cout << "after SV voxel is：" << supervoxel_cloud->points.size() << endl;
    //-----------------------------------------XYZL2XYZ------------------------------------------------------------
    for (int i = 0; i < supervoxel_cloud->points.size(); i++)
    {
        pcl::PointXYZ p;
        p.x = supervoxel_cloud->points[i].x;
        p.y = supervoxel_cloud->points[i].y;
        p.z = supervoxel_cloud->points[i].z;
        cloud_2XYZ->points.push_back(p);
    }
    int M = supervoxel_cloud->points.size();

    cloud_2XYZ->width = 1;
    cloud_2XYZ->height = M;
    cout << "To XYZ cloud_2XYZ size is:" << cloud_2XYZ->points.size() << endl;
    
    //-----------------------------------------Octree------------------------------------------------------------
    //float resolution = 0.001; //体素的大小
    //OctreeSample oct(resolution);
    ////oct.OctreeCenter(cloud, cloud_filtered);  // 调用八叉树体素中心点滤波程序精简点云
    //oct.OctreeCenterKNN(supervoxel_cloud, cloud_Octree);// 调用八叉树体素中心点最近邻点滤波程序精简点云
    //cout << "Object cloud after Octree_voxel:: " << cloud_Octree->points.size() << " points" << endl;
    //show_point_cloud(cloud_Octree, "after Octree");
    //pcl::io::savePCDFileASCII<pcl::PointXYZ>("cloud_Octree.pcd", *cloud_Octree);
    
    //-----------------------------------------Dbscan---------------------------------------------------------
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_2XYZ);
    // Segmentation, [Euclidean Cluster Extraction](https://pcl.readthedocs.io/projects/tutorials/en/latest/cluster_extraction.html#cluster-extraction)
    std::vector<pcl::PointIndices> cluster_indices;


    // test 1. uncomment the following two lines to test the simple dbscan
     //DBSCANSimpleCluster<pcl::PointXYZ> ec;    // DBSCAN
     //ec.setCorePointMinPts(10);

    // test 2. uncomment the following two lines to test the precomputed dbscan
     //DBSCANPrecompCluster<pcl::PointXYZ>  ec;   // dbscan 预先计算距离矩阵
     //ec.setCorePointMinPts(10);

    // test 3. uncomment the following two lines to test the dbscan with Kdtree for accelerating
    DBSCANKdtreeCluster<pcl::PointXYZ> ec;       // kdtree加速
    ec.setCorePointMinPts(5);// 越小聚类越少

    // test 4. uncomment the following line to test the EuclideanClusterExtraction
     //pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;     // 欧式聚类
     //点10  距离 0.006   判定为相同颜色修改点
    ec.setClusterTolerance(0.005); // eqs 0.004以上
    ec.setMinClusterSize(100); // 聚类过多调大
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_2XYZ);
    ec.extract(cluster_indices);

     /*--------------------------------------------------------------------------------------------------*/
    int j = 0;

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
        cloud_cluster->points.push_back(cloud_2XYZ->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        
        // std::stringstream ss;
        // ss << "cloud_cluster_" << j << ".pcd";
        // pcl::io::savePCDFileASCII(ss.str(), *cloud_cluster);
        // cout << ss.str() << " Saved" << endl;
        j++;
        std::cout << "Cluster  " << j << " size is : " << cloud_cluster->points.size() << "  points" << std::endl;

        for (int i = 0; i < cloud_cluster->points.size(); i++)
        {
            pcl::PointXYZL pp;
            pp.x = cloud_cluster->points[i].x;
            pp.y = cloud_cluster->points[i].y;
            pp.z = cloud_cluster->points[i].z;
            pp.label = j;
            cloud_segmentation_Part->points.push_back(pp);
        }
        cloud_segmentation_Part->width = 1;
        cloud_segmentation_Part->height = cloud_segmentation_Part->points.size();
        // 可视化相关的代码

        //if (j == 1)  // 红色
        //{
        //    R = 255; G = 0; B = 0;
        //}
        //else if (j == 2) // 黄色
        //{
        //    R = 255; G = 255; B = 0;
        //}
        //else if (j == 3) // 品蓝色
        //{
        //    R = 65; G = 105; B = 225;
        //}
        //else if (j == 4)  // 绿色
        //{
        //    R = 0; G = 255; B = 0;
        //}
        //else if (j == 5) //紫色
        //{
        //    R = 160; G = 32; B = 240;
        //}
        //else if (j == 6) //深红
        //{
        //    R = 255; G = 0; B = 255;
        //}
        //else //棕色
        //{
        //    R = 210; G = 180; B = 140;
        //}
        uint8_t R = rand() % (256) + 0;
        uint8_t G = rand() % (256) + 0;
        uint8_t B = rand() % (256) + 0;

        for (int i = 0; i < cloud_cluster->points.size(); i++)
        {
            pcl::PointXYZRGB p;
            p.x = cloud_cluster->points[i].x;
            p.y = cloud_cluster->points[i].y;
            p.z = cloud_cluster->points[i].z;
            p.r = R;
            p.g = G;
            p.b = B;
            cloud_cluster_colour->points.push_back(p);
        }
        cloud_cluster_colour->width = 1;
        cloud_cluster_colour->height = cloud_cluster_colour->points.size();
    }
    clock_t end_ransac = clock();
    cout << "cluster time cost:" << double(end_ransac - start_ransac) / CLOCKS_PER_SEC << " s" << endl;
    show_point_cloud(cloud_cluster_colour, "colored clusters of point cloud");
    std::cout << "XYZRGB points is " << cloud_cluster_colour->points.size() << std::endl;
    std::cout << "XYZL points is " << cloud_segmentation_Part->points.size() << std::endl;
    //-----------------------------------------Guide Filter---------------------------------------------------------
    plug::GuildFilter<pcl::PointXYZRGB, pcl::PointXYZRGB> gf;
    gf.setInputCloud(cloud_cluster_colour);
    gf.setKsearch(20);
    gf.setEpsilonThresh(0.01);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGB>);
    gf.filter(*cloudOut);
    cout << "Object cloud after guild filter:: " << cloudOut->points.size() << " points" << endl;

    //show_point_cloud(cloudOut, "Object cloud after guild filter");
    //-----------------------------------------Precision recall F1-score---------------------------------------------------------
    //
    //Testing test(cloud_segmentation_Part, cloud_segmentation);
    //cout << "VCCS compactness:" << brsv.getVCCS_NCE() << endl;// 紧凑性
    //cout << "BRSS compactness:" << brsv.getBRSS_NCE() << endl;   
    //cout << "Precision is:" << test.eval_precision()  << endl;
    //cout << "Recall is:" << test.eval_recall()  << endl;
    //cout << "F1-score is:" << test.eval_fscore()  << endl;

    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZRGB>("floor1_cluster.pcd", *cloudOut, false);
    return 0;
}