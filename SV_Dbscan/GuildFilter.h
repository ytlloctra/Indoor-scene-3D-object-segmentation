#pragma once
#include<pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h> // for KdTree
namespace plug
{

	template<typename PointTin, typename PointTout>
	class GuildFilter : public pcl::PointCloud<PointTin>
	{
	protected:

		using SearcherPtr = typename pcl::search::Search<PointTin>::Ptr;

	public:
		// 输入点云
		inline bool
			setInputCloud(pcl::PointCloud<PointTin>::Ptr& input_cloud)
		{
			m_input = input_cloud;
			return true;
		};
		// k近邻搜索的邻域点数
		inline void
			setKsearch(int nr_k)
		{
			m_k = nr_k;
		}

		// 参数阈值
		inline void
			setEpsilonThresh(double epsilon)
		{
			m_epsilon = epsilon;
		}

		// 导向滤波实现过程
		void filter(pcl::PointCloud<PointTout>& output)
		{
			output = *m_input;
			if (!m_searcher)
				m_searcher.reset(new pcl::search::KdTree<PointTin>(false));

			m_searcher->setInputCloud(m_input);
			// The arrays to be used
			pcl::Indices nn_indices(m_k);
			std::vector<float> nn_dists(m_k);

			for (int i = 0; i < static_cast<int> (m_input->size()); ++i)
			{

				if (m_searcher->nearestKSearch(m_input->points[i], m_k, nn_indices, nn_dists) > 0)
				{
					Eigen::Vector4f point_mean;
					pcl::compute3DCentroid(*m_input, nn_indices, point_mean);

					double neigh_mean_2 = 0.0;
					for (size_t idx = 0; idx < nn_indices.size(); ++idx)
					{
						neigh_mean_2 += m_input->points[nn_indices[idx]].getVector3fMap().squaredNorm();
					}

					neigh_mean_2 /= nn_indices.size();

					double point_mean_2 = point_mean.head<3>().squaredNorm();
					double a = (neigh_mean_2 - point_mean_2) / (neigh_mean_2 - point_mean_2 + m_epsilon);
					pcl::PointXYZRGB b;
					b.x = (1.0 - a) * point_mean[0];
					b.y = (1.0 - a) * point_mean[1];
					b.z = (1.0 - a) * point_mean[2];


					output.points[i].x = a * m_input->points[i].x + b.x;
					output.points[i].y = a * m_input->points[i].y + b.y;
					output.points[i].z = a * m_input->points[i].z + b.z;
				}
			}
		}


	private:
		pcl::PointCloud<PointTin>::Ptr m_input;

		SearcherPtr m_searcher;
		int m_k = 20;
		double m_epsilon = 0.05;
	};
}


