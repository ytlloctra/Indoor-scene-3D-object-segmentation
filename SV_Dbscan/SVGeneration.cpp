#include "SVGeneration.h"

//VCCS��ȡ�����أ�δʹ����ɫ��Ϣ
void SVGeneration::getVCCS(float Rvoxel, float Rseed, PointCloud<PointXYZL>::Ptr &sv_labeled_cloud1, PointCloud<PointXYZL>::Ptr &sv_labeled_cloud2)
{
	Rvoxel_ = Rvoxel;
	Rseed_ = Rseed;
	SupervoxelClustering<PointT> super(Rvoxel, Rseed);
	super.setInputCloud(cloud_);
	super.setNormalCloud(normalcloud_);
	//super.setUseSingleCameraTransform(true);
	//Ȩ��ΪĬ��ֵ���ɸ���ʵ�ʳ�������
	float wc = 0.0f;
	float ws = 1.0f;//ws=2;wn=1
	float wn = 4.0f;//Ĭ��ws=1,wn=4
	super.setColorImportance(wc);
	super.setSpatialImportance(ws);
	super.setNormalImportance(wn);
	super.extract(sv_clusters_); // VCCS�Ľ��
	vccs_nce = calNCE(sv_clusters_);//NCE(3410)������λ�����ĺ�ƽ�����ľ��룬���������ؽ�����
	super.getSupervoxelAdjacency(nei_labels_);
	sv_labeled_cloud1 = super.getLabeledVoxelCloud();//�߽��Ż�ǰ��õĳ�����
	//sv_labeled_cloud1 = super.getVoxelCentroidCloud();//
	//sv_labeled_cloud1 = super.getLabeledCloud();
	//��������ȡ�ȶ������K-means���� 
	for (int i = 0; i < 20; i++)
	{
		map<uint32_t, vector<SVGeneration::BoundaryData>> SVs_boundary;//�洢ÿ��SV�ı߽������
		getSVBoundary(SVs_boundary);//������Ʋ�ؾ��룬͸��ͶӰ�������, �������ʣ���ȡ��ʼ�������ڵı߽��
		expandBSV();//ͨ������߽������Χ���������ĵ�difference���������·��䣬result��boundary refinement+compactness
		updateCentroid();//update sv_clusters_ information for the new interation
	}
	sv_labeled_cloud2 = getLabelCloud();
	brss_nce = calNCE(sv_clusters_);
}
//������Ʋ�ؾ��룬͸��ͶӰ�������, �������ʣ���ȡ��ʼ�������ڵı߽��
void SVGeneration::getSVBoundary(map<uint32_t, vector<BoundaryData>> &boundaries)
{
	for (supervoxelmap::iterator labelsv_itr = sv_clusters_.begin(); labelsv_itr != sv_clusters_.end(); ++labelsv_itr)
	{
		uint32_t label = labelsv_itr->first;
		Supervoxel<PointT>::Ptr cur_sv = labelsv_itr->second;
		Eigen::Vector3f pc = cur_sv->centroid_.getVector3fMap();
		PointCloudT::Ptr voxels = cur_sv->voxels_;
		PointCloud<Normal>::Ptr normals = cur_sv->normals_;
		vector<BoundaryData> pns;
		float curvature = 0;
		for (int i = 0; i < normals->size(); i++)
		{
			curvature += normals->points[i].curvature;
		}
		//curvatures[label] = sum_curvature / cur_normals->size();//��ȡÿ��SV�����ʣ�ƽ����
		curvature = curvature / normals->size();
		float r = 1.0 / curvature;//���ʰ뾶
		for (int i = 0; i < voxels->size(); i++)
		{
			Eigen::Vector3f pi = voxels->points[i].getVector3fMap();//������Ϊԭ��
			pi = pi - pc;//ȥ�Ļ�
			//float Dproj = r*sqrt(pi[0] * pi[0] + pi[1] * pi[1] + 4 * pi[2] * pi[2]) / abs(r - pi[2]);
			float Dproj = r*sqrt(pi[0] * pi[0] + pi[1] * pi[1])/ abs(r + pi[2]);
			//cout << Dproj << endl;
			if (Dproj > Rseed_ / 2)
			{
				BoundaryData pn;
				pn.xyz_ = voxels->points[i].getVector3fMap();
				pn.normal_ = normals->points[i].getNormalVector3fMap();
				pn.curvature_ = normals->points[i].curvature;
				pn.idx_ = i;
				pns.push_back(pn);
			}
			else
				continue;
		}
		boundaries[label] = pns;
	}
	sv_boundaries_ = boundaries;
}
//ͨ������߽������Χ���������ĵ�difference���������·��䣬result��boundary refinement+compactness
void SVGeneration::expandBSV()
{
	map<uint32_t, vector<BoundaryData>>::iterator b_itr;
	for (b_itr = sv_boundaries_.begin(); b_itr != sv_boundaries_.end(); ++b_itr)
	{
		uint32_t label = b_itr->first;//��ǰ�����ر�ǩ
		PointCloud<PointT>::Ptr outliers(new PointCloud<PointT>);
		pair<int_multimap, int_multimap> cur_labels = nei_labels_.equal_range(label);//����sv
		vector<BoundaryData> curBoundaries = b_itr->second;
		Supervoxel<PointT>::Ptr pc0(new Supervoxel<PointT>);
		pc0 = sv_clusters_[label];//��ǰ�߽��ĳ�����
		//ѭ����ǰ�����������б߽��
		for (int i = 0; i < curBoundaries.size(); i++)
		{
			//����ÿ���߽������Χ�����������ĵ����С���룬����¼��Ӧ�ı�ǩ
			BoundaryData pn = curBoundaries[i];	//��ǰ�߽������
			uint32_t min_label = label;
			float min_dist = distCal(pn, pc0);//�뵱ǰ�����ؾ�����Ϊ��Сֵ
			//�ҵ���֮������С�ĳ�����label
			for (auto iter = cur_labels.first; iter != cur_labels.second; ++iter)
			{
				uint32_t cur_label = iter->second;
				Supervoxel<PointT>::Ptr pc = sv_clusters_[cur_label];
				if (distCal(pn, pc) < min_dist)
				{
					min_dist = distCal(pn, pc);
					min_label = cur_label;
				}
			}
			//���ڲ������ʺϵı߽�����
			if (min_label != label)
			{
				PointT p;
				p.x = pn.xyz_[0];
				p.y = pn.xyz_[1];
				p.z = pn.xyz_[2];
				Normal n;
				n.normal_x = pn.normal_[0];
				n.normal_y = pn.normal_[1];
				n.normal_z = pn.normal_[2];
				n.curvature = pn.curvature_;
				sv_clusters_[min_label]->voxels_->push_back(p);
				sv_clusters_[min_label]->normals_->push_back(n);
				outliers->push_back(p);//�洢��ǰsvҪɾ���ı߽��
			}
		}
		//�Ե�ǰ�������ѵ������ı߽�㣨����&���ߣ�����ɾ��
		removeOutliers(outliers, pc0->voxels_, pc0->normals_);
	}
}
//update sv_clusters_ information for the new interation
void SVGeneration::updateCentroid()
{
	float cenDist = 0;
	for (supervoxelmap::iterator labelsv_itr = sv_clusters_.begin(); labelsv_itr != sv_clusters_.end(); ++labelsv_itr)
	{
		Supervoxel<PointT>::Ptr clusters = labelsv_itr->second;
		PointCloudT::Ptr voxels = clusters->voxels_;
		PointCloud<Normal>::Ptr normals = clusters->normals_;
		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		Eigen::Vector3f normal = Eigen::Vector3f::Zero();
		for (int i = 0; i < voxels->size(); i++)
		{
			normal += normals->points[i].getNormalVector3fMap();
			centroid += voxels->points[i].getVector3fMap();
		}
		normal.normalize();
		centroid /= voxels->size();
		Eigen::Vector3f pre_centroid = clusters->centroid_.getVector3fMap();
		cenDist += (centroid - pre_centroid).norm();
		clusters->centroid_.x = centroid[0];
		clusters->centroid_.y = centroid[1];
		clusters->centroid_.z = centroid[2];
		clusters->normal_.normal_x = normal[0];
		clusters->normal_.normal_y = normal[1];
		clusters->normal_.normal_z = normal[2];
	}
	cout << "����ǰ�����Ĳ�ƽ��֮�ͣ�" << cenDist << endl;
}
//�߽��������SV���ľ��룬�����׼���Ը���������
float SVGeneration::distCal(BoundaryData p1, Supervoxel<PointT>::Ptr p2)
{
	float wn = 1;
	float wd = 1.0 / Rvoxel_;//�������Ȩ�ؽϴ��ʹ�����ؿ�Խ�߽�
	Eigen::Vector3f dnormal = p1.normal_ - p2->normal_.getNormalVector3fMap();
	Eigen::Vector3f dxyz = p1.xyz_ - p2->centroid_.getVector3fMap();
	return sqrt(wn*dnormal.squaredNorm() + wd*dxyz.squaredNorm());
}
//ɾ��cloud2�к���cloud1�ĵ�
void SVGeneration::removeOutliers(PointCloudT::Ptr cloud1, PointCloudT::Ptr cloud2, PointCloud<Normal>::Ptr normal)
{
	KdTreeFLANN<PointT> kdtree;
	PointT searchPoint;
	int K = 1;
	vector<int> pointIdxNKNSearch(K);      //�洢��ѯ���������
	vector<float> pointNKNSquaredDistance(K); //�洢���ڵ��Ӧ����ƽ��
	int num = 0;
	//vector<PointT> DeleteData;
	for (auto iter1 = cloud1->begin(); iter1 != cloud1->end(); ++iter1)
	{
		searchPoint.x = iter1->x;
		searchPoint.y = iter1->y;
		searchPoint.z = iter1->z;
		kdtree.setInputCloud(cloud2);
		num = kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance);//��cloud2���ҵ�cloud1�ж�Ӧ�ĵ㣨��Ӧcloud2��index��
		if (num > 0)
		{
			if (sqrt(pointNKNSquaredDistance[0]) < eps)
			{
				auto iter2 = cloud2->begin() + pointIdxNKNSearch[0];
				auto iter3 = normal->begin() + pointIdxNKNSearch[0];
				cloud2->erase(iter2);//delete point xyz_
				normal->erase(iter3);//delete point normal_
				//DeleteData.push_back(searchPoint);
				if (cloud2->size() == 0)
				{
					break;
				}
				//reset
				searchPoint.x = 0;
				searchPoint.y = 0;
				searchPoint.z = 0;
				num = 0;
				pointIdxNKNSearch.clear();
				pointNKNSquaredDistance.clear();
			}
		}
	}
	//cout << DeleteData.size() << endl;
}
//��ȡ��ǩ����
PointCloud<PointXYZL>::Ptr SVGeneration::getLabelCloud()
{
	PointCloud<PointXYZL>::Ptr label_cloud(new PointCloud<PointXYZL>);
	for (map<uint32_t, Supervoxel<PointT>::Ptr >::iterator labelsv_itr = sv_clusters_.begin(); labelsv_itr != sv_clusters_.end(); ++labelsv_itr)
	{
		PointCloudT::Ptr voxels = labelsv_itr->second->voxels_;
		int label = labelsv_itr->first;
		PointCloud<PointXYZL> xyzl_copy;
		copyPointCloud(*voxels, xyzl_copy);
		PointCloud<PointXYZL>::iterator xyzl_copy_itr = xyzl_copy.begin();
		for (; xyzl_copy_itr != xyzl_copy.end(); ++xyzl_copy_itr)
			xyzl_copy_itr->label = label;//tag the label for every point in the voxels
		*label_cloud += xyzl_copy;
	}
	return label_cloud;
}
//NCE(3410)������λ�����ĺ�ƽ�����ľ��룬���������ؽ�����
float SVGeneration::calNCE(supervoxelmap sv)
{
	float sum = 0.0;
	for (map<uint32_t, Supervoxel<PointT>::Ptr>::iterator labelsv_itr = sv_clusters_.begin(); labelsv_itr != sv_clusters_.end(); ++labelsv_itr)
	{
		PointCloudT::Ptr voxels = labelsv_itr->second->voxels_;
		Eigen::Vector3f c = labelsv_itr->second->centroid_.getArray3fMap();
		Eigen::Vector3f median;
		vector<float> x_vec, y_vec, z_vec;
		int num = voxels->size();
		for (int i = 0; i < num; i++){
			x_vec.push_back(voxels->at(i).x);
			y_vec.push_back(voxels->at(i).y);
			z_vec.push_back(voxels->at(i).z);
		}
		sort(x_vec.begin(), x_vec.end());
		sort(y_vec.begin(), y_vec.end());
		sort(z_vec.begin(), z_vec.end());
		if (num % 2 != 0){
			median[0] = x_vec[num / 2];
			median[1] = y_vec[num / 2];
			median[2] = z_vec[num / 2];
		}
		else{
			median[0] = (x_vec[num / 2] + x_vec[num / 2 - 1]) / 2;
			median[1] = (y_vec[num / 2] + y_vec[num / 2 - 1]) / 2;
			median[2] = (z_vec[num / 2] + z_vec[num / 2 - 1]) / 2;
		}

		sum += (c - median).norm();
	}
	return sum/sv.size();
}