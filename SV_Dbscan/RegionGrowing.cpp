#include "RegionGrowing.h"

//���µ�:����+�в�ֵ+����==>ȷ�����ӵ�
void RegionGrowing::getRegions()
{
	vector<pair<float, int>> wis_label;
	for (supervoxelmap::iterator labelsv_itr = sv_clusters_.begin(); labelsv_itr != sv_clusters_.end(); ++labelsv_itr)
	{
		int label = labelsv_itr->first;
		sv_labels_[label] = 0;//��ʼ��
		PointCloud<Normal>::Ptr normals = labelsv_itr->second->normals_;
		PointCloudT::Ptr voxels = labelsv_itr->second->voxels_;
		int num = normals->size();
		float density = 1.0 / num;
		float curvature = 0;
		float residual = 0;
		Eigen::Vector3f pc = labelsv_itr->second->centroid_.getVector3fMap();
		Eigen::Vector3f nc = labelsv_itr->second->normal_.getNormalVector3fMap();
		for (int i = 0; i < num; i++)
		{
			curvature += normals->points[i].curvature;
			Eigen::Vector3f pi = voxels->at(i).getVector3fMap();
			residual += std::pow(nc.dot(pi - pc), 2);
		}
		curvature /= num;
		residual = sqrt(residual / num);
		float k = 0.1;
		float wd = exp(-(density*density) / (2 * k*k));//����
		float wc = exp(-(curvature*curvature) / (2 * k*k));//����
		float wr = exp(-(residual*residual) / (2 * k*k));//�в�ֵ
		float wis = 1 - wd*(wc + wr) / 2;
		label_wis_[label] = wis;
		if (isnan(wis))
		{
			continue;
		}
		wis_label.push_back(pair<float, int>(wis, label));
	}
	sort(wis_label.begin(), wis_label.end());//���򣺸���wiѡȡ���ӣ����ʺͲв�ֵ������С�����������ܶ�

	int num_of_sv = sv_clusters_.size();
	int seed_counter = 0;//get next seed
	int seed = wis_label[seed_counter].second;//minmum rediual+1/num -> label
	int segmented_num = 0;//�Ѿ��ָ��sv��
	int segment_label = 1;//�ָ��ǩ
	while (segmented_num < num_of_sv)
	{
		int sv_in_segment;
		sv_in_segment = svGrowing(seed, segment_label);
		segmented_num += sv_in_segment;
		//num_svs_in_segment_.push_back(sv_in_segment);
		segment_label++;
		//find the next seed that is not segmented
		for (int i_seed = seed_counter + 1; i_seed < wis_label.size(); i_seed++)
		{
			int index = wis_label[i_seed].second;
			if (sv_labels_[index] == 0)
			{
				seed = index;
				seed_counter = i_seed;
				break;
			}
		}
	}

	//����ǩһ����sv�ϲ�
	for (map<int, int>::iterator itr = sv_labels_.begin(); itr != sv_labels_.end(); ++itr)
	{
		int label = itr->second;//�����ǩ
		int svlabel = itr->first;//�����ر�ǩ
		if (label == 0)
			continue;
		regions_[label].push_back(sv_clusters_[svlabel]);//�ڲ�������ʵ��洢
		seglabel_to_svlist_[label].push_back(svlabel);//�ڲ������ر�ǩ�洢	
	}
}
//�������ӵ�����
int RegionGrowing::svGrowing(int seed, int segment_number)
{
	queue<int> seeds;
	seeds.push(seed);
	sv_labels_[seed] = segment_number;
	int sv_in_segment = 1;
	pair<int_multimap, int_multimap> seed_nei_labels = nei_labels_.equal_range(seed);
	if (seed_nei_labels.first == seed_nei_labels.second)//û������ֱ�ӷ���
	{
		return sv_in_segment;
	}
	while (!seeds.empty())
	{
		int curr_seed;
		curr_seed = seeds.front();//ȡ������
		seeds.pop();
		//������ǰ��������
		pair<int_multimap, int_multimap> seed_nei_labels = nei_labels_.equal_range(curr_seed);
		auto iter = seed_nei_labels.first;
		while (iter != seed_nei_labels.second)
		{
			int nei_label = iter->second;
			if (sv_labels_[nei_label] != 0)//�ѱ���ǩ����������ѭ��
			{
				iter++;
				continue;
			}
			bool is_a_seed = false;
			bool in_segment = isValidated(seed, nei_label, is_a_seed);
			if (!in_segment)
			{
				iter++;
				continue;
			}
			sv_labels_[nei_label] = segment_number;
			sv_in_segment++;
			if (is_a_seed)
			{
				seeds.push(nei_label);
			}
			iter++;
		}//next neighbor
	}//next seed
	return sv_in_segment;
}
//��������׼��:smoothness+continuity�����޸�
bool RegionGrowing::isValidated(int seed, int neighbor, bool &is_a_seed)
{
	is_a_seed = true;
	float cosine_threshold = cos(25.0f / 180.0f * static_cast<float> (M_PI));
	Eigen::Vector3f seed_point = sv_clusters_[seed]->centroid_.getVector3fMap();
	Eigen::Vector3f n_seed = sv_clusters_[seed]->normal_.getNormalVector3fMap();
	Eigen::Vector3f n_nei = sv_clusters_[neighbor]->normal_.getNormalVector3fMap();
	Eigen::Vector3f nei_point = sv_clusters_[neighbor]->centroid_.getVector3fMap();
	//����н�,smothness_check and use residual check again pass
	float Dn = abs(n_seed.dot(n_nei));
	//smoothness+distance
	float parallel_th = cos(5.0f / 180.0f * static_cast<float> (M_PI));
	//
	if ((Dn > parallel_th && abs((seed_point - nei_point).dot(n_seed)) < 0.05) || Dn >cosine_threshold)
		return true;//< 25��
	//check curture for puching the point into the seed quece
	if (label_wis_[neighbor] > 0.01)//0.01
		is_a_seed = false;
	return false;
}
//���ڰ�͹�Ժϲ�facets
void RegionGrowing::mergingConvex(){
	//���µ㣺�Ӵ�����ʼ���ұ߽糬���أ���ֹС����Ĵ����ں�
	//����С����ʼ���ҿ��Լ��ٴ���
	vector<pair<int, vector<int> > > seglabel_svnums;//�����ǩ->�ڲ������ر�ǩ����������
	for (map<int, vector<int> >::iterator itr = seglabel_to_svlist_.begin(); itr != seglabel_to_svlist_.end(); ++itr)
	{
		int label = itr->first;
		seglabel_svnums.push_back(make_pair(label, itr->second));
		region_object_[label] = label;
	}
	sort(seglabel_svnums.begin(), seglabel_svnums.end(), cmp);

	map<int, int> svp;//�洢�ڽ����ܳ����ض�
	map<int, int> convex_svp;//�洢�ڽ���͹�����ض�
	
	for (vector<pair<int, vector<int> > >::iterator itr = seglabel_svnums.begin(); itr != seglabel_svnums.end(); ++itr)
	{
		int region_label = itr->first;//�����ǩ f1
		int object_label = region_object_[region_label];//�����ǩ��Ӧ�������ǩ 1
		vector<int> svlist = itr->second;//�ڲ������ر�ǩ���� 
		for (int i = 0; i < svlist.size(); i++)//ѭ����ǰregion�ڲ�sv
		{
			int svlabel = svlist[i];
			pair<int_multimap, int_multimap> nei_sv_labels = nei_labels_.equal_range(svlabel);//ȷ����ǰsv������
			for (auto iter = nei_sv_labels.first; iter != nei_sv_labels.second; ++iter)//��������
			{
				int cur_sv_label = iter->second;
				int seg_label = sv_labels_[cur_sv_label];//��ȡ����label��Ӧ��region label 2,3
				int obj = region_object_[seg_label];
				if (obj != object_label)//�жϱ�ǩ�Ƿ�һ�£����Ƿ�����
				{
					svp[seg_label]++;
					if (isConvex1(svlabel, cur_sv_label))//�ж��Ƿ�Ϊ͹
						convex_svp[seg_label]++;
				}
			}
		}
		//�ϲ�͹����ͨ���жϱ�ֵ��͹�����ض�/�ܳ����ض�
		for (map<int, int>::iterator svp_itr = svp.begin(); svp_itr != svp.end(); ++svp_itr)
		{
			int label = svp_itr->first;
			region_neis_[region_label].insert(label);
			int n1 = svp_itr->second;
			int n2 = convex_svp[label];
			//ȥ���������ӣ�����һ�������ض����ӣ�������Ӧ��������
			/*if (n2 == 1)
				continue;
				*/
			if (double(n2) / n1 >= 0.5 && crossVal(label, region_label))
			{	
				//�ı�ǩ��ɾ������
				vector<Supervoxel<PointT>::Ptr> nei_svs = regions_[label];//�����ڲ�������ʵ��
				vector<int> nei_sv_labels = seglabel_to_svlist_[label];//�����ڲ������ر�ǩ
				for (int k = 0; k < nei_svs.size(); k++)
				{
					regions_[object_label].push_back(nei_svs[k]);//�ϲ�����͹���ڽ����ڵĳ�����ʵ����뵱ǰ����
					seglabel_to_svlist_[object_label].push_back(nei_sv_labels[k]);//�ϲ�����͹���ڽ����ڵĳ����ر�ǩ���뵱ǰ����
					//���ѱ��ϲ��ĳ����أ��޸����Ӧ�������ǩ
					//sv_labels_[nei_sv_labels[k]] = region_label;
					region_object_[label] = object_label;//1

				}
				//ɾ�����ںϵ���Ƭ
				regions_.erase(label);//��������ȡobject��ǩ����
				seglabel_to_svlist_.erase(label);			
			}
		}
		svp.clear();
		convex_svp.clear();	
	}
}
//������֤
bool RegionGrowing::crossVal(int label_i, int label_j){
	int svp=0;//�洢�ڽ����ܳ����ض�
	int convex_svp=0;//�洢�ڽ���͹�����ض�
	vector<int> svlist = seglabel_to_svlist_[label_i];
	for (int i = 0; i < svlist.size(); i++)//ѭ����ǰregion�ڲ�sv
	{
		int svlabel = svlist[i];
		pair<int_multimap, int_multimap> nei_sv_labels = nei_labels_.equal_range(svlabel);//ȷ����ǰsv������
		for (auto iter = nei_sv_labels.first; iter != nei_sv_labels.second; ++iter)//��������
		{
			int cur_sv_label = iter->second;
			int seg_label = sv_labels_[cur_sv_label];//��ȡ��������label��Ӧ��region label
			if (seg_label == label_j)//�жϱ�ǩ�Ƿ�һ�£����Ƿ�����
			{
				svp++;
				if (isConvex1(svlabel, cur_sv_label))//�ж��Ƿ�Ϊ͹
					convex_svp++;
			}
		}
	}
	//͹�����ӣ��жϱ�ֵ��͹�����ض�/�ܳ����ض�
	/*if (svp == 1)
		return false;*/
	if (double(convex_svp) / svp >= 0.5 )
		return true;

	return false;
}
//�жϰ�͹�ԣ�input���ڽӳ����ر�ǩ�ԣ�out��true/false
bool RegionGrowing::isConvex1(int s_label, int t_label){
	Eigen::Vector3f s_point = sv_clusters_[s_label]->centroid_.getArray3fMap();
	Eigen::Vector3f t_point = sv_clusters_[t_label]->centroid_.getArray3fMap();
	Eigen::Vector3f vec_s_to_t, vec_t_to_s;
	vec_s_to_t = t_point - s_point;
	vec_t_to_s = -vec_s_to_t;
	Eigen::Vector3f s_normal = sv_clusters_[s_label]->normal_.getNormalVector3fMap();
	Eigen::Vector3f t_normal = sv_clusters_[t_label]->normal_.getNormalVector3fMap();
	Eigen::Vector3f ncross;
	ncross = s_normal.cross(t_normal);
	float normal_angle = getAngle3D(s_normal, t_normal, true);
	//Sanity Criterion
	float intersection_angle = getAngle3D(ncross, vec_t_to_s, true);
	float min_intersect_angle = (intersection_angle < 90.) ? intersection_angle : 180. - intersection_angle;
	float intersect_thresh = 60. * 1. / (1. + exp(-0.25 * (normal_angle - 25.)));
	//Convexity Criterion
	float angle = getAngle3D(vec_t_to_s, s_normal) - getAngle3D(vec_t_to_s, t_normal);
	/***Only use CC***/
	if (angle <= 0 || normal_angle<10)
		return true;
	/***use SC&CC***/
	/*if (min_intersect_angle>intersect_thresh && (angle <= 0 || normal_angle<10))
		return true;*/
	return false;
}
//�жϰ�͹�ԣ����������нǣ�����3461��Ч�����ã���ֵ���õ�
bool RegionGrowing::isConvex2(int s_label, int t_label){
	Eigen::Vector3f s_point = sv_clusters_[s_label]->centroid_.getArray3fMap();
	Eigen::Vector3f t_point = sv_clusters_[t_label]->centroid_.getArray3fMap();
	Eigen::Vector3f s_normal = sv_clusters_[s_label]->normal_.getNormalVector3fMap();
	Eigen::Vector3f t_normal = sv_clusters_[t_label]->normal_.getNormalVector3fMap();
	float theta = getAngle3D(s_normal, t_normal, true);
	theta = theta < 90 ? theta : 180 - theta;
	Eigen::Vector3f vec_s_to_t, vec_t_to_s;
	vec_s_to_t = t_point - s_point;
	vec_t_to_s = -vec_s_to_t;
	float beta_s = getAngle3D(s_normal, vec_s_to_t, true);
	beta_s = beta_s < 90 ? beta_s : 180 - beta_s;
	float alpha_s = 90 - beta_s;
	float beta_t = getAngle3D(t_normal, vec_t_to_s, true);
	beta_t = beta_t < 90 ? beta_t : 180 - beta_t;
	float alpha_t = 90 - beta_t;
	float alpha = (alpha_s + alpha_t) / 2;
	float min = theta < alpha ? theta : alpha;
	if (min < 30)
		return true;
	return false;
}
//ϸ������
void RegionGrowing::mergeSmallRegions(){
	for (map<int, set<int>>::iterator itr = region_neis_.begin(); itr != region_neis_.end(); itr++)
	{
		int label = itr->first;
		set<int> adj_labels = itr->second;
		int max_size = seglabel_to_svlist_[label].size();
		int max_seglabel = label;
		if (seglabel_to_svlist_[label].size() < 3)
		{
			
			for (set<int>::iterator set_itr = adj_labels.begin(); set_itr != adj_labels.end(); ++set_itr)
			{
				int tmp = seglabel_to_svlist_[*set_itr].size();
				if (tmp > max_size)
				{
					if (seglabel_to_svlist_[label].size() == 1){//��������������Χ�������ϲ�
						max_size = tmp;
						max_seglabel = *set_itr;}
					else{//������������ɵ�С�棬������ڽ����壬���ų�����
						if (tmp > 50)
							continue;
						max_size = tmp;
						max_seglabel = *set_itr;}				
				}				
			}
			if (max_seglabel != label)
			{
				vector<Supervoxel<PointT>::Ptr> nei_svs = regions_[label];
				vector<int> nei_sv_labels = seglabel_to_svlist_[label];
				for (int k = 0; k < nei_svs.size(); k++)
				{
					regions_[max_seglabel].push_back(nei_svs[k]);
					seglabel_to_svlist_[max_seglabel].push_back(nei_sv_labels[k]);
				}
				regions_.erase(label);
				seglabel_to_svlist_.erase(label);
			}
		}
	}
}
//��ȡ����ǩ���������
PointCloud<PointXYZL>::Ptr RegionGrowing::getPatchCloud()
{
	PointCloud<PointXYZL>::Ptr segment_cloud(new PointCloud<PointXYZL>);
	for (map<int, vector<Supervoxel<PointT>::Ptr>>::iterator itr = regions_.begin(); itr != regions_.end(); ++itr)
	{
		int label = itr->first;
		vector<Supervoxel<PointT>::Ptr> svs = itr->second;
		PointCloud<PointXYZL> xyzl_copy;
		if (svs.size() == 1)
			continue;
		for (int i = 0; i < svs.size(); i++)
		{
			PointCloud<PointXYZL> temp;
			copyPointCloud(*svs[i]->voxels_, temp);
			xyzl_copy += temp;
		}
		for (PointCloud<PointXYZL>::iterator xyzl_copy_itr = xyzl_copy.begin(); xyzl_copy_itr != xyzl_copy.end(); ++xyzl_copy_itr)
			xyzl_copy_itr->label = label;//tag the label for every point in the voxels
		*segment_cloud += xyzl_copy;
	}
	return segment_cloud;
}
