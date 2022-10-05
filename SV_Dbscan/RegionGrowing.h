#pragma once
#ifndef REGIONGROWING_H
#define REGIONGROWING_H
#include "SVGeneration.h"


class RegionGrowing
{
public:
	RegionGrowing(supervoxelmap sv_clusters, multimap<uint32_t, uint32_t> nei_labels)
		: sv_clusters_(sv_clusters), nei_labels_(nei_labels)
	{}
	~RegionGrowing()
	{}
	/****��������***/
	void getRegions();
	int svGrowing(int seed, int segment_number);
	bool isValidated(int seed, int neighbor, bool &is_a_seed);
	PointCloud<PointXYZL>::Ptr getPatchCloud();
	/***����͹�Ժϲ�***/
	void mergingConvex();
	bool crossVal(int label_i, int label_j);//������֤
	//�жϰ�͹��
	bool isConvex1(int s_label, int t_label);
	bool isConvex2(int s_label, int t_label);
	//refinement
	void mergeSmallRegions();
	//��̬����������map��value����
	static bool cmp(const pair<int, vector<int> > &r1, const pair<int, vector<int> > &r2){
		return r1.second.size() > r2.second.size();
	}

private:
	supervoxelmap sv_clusters_;//�߽�ϸ�������ؼ�����ǩ->ʵ��
	multimap<uint32_t, uint32_t> nei_labels_;//�������ڽӹ�ϵ����ǩ��һ�Զ�

	map<int, int> sv_labels_;//�����ر�ǩ->�����ǩ
	map<int, float> label_wis_;//���ڴ洢ÿ�������ص�wi

	map<int, vector<Supervoxel<PointT>::Ptr>> regions_;//�����ǩ->�ڲ�������ʵ�壬���ڴ�Ž����������ʾ
	map<int, vector<int> > seglabel_to_svlist_;//�����ǩ->�ڲ������ر�ǩ
	map<int, int> region_object_;

	map<int, set<int>> region_neis_;//�����ǩ->�ڽ������ǩ��
};

#endif REGIONGROWING_H