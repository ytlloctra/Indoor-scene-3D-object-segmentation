# Indoor scene 3D object segmentation

## Methods

### Pre-Processing

In order to obtain object level segmentation, the background in indoor scenes needs to be removed as well as some common filtering processes. The RANSAC algorithm is a powerful model fitting algorithm that is heavily used in the point cloud data pre-processing step.  

![Fig 2](https://image-1312312327.cos.ap-shanghai.myqcloud.com/Fig%202.png)

### DBSCAN Cluster Based on Supervoxel

In the first stage of clustering, we use the improved VCCS algorithm to locally cluster geometrically similar voxels into supervoxels.  

After hypervoxels are generated, different hypervoxels are fused into larger clustering objects based on density characteristics.

![Fig 4](https://image-1312312327.cos.ap-shanghai.myqcloud.com/Fig%204.png)

### Guided boundary enhancement

After the segmentation and clustering of indoor scenes are completed, it is usually necessary to refine and denoise the segmented edges.

![Fig 5](https://image-1312312327.cos.ap-shanghai.myqcloud.com/Fig%205.png)

## Result

The algorithm is written in C++ and runs on Intel i7-11700K CPU@3.6GHz with 32GB RAM. To compare the performance of the proposed algorithm, we compared it with two other approaches for point cloud segmentation.   

![Fig 7](https://image-1312312327.cos.ap-shanghai.myqcloud.com/Fig%207.png)