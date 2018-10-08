#ifndef _stereo_mapper_h
#define _stereo_mapper_h

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/StdVector>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"

#include <ros/console.h>

#include "parameters.h"
#include "calc_cost.h"
#include "sgm.h"
#include "depth_fusion.h"
#include "filter_cost.h"
#include "cuda_utils.h"
#include "convert.h"

class StereoMapper
{
public:
    StereoMapper();
    void initIntrinsic(const cv::Mat &_K1, float bf, float dep_sample );
    void initReference(const cv::Mat &_img_l);

    void update(const cv::Mat &_img_r, const cv::Mat &R_l, const cv::Mat &T_l, const cv::Mat &R_r, const cv::Mat &T_r);

    void epipolar(double x, double y, double z);

    void output(cv::Mat &result);
    void outputFusion(cv::Mat &result, Eigen::Matrix3d R, Eigen::Vector3d t, float fx, float fy, float cx, float cy);
    void outputFusionCPU(cv::Mat &result, Eigen::Matrix3d R, Eigen::Vector3d t, float fx, float fy, float cx, float cy);
    void depthFuseCPU(float fx, float fy, float cx, float cy,
                      Eigen::Matrix3d R_p_2_c, Eigen::Vector3d t_p_2_c, int flag,
                      cv::Mat& curDepth);

    cv::cuda::GpuMat img_l, img_r, img_warp, img_diff;
    cv::cuda::GpuMat raw_cost, sgm_cost, measurement_cnt;
    cv::cuda::GpuMat dep;
    cv::cuda::GpuMat propogate_table ;
    cv::cuda::GpuMat alpha0, beta0, mu0, sigma0 ;
    cv::cuda::GpuMat alpha1, beta1, mu1, sigma1 ;
    cv::cuda::GpuMat tmpDepthMap ;
    cv::cuda::GpuMat tmpMap, debugMap ;
    cv::cuda::GpuMat fuseDepth ;

    int pre_index, cur_index ;
    bool firstDepth;
    cv::Mat nK1;
    cv::Mat img_intensity;  // ref_img
    cv::Mat R, T;
    Eigen::Matrix3d R_b_2_w ;
    Eigen::Vector3d t_b_2_w ;

    float DEP_SAMPLE ;
};

#endif
