#include<chrono>
#include<mutex>
#include<ros/ros.h>
#include<rosbag/bag.h>
#include<rosbag/chunked_file.h>
#include<rosbag/view.h>
#include<rosbag/query.h>
#include <stdio.h>
#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"
#include <boost/filesystem.hpp>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Image.h"
#include "src/tools/transport_util.h"
#include "geometry_msgs/PoseStamped.h"
#include "motion_stereo/stereo_mapper.h"
#include "src/tools/tic_toc.h"

const int downSampleTimes = 0 ;

using namespace std;
const int aggNum = 3 ;
const int interval = 6 ;
const int candidateNum = aggNum*interval + 10 ;
float maxDepth = 20.0 ;
float minDepth = 0.2 ;

//TUM
//float fx = 525.0 ;
//float fy = 525.0 ;
//float cx = 319.5 ;
//float cy = 239.5 ;

//ICL-NUIM
float fx = 481.2017 ;
float fy = -480.0002 ;
float cx = 319.5 ;
float cy = 239.5 ;



ros::Publisher pub_depth ;

class ImageMeasurement
{
  public:
    ros::Time t;
    cv::Mat   image;

    ImageMeasurement(const ros::Time& _t, const cv::Mat& _image)
    {
      t     = _t;
      image = _image.clone();
    }

    ImageMeasurement(const ImageMeasurement& i)
    {
      t     = i.t;
      image = i.image.clone();
    }

    ~ImageMeasurement() { ;}
};

struct PoseElement
{
    ros::Time t;
    double tx,ty,tz;
    double qx,qy,qz,qw;
};

std::list<ImageMeasurement> imageBuf;
std::mutex mMutexImg;
std::list<PoseElement> poseBuf;
std::mutex mMutexPose;
StereoMapper motion_stereo_mapper;


void GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat img ;
    img = matFromImage(*msg);

    //puts("recieved img") ;

    unique_lock<mutex> lock(mMutexImg) ;
    ros::Time tImage = msg->header.stamp;
    imageBuf.push_front(ImageMeasurement(tImage, img));
}

void GrabPose(const geometry_msgs::PoseStamped& pose)
{
    PoseElement tt ;
    tt.qw = pose.pose.orientation.w ;
    tt.qx = pose.pose.orientation.x ;
    tt.qy = pose.pose.orientation.y ;
    tt.qz = pose.pose.orientation.z ;
    tt.tx = pose.pose.position.x;
    tt.ty = pose.pose.position.y;
    tt.tz = pose.pose.position.z;
    tt.t = pose.header.stamp;

    unique_lock<mutex> lock(mMutexPose) ;
    poseBuf.push_front(tt);
}


int lastSize;

struct meshingFrame
{
    cv::Mat img ;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

static std::vector<meshingFrame> toAggNonKFs ;

void depthUp(cv::Mat& in, cv::Mat& out)
{
    cv::Mat tmp = cv::Mat(in.rows*2, in.cols*2, in.type() ) ;
    for( int i = 0 ; i < in.rows; i++ )
    {
        for( int j = 0; j < in.cols; j++ )
        {
            tmp.at<float>((i<<1), (j<<1)) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1)+1, (j<<1)) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1), (j<<1)+1) = in.at<float>(i, j) ;
            tmp.at<float>((i<<1)+1, (j<<1)+1) = in.at<float>(i, j) ;
        }
    }
    out = tmp.clone() ;
}

void processImg()
{
    std::list<ImageMeasurement>::iterator iterImg, iterImg_pre ;
    std::list<PoseElement>::iterator iterPose, iterPose_pre ;
    cv::Mat curImg, img ;
    Eigen::Quaterniond q_R ;
    Eigen::Matrix3d cur_R ;
    Eigen::Vector3d cur_t ;
    cv::Mat cv_R_l, cv_T_l;
    Eigen::Matrix3d R ;
    Eigen::Vector3d t ;
    cv::Mat cv_R_r, cv_T_r;
    bool findPose ;
    bool flag = false ;
    std_msgs::Header head ;
    head.frame_id = "world" ;

    {
        unique_lock<mutex> lock_0(mMutexImg) ;
        unique_lock<mutex> lock_1(mMutexPose) ;
        if ( imageBuf.size() <= lastSize )
        {
            return ;
        }
        if ( flag == false ){
            iterImg = imageBuf.begin() ;
            iterPose = poseBuf.begin() ;
        }
        else{
            iterImg = imageBuf.begin() ;
            iterPose = poseBuf.begin() ;
//            iterImg  = iterImg_pre ;
//            iterPose = iterPose_pre ;
        }

        if ( iterPose->t < iterImg->t ){
            return ;
        }

        curImg = iterImg->image.clone() ;
        head.stamp = iterImg->t ;
        for( int i = 0 ; i < downSampleTimes; i++ ){
            cv::pyrDown(curImg, curImg, cv::Size(curImg.cols/2, curImg.rows/2) ) ;
        }

        findPose = false ;
        while ( iterPose != poseBuf.end() )
        {
            if ( iterPose->t != iterImg->t ){
                iterPose++ ;
            }
            else
            {
                q_R.x() = iterPose->qx ;
                q_R.y() = iterPose->qy ;
                q_R.z() = iterPose->qz ;
                q_R.w() = iterPose->qw ;
                cur_R = q_R.toRotationMatrix();
                cur_t << iterPose->tx, iterPose->ty, iterPose->tz ;

                findPose = true ;
                break ;
            }
        }
        if ( findPose == false )
        {
            ROS_WARN("can not find pose for the current Img") ;
            return ;
        }

        iterImg_pre = iterImg ;
        iterPose_pre = iterPose ;
        flag = true ;


        for( int i = 0 ; i < aggNum; i++ )
        {
            iterImg = std::next(iterImg, interval ) ;
            iterPose = std::next(iterPose, interval ) ;

            q_R.x() = iterPose->qx ;
            q_R.y() = iterPose->qy ;
            q_R.z() = iterPose->qz ;
            q_R.w() = iterPose->qw ;
            R = q_R.toRotationMatrix();
            t << iterPose->tx, iterPose->ty, iterPose->tz ;

            toAggNonKFs[i].img = iterImg->image.clone() ;
            for( int j = 0 ; j < downSampleTimes; j++ )
            {
                cv::pyrDown(toAggNonKFs[i].img, toAggNonKFs[i].img,
                            cv::Size(toAggNonKFs[i].img.cols/2, toAggNonKFs[i].img.rows/2) ) ;
            }
            //cv::imshow(std::to_string(i), toAggNonKFs[i].img ) ;

            toAggNonKFs[i].R = R ;
            toAggNonKFs[i].t = t ;
        }
        lastSize++ ;

        imageBuf.pop_back();
        poseBuf.pop_back();
    }
    cv::imshow("curImg", curImg ) ;
//    cv::waitKey(0) ;

    TicToc tc_sgm ;

    cv::eigen2cv(cur_R, cv_R_l);
    cv::eigen2cv(cur_t, cv_T_l);
    motion_stereo_mapper.initReference(curImg);

    for( int i = 0 ; i < aggNum; i++ )
    {
        img = toAggNonKFs[i].img.clone() ;
        R = toAggNonKFs[i].R ;
        t = toAggNonKFs[i].t ;
        cv::eigen2cv(R, cv_R_r);
        cv::eigen2cv(t, cv_T_r);
        motion_stereo_mapper.update(img, cv_R_l, cv_T_l, cv_R_r, cv_T_r);
    }

    cv::Mat curDepth ;
    motion_stereo_mapper.outputFusionCPU(curDepth, cur_R, cur_t, fx, fy, cx, cy) ;

    for( int j = 0 ; j < downSampleTimes; j++ )
    {
        depthUp(curDepth, curDepth) ;
        //cv::pyrUp(curDepth, curDepth, cv::Size(curDepth.cols*2, curDepth.rows*2) ) ;
    }

    static double sum_time = 0 ;
    static int sum_cnt = 0 ;
    sum_time += tc_sgm.toc() ;
    sum_cnt++ ;
    ROS_WARN("AVERAE CAL TIME %lf", sum_time/sum_cnt );

    sensor_msgs::Image msg_img ;
    toImageMsg(msg_img, curDepth, head, "32FC1") ;
    pub_depth.publish(msg_img) ;

    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < minDepth ) {
                curDepth.at<float>(i, j) = 0 ;
            }
            if ( curDepth.at<float>(i, j) > maxDepth ){
                curDepth.at<float>(i, j) = 0 ;
            }
        }
    }

    static cv::Mat color_disp, disp_depth;
    //disp_depth = curDepth/maxDepth*255;
    //disp_depth.convertTo(disp_depth, CV_8U);
    cv::normalize(curDepth, disp_depth, 0, 255, CV_MINMAX, CV_8U);
    cv::applyColorMap(disp_depth, color_disp, cv::COLORMAP_RAINBOW);
    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < 0.001 ) {
                color_disp.at<cv::Vec3b>(i, j)[0] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[1] = 0 ;
                color_disp.at<cv::Vec3b>(i, j)[2] = 0 ;
            }
        }
    }
    cv::imshow("Current depth", color_disp ) ;
    cv::waitKey(1) ;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapping");
    if ( argc < 2 ){
        CASE = 0 ;
    }
    else {
        sscanf(argv[1], "%d", &CASE ) ;
    }
    ros::start();
    ros::NodeHandle nh("~") ;

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    string packagePath = ros::package::getPath("mapping");
    ros::Subscriber sub_img = nh.subscribe("/image", 100, &GrabImage ) ;
    ros::Subscriber sub_pose = nh.subscribe("/cur_pose", 100, &GrabPose ) ;
    pub_depth = nh.advertise<sensor_msgs::Image>("/depth", 1000 );

    ROS_WARN("CASE=%d", CASE ) ;
    for( int i = 0 ; i < downSampleTimes; i++ ){
        fx /= 2 ;
        fy /= 2 ;
        cx = (cx+0.5)/2.0 - 0.5;
        cy = (cy+0.5)/2.0 - 0.5;
    }

    cv::Mat K(3, 3, CV_64F) ;
    K.setTo(0.0) ;
    K.at<double>(0, 0) = fx ;
    K.at<double>(1, 1) = fy ;
    K.at<double>(0, 2) = cx ;
    K.at<double>(1, 2) = cy ;
    K.at<double>(2, 2) = 1.0 ;
    float bf = 0.02*fx ;
    float dep_sample = 1.0f / (0.15 * 160.0);
    motion_stereo_mapper.initIntrinsic( K, bf, dep_sample );

    lastSize = aggNum*interval ;
    toAggNonKFs.clear();
    toAggNonKFs.resize(aggNum+5);

    ros::Rate r(1000) ;
    while( ros::ok() )
    {
        processImg();
        ros::spinOnce();
        r.sleep();
    }

    ros::shutdown();

    return 0;
}
