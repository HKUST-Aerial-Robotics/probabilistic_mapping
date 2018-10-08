#include "stereo_mapper.h"
#include <iostream>

using namespace std;

static float alpha[2][HEIGHT][WIDTH];
static float beta[2][HEIGHT][WIDTH];
static float mu[2][HEIGHT][WIDTH];
static float sigma[2][HEIGHT][WIDTH];
static unsigned int p_table[2][HEIGHT][WIDTH] ;
static bool vst[HEIGHT][WIDTH] ;
static cv::Mat previousMap, previousUncertainty, previousInlier;
static cv::Mat predictMapNoFill, predictMapNoFillUncertainty, predictMapNoFillInlier ;
static cv::Mat predictMap, predictMapUncertainty, predictMapInlier ;
static cv::Mat uncertaintyMap, inlierMap ;
static bool debugInfo = false ;

inline float expectation( float a, float b){
    if ( a < FLOAT_EPS ){
        return 0.0 ;
    }
    else {
        return a/(a+b) ;
    }
}

StereoMapper::StereoMapper()
    : raw_cost(1, HEIGHT * WIDTH * DEP_CNT, CV_32F),
      sgm_cost(1, HEIGHT * WIDTH * DEP_CNT, CV_32F),
      measurement_cnt(1, HEIGHT * WIDTH * DEP_CNT, CV_32F),
      propogate_table(1, HEIGHT * WIDTH, CV_32S),
      tmpDepthMap(1, HEIGHT * WIDTH, CV_32F),
      tmpMap(1, HEIGHT * WIDTH, CV_32F),
      dep(HEIGHT, WIDTH, CV_32F),
      fuseDepth(HEIGHT, WIDTH, CV_32F),
      debugMap(HEIGHT, WIDTH, CV_32F),
      alpha0(1, HEIGHT * WIDTH, CV_32F),
      beta0(1, HEIGHT * WIDTH, CV_32F),
      mu0(1, HEIGHT * WIDTH, CV_32F),
      sigma0(1, HEIGHT * WIDTH, CV_32F),
      alpha1(1, HEIGHT * WIDTH, CV_32F),
      beta1(1, HEIGHT * WIDTH, CV_32F),
      mu1(1, HEIGHT * WIDTH, CV_32F),
      sigma1(1, HEIGHT * WIDTH, CV_32F)
{
    pre_index = 0 ;
    cur_index = 1 ;
    firstDepth = true ;
}

void StereoMapper::initIntrinsic(const cv::Mat &_K1, float bf, float dep_sample)
{
    nK1 = _K1.clone();
    DEP_SAMPLE = dep_sample;

    puts("Enumerate Depth:") ;
    for( int i = 1 ; i < DEP_CNT; i++ ){
        printf("%f ", 1.0/(i*DEP_SAMPLE) ) ;
    }
    printf("\n") ;
}

void StereoMapper::initReference(const cv::Mat &_img_l)
{
    cv::Mat tmp_img;
    _img_l.convertTo(tmp_img, CV_32F);
    img_l.upload(tmp_img);

    raw_cost.setTo(cv::Scalar_<float>(-1.0));
    measurement_cnt.setTo(cv::Scalar_<float>(0.0)) ;
}

void StereoMapper::update(const cv::Mat &_img_r, const cv::Mat &R_l, const cv::Mat &T_l, const cv::Mat &R_r, const cv::Mat &T_r)
{
    cv::Mat tmp_img;
    _img_r.convertTo(tmp_img, CV_32F);
    img_r.upload(tmp_img);

    R = nK1 * R_r.t() * R_l * nK1.inv();  // H1
    T = nK1 * R_r.t() * (T_l - T_r);  // H2

    ad_calc_cost(
                measurement_cnt.data,
                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                T.at<double>(0, 0), T.at<double>(0, 1), T.at<double>(0, 2),
                img_l.data, img_l.step,
                img_r.data, img_r.step,  // img_l.step = img_r.step = 1536
                raw_cost.data, DEP_SAMPLE);
}

void StereoMapper::output(cv::Mat& result)
{
    //    int64 t = cv::getTickCount();
    //    ROS_WARN("P1:%.0lf, P2:%.0lf, tau_so:%.0lf, q1:%.0lf, q2:%.0lf",
    //    pi1, pi2, tau_so, sgm_q1, sgm_q2);


    sgm_cost.setTo(cv::Scalar_<float>(0.0));
    sgm2(raw_cost.data,
         sgm_cost.data);


    filter_cost(sgm_cost.data,
                dep.data, dep.step, DEP_SAMPLE);

    dep.download(result);
}

void StereoMapper::outputFusion(cv::Mat& result, Eigen::Matrix3d R, Eigen::Vector3d t,
                                float fx, float fy, float cx, float cy)
{
    //    int64 t = cv::getTickCount();
    //    ROS_WARN("P1:%.0lf, P2:%.0lf, tau_so:%.0lf, q1:%.0lf, q2:%.0lf",
    //    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm_cost.setTo(cv::Scalar_<float>(0.0));
    sgm2(raw_cost.data,
         sgm_cost.data);

    //    int64 t2 = cv::getTickCount();
    //    printf("PW CUDA Time part 3 (SGM): %fms\n", (t2 - t) * 1000 / cv::getTickFrequency());

    filter_cost(sgm_cost.data,
                dep.data, dep.step, DEP_SAMPLE);

    if ( firstDepth ){
        firstDepth = false ;
        //puts("aaa");
        //printf("img_l.rows=%d img_l.cols=%d\n", img_l.rows, img_l.cols ) ;
        result = cv::Mat(img_l.rows, img_l.cols, CV_32F ) ;
        //printf("type=%d\n", result.type() ) ;
        result.setTo(0.0) ;

        R_b_2_w = R ;
        t_b_2_w = t ;
        pre_index = 0 ;
        cur_index = 1 ;

        //puts("abbbb");
        alpha0.setTo(cv::Scalar_<float>(0.0) ) ;
        alpha1.setTo(cv::Scalar_<float>(0.0) ) ;

        //puts("abb");
    }
    else
    {
        convert(dep.data,
                tmpDepthMap.data,
                false,
                dep.step ) ;


        Eigen::Matrix3d R_p_2_c = R.transpose()*R_b_2_w ;
        Eigen::Vector3d t_p_2_c = R.transpose()*(t_b_2_w-t) ;

        //        cout << R_p_2_c << "\n" ;
        //        cout << t_p_2_c.transpose() << "\n" ;

        //printf("fx=%f fy=%f cx=%f cy=%f", fx, fy, cx, cy ) ;

        propogate_table.setTo(cv::Scalar_<int>(0)) ;
        //        alpha0.setTo(cv::Scalar_<float>(0.0) ) ;
        //        alpha1.setTo(cv::Scalar_<float>(0.0) ) ;
        if ( pre_index == 0 )
        {
            depth_fuse(fx, fy, cx, cy,
                       (float)R_p_2_c(0, 0), (float)R_p_2_c(0, 1), (float)R_p_2_c(0, 2),
                       (float)R_p_2_c(1, 0), (float)R_p_2_c(1, 1), (float)R_p_2_c(1, 2),
                       (float)R_p_2_c(2, 0), (float)R_p_2_c(2, 1), (float)R_p_2_c(2, 2),
                       (float)t_p_2_c(0), (float)t_p_2_c(1), (float)t_p_2_c(2),
                       tmpDepthMap.data,
                       propogate_table.data,
                       alpha0.data,
                       beta0.data,
                       mu0.data,
                       sigma0.data,
                       alpha1.data,
                       beta1.data,
                       mu1.data,
                       sigma1.data
                       );

            convert(tmpDepthMap.data,
                    fuseDepth.data,
                    true,
                    fuseDepth.step) ;

            convert(alpha1.data,
                    debugMap.data,
                    true,
                    debugMap.step) ;
        }
        else
        {
            depth_fuse(fx, fy, cx, cy,
                       (float)R_p_2_c(0, 0), (float)R_p_2_c(0, 1), (float)R_p_2_c(0, 2),
                       (float)R_p_2_c(1, 0), (float)R_p_2_c(1, 1), (float)R_p_2_c(1, 2),
                       (float)R_p_2_c(2, 0), (float)R_p_2_c(2, 1), (float)R_p_2_c(2, 2),
                       (float)t_p_2_c(0), (float)t_p_2_c(1), (float)t_p_2_c(2),
                       tmpDepthMap.data,
                       propogate_table.data,
                       alpha1.data,
                       beta1.data,
                       mu1.data,
                       sigma1.data,
                       alpha0.data,
                       beta0.data,
                       mu0.data,
                       sigma0.data
                       );

            convert(tmpDepthMap.data,
                    fuseDepth.data,
                    true,
                    fuseDepth.step) ;

            convert(alpha0.data,
                    debugMap.data,
                    true,
                    debugMap.step) ;
        }



        //printf("a=%f\n", alpha[cur_index](10, 10) ) ;

        //fuseDepth.download(result);
        dep.download(result);

        static cv::Mat debugDisp ;
        debugMap.download(debugDisp);
        cv::imshow("dubug", debugDisp/10 ) ;
        cv::waitKey(3) ;

        R_b_2_w = R ;
        t_b_2_w = t ;

        pre_index = cur_index ;
        cur_index ^= 1 ;
    }

}

void transformMapCPU(float fx, float fy, float cx, float cy,
                     Eigen::Matrix3d R_p_2_c, Eigen::Vector3d t_p_2_c, int pre, int cur)
{
    int cnt0 = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    int cnt3 = 0;
    int cnt4 = 0;

    memset(p_table[cur], 0, sizeof(unsigned int)*WIDTH*HEIGHT ) ;

    //printf("fx=%f fy=%f cx=%f cy=%f\n", fx, fy, cx, cy) ;

    if ( debugInfo ){
        previousMap = cv::Mat(HEIGHT, WIDTH, CV_32F) ;
        previousMap.setTo(0.0) ;

        previousUncertainty = cv::Mat(HEIGHT, WIDTH, CV_32F) ;
        previousUncertainty.setTo(0.0) ;

        previousInlier = cv::Mat(HEIGHT, WIDTH, CV_32F) ;
        previousInlier.setTo(0.0) ;
    }

    for( int y = 0; y < HEIGHT; y++ )
    {
        for( int x = 0 ; x < WIDTH; x++ )
        {
            if ( alpha[pre][y][x] < FLOAT_EPS ){
                cnt0++ ;
                continue ;
            }
            if ( alpha[pre][y][x]/(alpha[pre][y][x]+beta[pre][y][x]) < 0.5 ){
                cnt1++ ;
                continue ;
            }
            float depth = mu[pre][y][x] ;
            if ( debugInfo ){
                previousMap.at<float>(y, x) = mu[pre][y][x] ;
                previousUncertainty.at<float>(y, x) = sqrt(sigma[pre][y][x]) ;
                previousInlier.at<float>(y, x) = expectation(alpha[pre][y][x], beta[pre][y][x]) ;
            }

            Eigen::Vector3d p, p2 ;
            p << (x-cx)/fx*depth, (y-cy)/fy*depth, depth;
            p2 = R_p_2_c*p + t_p_2_c ;
            if ( p2(2) < FLOAT_EPS ){
                cnt2++ ;
                //                if ( cnt2 < 20 ){
                //                    cout << p.transpose() << " " << p2.transpose() << "\n" ;
                //                }
                continue ;
            }
            int x2 = p2(0)/p2(2)*fx + cx + 0.5 ;
            int y2 = p2(1)/p2(2)*fy + cy + 0.5 ;
            if ( x2 < 0 || x2 >= WIDTH || y2 < 0 || y2 >= HEIGHT ){
                cnt3++ ;
                //                if ( cnt3 < 20 ){
                //                    cout << x << " " << y << " " << x2 << " " << y2 << "\n" ;
                //                }
                continue ;
            }
            //alpha[pre][y][x] *= 0.95 ;
            mu[pre][y][x] = p2(2);
            sigma[pre][y][x] += p2(2)*0.05;
            if ( p_table[cur][y2][x2] == 0 ){
                p_table[cur][y2][x2] = (y<<16) | x ;
            }
            else
            {
                int now_x = p_table[cur][y2][x2] & 0xffff;
                int now_y = p_table[cur][y2][x2] >> 16 ;
                if ( mu[pre][now_y][now_x] <= depth ){
                    continue ;
                }
                else {
                    mu[pre][now_y][now_x] = depth ;
                    p_table[cur][y2][x2] = (y<<16) | x ;
                }
            }
            cnt4++ ;
        }
    }
    //ROS_WARN("[transformMapCPU] %d %d %d %d %d", cnt0, cnt1, cnt2, cnt3, cnt4 ) ;
}

void holeFillingCPU(int cur, int expand_sz = 3)
{
    if ( debugInfo )
    {
        predictMapNoFill = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMapNoFill.setTo(0.0) ;

        predictMapNoFillUncertainty = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMapNoFillUncertainty.setTo(0.0) ;

        predictMapNoFillInlier = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMapNoFillInlier.setTo(0.0) ;

        for( int y = 0; y < HEIGHT; y++ )
        {
            for( int x = 0 ; x < WIDTH; x++ )
            {
                if ( p_table[cur][y][x] == 0 ){
                    continue ;
                }
                else
                {
                    int transform_x = p_table[cur][y][x] & 0xffff;
                    int transform_y = p_table[cur][y][x] >> 16 ;
                    predictMapNoFill.at<float>(y, x) = mu[cur^1][transform_y][transform_x] ;

                    predictMapNoFillUncertainty.at<float>(y, x) = sqrt(sigma[cur^1][transform_y][transform_x]);
                    predictMapNoFillInlier.at<float>(y, x) = expectation(alpha[cur^1][transform_y][transform_x], beta[cur^1][transform_y][transform_x]) ;

                }
            }
        }
    }

    memset(vst, 0, sizeof(vst) ) ;
    for( int y = expand_sz; y < HEIGHT-expand_sz; y++ )
    {
        for( int x = expand_sz ; x < WIDTH-expand_sz; x++ )
        {
            if ( vst[y][x] ){
                continue ;
            }
            if ( p_table[cur][y][x] == 0 ){
                continue ;
            }
            for(int i = -expand_sz; i <= expand_sz; i++)
            {
                for(int j = -expand_sz; j <= expand_sz; j++)
                {
                    if ( p_table[cur][y+i][x+j] == 0 && vst[y+i][x+j] == false ){
                        p_table[cur][y+i][x+j] = p_table[cur][y][x];
                        vst[y+i][x+j] = true ;
                    }
                }
            }
        }
    }
}

void depthPreditCPU( int pre, int cur)
{
    int cnt0 = 0;
    int cnt1 = 0;
    if ( debugInfo ){
        predictMap = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMap.setTo(0.0) ;

        predictMapUncertainty = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMapUncertainty.setTo(0.0) ;
        predictMapInlier = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        predictMapInlier.setTo(0.0) ;
    }

    for( int y = 0; y < HEIGHT; y++ )
    {
        for( int x = 0 ; x < WIDTH; x++ )
        {
            if ( p_table[cur][y][x] == 0 ){
                alpha[cur][y][x] = 0.0 ;
                cnt0++ ;
            }
            else
            {
                int transform_x = p_table[cur][y][x] & 0xffff;
                int transform_y = p_table[cur][y][x] >> 16 ;
                alpha[cur][y][x] = alpha[pre][transform_y][transform_x] ;
                beta[cur][y][x] = beta[pre][transform_y][transform_x] ;
                mu[cur][y][x] = mu[pre][transform_y][transform_x] ;
                sigma[cur][y][x] = sigma[pre][transform_y][transform_x] ;
                cnt1++ ;

                if ( debugInfo ){
                    predictMap.at<float>(y, x) = mu[cur][y][x] ;
                    predictMapUncertainty.at<float>(y, x) = sqrt(sigma[pre][transform_y][transform_x]) ;
                    predictMapInlier.at<float>(y, x) = expectation(alpha[pre][transform_y][transform_x], beta[pre][transform_y][transform_x]) ;
                }
            }
        }
    }
    //ROS_WARN("predict Num=%d %d", cnt0, cnt1 ) ;
}

inline float normpdfCPU(const float &x, const float &mu, const float &sigma_sq)
{
    return (exp(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * sqrt(2.0f*M_PI*sigma_sq);
}

void depthUpdateCPU(cv::Mat& curDepth, int cur, double outputThreshold = 0.6, float DEP_SAMPLE = 0.0625 )
{
    int cnt0 = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    int cnt3 = 0;
    int cnt4 = 0;

    if ( debugInfo ){
        uncertaintyMap = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        uncertaintyMap.setTo(0.0) ;;
        inlierMap = cv::Mat(HEIGHT, WIDTH, CV_32F ) ;
        inlierMap.setTo(0.0) ;
    }

    for( int y = 0; y < HEIGHT; y++ )
    {
        for( int x = 0 ; x < WIDTH; x++ )
        {
            float depth_estimate = curDepth.at<float>(y, x) ;
            //            float uncertianity = depth_estimate*depth_estimate*depth_estimate*depth_estimate*DEP_SAMPLE*DEP_SAMPLE ;
            float uncertianity = 64 ;

            //            alpha[y][x] = 10.0f;
            //            beta[y][x] = 10.0f;
            //            mu[y][x] = depth_estimate;
            //            sigma[y][x] = 2500.0f;
            //            continue ;

            if( alpha[cur][y][x] < FLOAT_EPS )
            {
                if ( depth_estimate < FLOAT_EPS ){
                    curDepth.at<float>(y, x) = 0;
                    mu[cur][y][x] = depth_estimate;
                    cnt0++ ;

                    if ( debugInfo ) {
                        inlierMap.at<float>(y, x) = 0.0;
                        uncertaintyMap.at<float>(y, x) = 256.0 ;
                    }
                }
                else
                {
                    curDepth.at<float>(y, x) = 0;
                    alpha[cur][y][x] = 10.0f;
                    beta[cur][y][x] = 10.0f;
                    mu[cur][y][x] = depth_estimate;
                    sigma[cur][y][x] = 256.0f;
                    cnt1++ ;

                    if ( debugInfo ){
                        inlierMap.at<float>(y, x) = 0.5;
                        uncertaintyMap.at<float>(y, x) = sqrt(sigma[cur][y][x]) ;
                    }
                }
                continue ;
            }

            if ( depth_estimate < FLOAT_EPS )
            {
                //alpha[cur][y][x] = 0 ;
                beta[cur][y][x] = beta[cur][y][x] + 1.0;
                if( alpha[cur][y][x]/(alpha[cur][y][x] + beta[cur][y][x]) > outputThreshold ){
                    curDepth.at<float>(y, x) = mu[cur][y][x];

                    if ( debugInfo ){
                        uncertaintyMap.at<float>(y, x)  = sqrt(sigma[cur][y][x]) ;
                    }
                }
                else {
                    curDepth.at<float>(y, x) = 0.0;
                    if ( debugInfo ){
                        uncertaintyMap.at<float>(y, x) = 256.0 ;
                    }
                }
                if ( debugInfo ){
                    inlierMap.at<float>(y, x) = alpha[cur][y][x]/(alpha[cur][y][x] + beta[cur][y][x]);
                }
                cnt2++ ;
                continue ;
            }

            //orieigin info
            float a = alpha[cur][y][x] ;
            float b = beta[cur][y][x] ;
            float miu = mu[cur][y][x] ;
            float sigma_sq = sigma[cur][y][x];

            float new_sq = uncertianity * sigma_sq / (uncertianity + sigma_sq);
            float new_miu = (depth_estimate * sigma_sq + miu * uncertianity) / (uncertianity + sigma_sq);
            float c1 = (a / (a+b)) * normpdfCPU(depth_estimate, miu, uncertianity + sigma_sq);
            float c2 = (b / (a+b)) * 1 / 16.0f;

            const float norm_const = c1 + c2;
            c1 = c1 / norm_const;
            c2 = c2 / norm_const;
            const float f = c1 * ((a + 1.0f) / (a + b + 1.0f)) + c2 *(a / (a + b + 1.0f));
            const float e = c1 * (( (a + 1.0f)*(a + 2.0f)) / ((a + b + 1.0f) * (a + b + 2.0f))) +
                    c2 *(a*(a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f)));

            const float mu_prime = c1 * new_miu + c2 * miu;
            const float sigma_prime = c1 * (new_sq + new_miu * new_miu) + c2 * (sigma_sq + miu * miu) - mu_prime * mu_prime;
            const float a_prime = ( e - f ) / ( f - e/f );
            const float b_prime = a_prime * ( 1.0f - f ) / f;
            //const float4 updated = make_float4(a_prime, b_prime, mu_prime, sigma_prime);

            alpha[cur][y][x] = a_prime ;
            beta[cur][y][x] = b_prime ;
            mu[cur][y][x] = mu_prime ;
            sigma[cur][y][x] = sigma_prime ;

            if( a_prime/(a_prime + b_prime) > outputThreshold ){
                curDepth.at<float>(y, x) = mu_prime;
                cnt3++ ;
            }
            else {
                curDepth.at<float>(y, x) = 0.0f;
                cnt4++ ;
            }

            if ( debugInfo ){
                inlierMap.at<float>(y, x) = alpha[cur][y][x]/(alpha[cur][y][x] + beta[cur][y][x]);
                uncertaintyMap.at<float>(y, x) = sqrt(sigma[cur][y][x]) ;
            }
        }
    }
    //ROS_WARN("[depthUpdate] cnt0=%d cnt1=%d cnt2=%d cnt3=%d cnt4=%d", cnt0, cnt1, cnt2, cnt3, cnt4 ) ;
}

void StereoMapper::depthFuseCPU(float fx, float fy, float cx, float cy,
                                Eigen::Matrix3d R_p_2_c, Eigen::Vector3d t_p_2_c,
                                int flag, cv::Mat& curDepth)
{
    float threshold = 0.51;
    if ( flag == 0 )
    {
        transformMapCPU(fx, fy, cx, cy, R_p_2_c, t_p_2_c, 0, 1);
        holeFillingCPU(1) ;
        depthPreditCPU(0, 1);
        depthUpdateCPU(curDepth, 1, threshold, DEP_SAMPLE) ;
    }
    else
    {
        transformMapCPU(fx, fy, cx, cy, R_p_2_c, t_p_2_c, 1, 0);
        holeFillingCPU(0) ;
        depthPreditCPU(1, 0);
        depthUpdateCPU(curDepth, 0, threshold, DEP_SAMPLE) ;
    }
}

void displayDepth(cv::Mat& curDepth, string name )
{
    cv::Mat disp_depth ;
    for( int i = 0 ; i < curDepth.rows; i++ )
    {
        for( int j=0; j < curDepth.cols; j++ )
        {
            if ( curDepth.at<float>(i, j) < 0.1 ) {
                curDepth.at<float>(i, j) = 0 ;
            }
            if ( curDepth.at<float>(i, j) > 10.0 ){
                curDepth.at<float>(i, j) = 0 ;
            }
        }
    }

    static cv::Mat color_disp;
    //    curDepth = curDepth*255.0/10 ;
    //    curDepth.convertTo(disp_depth, CV_8U);
    //    cv::imshow(name, curDepth/10.0) ;
    cv::normalize(curDepth/10.0, disp_depth, 0, 255, CV_MINMAX, CV_8U);
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
    cv::imshow(name, color_disp ) ;
}

void StereoMapper::outputFusionCPU(cv::Mat& result, Eigen::Matrix3d R, Eigen::Vector3d t,
                                   float fx, float fy, float cx, float cy)
{
    if ( CASE == 0 )
    {
        result = cv::Mat(HEIGHT, WIDTH, CV_32F);
        cv::Mat raw_depth_cost ;
        raw_cost.download(raw_depth_cost);

        //printf("cols=%d rows=%d type=%d\n", raw_depth_cost.cols, raw_depth_cost.rows, raw_depth_cost.type() ) ;
        for( int i = 0 ; i < HEIGHT; i++ )
        {
            for( int j = 0; j < WIDTH; j++ )
            {
                double min = 100000.0 ;
                int min_index ;
                for( int d = 0 ; d < DEP_CNT; d++ )
                {
                    int index = INDEX(i, j, d) ;
                    if ( raw_depth_cost.at<float>(0, index) < min ){
                        min = raw_depth_cost.at<float>(0, index) ;
                        min_index = d ;
                    }
                }
                if ( min < FLOAT_EPS ){
                    result.at<float>(i, j) = 10000.0 ;
                }
                else if ( min_index == 0 || min_index == DEP_CNT - 1 ){
                    result.at<float>(i, j) = 0.0 ;
                }
                else {
                    result.at<float>(i, j) = 1.0f / (min_index * DEP_SAMPLE) ;
                }
            }
        }

        return ;
    }

    //    int64 t = cv::getTickCount();
    //    ROS_WARN("P1:%.0lf, P2:%.0lf, tau_so:%.0lf, q1:%.0lf, q2:%.0lf",
    //    pi1, pi2, tau_so, sgm_q1, sgm_q2);

    sgm_cost.setTo(cv::Scalar_<float>(0.0));
    sgm2(raw_cost.data,
         sgm_cost.data);

    if ( CASE == 1 )
    {
        result = cv::Mat(HEIGHT, WIDTH, CV_32F);
        cv::Mat raw_depth_cost ;
        sgm_cost.download(raw_depth_cost);

        //printf("cols=%d rows=%d type=%d\n", raw_depth_cost.cols, raw_depth_cost.rows, raw_depth_cost.type() ) ;
        for( int i = 0 ; i < HEIGHT; i++ )
        {
            for( int j = 0; j < WIDTH; j++ )
            {
                double min = 100000.0 ;
                int min_index ;
                for( int d = 0 ; d < DEP_CNT; d++ )
                {
                    int index = INDEX(i, j, d) ;
                    if ( raw_depth_cost.at<float>(0, index) < min ){
                        min = raw_depth_cost.at<float>(0, index) ;
                        min_index = d ;
                    }
                }
                if ( min < FLOAT_EPS ){
                    result.at<float>(i, j) = 10000.0 ;
                }
                else if ( min_index == 0 || min_index == DEP_CNT - 1 ){
                    result.at<float>(i, j) = 0.0 ;
                }
                else {
                    result.at<float>(i, j) = 1.0f / (min_index * DEP_SAMPLE) ;
                }
            }
        }

        return ;
    }

    //    int64 t2 = cv::getTickCount();
    //    printf("PW CUDA Time part 3 (SGM): %fms\n", (t2 - t) * 1000 / cv::getTickFrequency());

    //depth refinement
    filter_cost(sgm_cost.data,
                dep.data, dep.step, DEP_SAMPLE);

    if ( CASE == 2 )
    {
        dep.download(result);
        return ;
    }

    //cout << "R and t\n" << R << "\n" << t.transpose() < "\n" ;

    if ( firstDepth ){
        firstDepth = false ;
        //puts("aaa");
        //printf("img_l.rows=%d img_l.cols=%d\n", img_l.rows, img_l.cols ) ;
        result = cv::Mat(img_l.rows, img_l.cols, CV_32F ) ;
        //printf("type=%d\n", result.type() ) ;
        result.setTo(0.0) ;

        R_b_2_w = R ;
        t_b_2_w = t ;
        pre_index = 0 ;
        cur_index = 1 ;

        memset(alpha, 0, sizeof(alpha) ) ;
        memset(beta, 0, sizeof(beta) ) ;
        memset(mu, 0, sizeof(mu) ) ;
        memset(sigma, 0, sizeof(sigma) ) ;
    }
    else
    {
        dep.download(result);
        cv::Mat mask(result.rows, result.cols, CV_8U) ;
        for( int i = 0 ; i < result.rows; i++ )
        {
            for( int j = 0 ; j < result.cols; j++ )
            {
                if ( result.at<float>(i, j) > 100 ){
                    result.at<float>(i, j) = 0.0 ;
                    mask.at<uchar>(i, j) = 255 ;
                }
                else{
                    mask.at<uchar>(i, j) = 0 ;
                }
                if ( result.at<float>(i, j) > 10 ){
                    result.at<float>(i, j) = 0.0 ;
                }
            }
        }
        //cv::imshow("mask", mask ) ;

        //
        static cv::Mat save_result ;
        save_result = result.clone();

        Eigen::Matrix3d R_p_2_c = R.transpose()*R_b_2_w ;
        Eigen::Vector3d t_p_2_c = R.transpose()*(t_b_2_w-t) ;

        //        int64 t2 = cv::getTickCount();

        depthFuseCPU(fx, fy, cx, cy, R_p_2_c, t_p_2_c, pre_index, result);
        for( int i = 0 ; i < result.rows; i++ )
        {
            for( int j = 0 ; j < result.cols; j++ )
            {
                if ( mask.at<uchar>(i, j) > 100 ){
                    result.at<float>(i, j) = 0.0 ;
                }
                if ( result.at<float>(i, j) < 0.1 ){
                    result.at<float>(i, j) = 0.0;
                }
            }
        }
        //        printf("FUSE: %fms\n", (cv::getTickCount() - t2) * 1000 / cv::getTickFrequency());

        if ( debugInfo )
        {
            cv::Mat forDisplay ;
            displayDepth(previousMap, "prevous") ;
            cv::imshow("previousUncertainty", previousUncertainty/8) ;
            cv::imshow("previousInlier", previousInlier*1.6 ) ;

            displayDepth(predictMapNoFill, "predictMapNoFill") ;
            cv::imshow("predictMapNoFillUncertainty", predictMapNoFillUncertainty/8) ;
            cv::imshow("predictMapNoFillInlier", predictMapNoFillInlier*1.6 ) ;

            displayDepth(predictMap, "predict") ;
            cv::imshow("predictMapUncertainty", predictMapUncertainty/8) ;
            cv::imshow("predictMapInlier", predictMapInlier*1.6 ) ;

            displayDepth(save_result, "raw depth") ;


            //cv::normalize(uncertaintyMap, forDisplay, 0, 255, CV_MINMAX, CV_8U);
            cv::imshow("updated uncertaintyMap", uncertaintyMap/8) ;
            //cv::normalize(inlierMap, forDisplay, 0, 255, CV_MINMAX, CV_8U);
            cv::imshow("updated inlierMap", inlierMap*1.6 ) ;
        }

        //        displayDepth(result, "fuse") ;
        //        cv::waitKey(3) ;

        R_b_2_w = R ;
        t_b_2_w = t ;

        pre_index = cur_index ;
        cur_index ^= 1 ;
    }
}

void StereoMapper::epipolar(double x, double y, double z)
{
    cv::Mat P{3, 1, CV_64F};
    P.at<double>(0, 0) = x;
    P.at<double>(1, 0) = y;
    P.at<double>(2, 0) = z;
    cv::Mat p = nK1 * P;
    double uu = p.at<double>(0, 0) / p.at<double>(2, 0);
    double vv = p.at<double>(1, 0) / p.at<double>(2, 0);
    int u = uu + 0.5;
    int v = vv + 0.5;
    printf("%f %f %d %d\n", uu, vv, u, v);
    cv::Mat img;
    img_r.download(img);
    cv::Mat pp{3, 1, CV_64F};
    pp.at<double>(0, 0) = u;
    pp.at<double>(1, 0) = v;
    pp.at<double>(2, 0) = 1;

    cv::Mat imgl = img_intensity.clone();
    cv::circle(imgl, cv::Point(u, v), 2, cv::Scalar(-1));

    for (int i = 0; i < DEP_CNT; i++)
    {
        cv::Mat ppp = R * pp;
        double idep = i * DEP_SAMPLE;
        double x = ppp.at<double>(0, 0);
        double y = ppp.at<double>(1, 0);
        double z = ppp.at<double>(2, 0);

        float w = z + T.at<double>(2, 0) * idep;
        float u = (x + T.at<double>(0, 0) * idep) / w;
        float v = (y + T.at<double>(1, 0) * idep) / w;
        printf("%f %f\n", u, v);
        cv::circle(img, cv::Point(u, v), 2, cv::Scalar(-1));
    }
    cv::imshow("r", imgl);
    cv::imshow("m", img);
    cv::waitKey(10);
}
