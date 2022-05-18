#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <cmath>
#define PI acos(-1)
#ifndef STITCH_C_IMAGE_STITCH_H
#define STITCH_C_IMAGE_STITCH_H
class ImageStitcher{
public:
    ImageStitcher();
    ImageStitcher(bool useProject);
    ~ImageStitcher();
    int count;
    int centerProjectF;
    bool useProject;
    void getHomography(const cv::Mat& src1, const cv::Mat& mask1, const cv::Mat& scr2, const cv::Mat& mask2, cv::Mat& homography, bool flag);
    double cylinderProjection(const cv::Mat& src, cv::Mat& dst);
    void reverseCylinderProjection(const cv::Mat& src, cv::Mat& dst);
    void stitchImages(const std::vector<cv::Mat>& srcs, cv::Mat& dst, std::vector<double> cr2r, std::vector<double> cr2l, int c_num);
    void chooseCenterPoint(cv::Mat &src, int* centerX);
    void collectLineSegments(cv::Mat &src, std::vector<std::pair<cv::Point, cv::Point>>& pointPairs);
    static void removeBlackSide(cv::Mat& src, cv::Mat& dst, int pos_num=8){
//        std::cout << src<< std::endl;
        cv::imshow("removeBlackSide", src);
        cv::waitKey(0);
        std::cout << src.cols << ", " << src.rows << std::endl;
        // 从右边开始找最远的不为0的像素点，去除黑边
        int maxNotZeroFromRight = 0, minNotZeroFromLeft = 0;
        uchar* p_pos[pos_num+1];
        for(int i = 0; i < pos_num; i++){
            int y = (int)((double)i*src.rows/(pos_num));
//            std::cout << y << std::endl;
            p_pos[i] = src.ptr<uchar>(y);
        }
        p_pos[pos_num] = src.ptr<uchar>(src.cols-1);
        for (int m = (src.cols-1)*3; m >= 0; m-=3)
        {
            std::vector<bool> pixel_exist(pos_num);
            bool allExist = false;
            for(int i = 0; i <= pos_num; i++){
//                std::cout << "col: " << m << std::endl;
                pixel_exist[i] = p_pos[i][m] || p_pos[i][m + 1] || p_pos[i][m + 2];
//                std::cout << (int)p_pos[0][m] <<", "<<  (int)p_pos[0][m + 1] <<", "<<  (int)p_pos[0][m + 2] << std::endl;
                allExist = (allExist||pixel_exist[i]);
            }
            if (allExist)
            {
//                std::cout << (int)p_pos[0][m] <<", "<<  (int)p_pos[0][m + 1] <<", "<<  (int)p_pos[0][m + 2] << std::endl;
                maxNotZeroFromRight = m/3;
                break;
            }
        }
        for (int m = 0; m < src.cols*3; m+=3)
        {
            std::vector<bool> pixel_exist(pos_num+1);
            bool allExist = false;
            for(int i = 0; i <= pos_num; i++){
                pixel_exist[i] = p_pos[i][m] || p_pos[i][m + 1] || p_pos[i][m + 2];
                allExist = (allExist||pixel_exist[i]);
            }
            if (allExist)
            {
                minNotZeroFromLeft = m/3;
                break;
            }
        }
        std::cout << "maxNotZero From Right: " << maxNotZeroFromRight << std::endl;
        std::cout << "minNotZero From Left: " << minNotZeroFromLeft << std::endl;
        cv::Mat tmp;
        tmp = src(cv::Rect(cv::Point2f(minNotZeroFromLeft, 0), cv::Point2f(std::min(maxNotZeroFromRight,src.cols-1), src.rows - 1)));
        tmp.copyTo(dst);
        cv::imshow("after removeBlackSide", dst);
        cv::waitKey(0);
        std::cout << src.cols << ", " << src.rows << std::endl;
    }
};
#endif //STITCH_C_IMAGE_STITCH_H
