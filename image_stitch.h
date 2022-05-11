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
    ~ImageStitcher();
    int count;
    int centerProjectF;
    void getHomography(const cv::Mat& src1, const cv::Mat& mask1, const cv::Mat& scr2, const cv::Mat& mask2, cv::Mat& homography);
    double cylinderProjection(const cv::Mat& src, cv::Mat& dst);
    void reverseCylinderProjection(const cv::Mat& src, cv::Mat& dst);
    void stitchImages(const std::vector<cv::Mat>& srcs, cv::Mat& dst, int c_num);
    void chooseCenterPoint(cv::Mat &src, int* centerX);
    void collectLineSegments(cv::Mat &src, std::vector<std::pair<cv::Point, cv::Point>>& pointPairs);
    static void removeBlackSide(const cv::Mat& src, cv::Mat& dst){
        // 从右边开始找最远的不为0的像素点，去除黑边
        int maxNotZeroFromRight = 0, minNotZeroFromLeft = 0;
        const uchar* p_middle = src.ptr<uchar>(src.rows / 2);
        for (int m = (src.cols-1)*3; m >= 0; m-=3)
        {
            bool b_middle = p_middle[m] || p_middle[m + 1] || p_middle[m + 2];
            if (b_middle)
            {
                maxNotZeroFromRight = m/3;
                break;
            }
        }
        for (int m = 0; m < src.cols*3; m+=3)
        {
            bool b_middle = p_middle[m] || p_middle[m + 1] || p_middle[m + 2];
            if (b_middle)
            {
                minNotZeroFromLeft = m/3;
                break;
            }
        }
        std::cout << "maxNotZero From Right: " << maxNotZeroFromRight << std::endl;
        std::cout << "minNotZero From Left: " << minNotZeroFromLeft << std::endl;
        dst = src(cv::Rect(cv::Point2f(minNotZeroFromLeft, 0), cv::Point2f(std::min(maxNotZeroFromRight,src.cols-1), src.rows - 1)));
    }
};
#endif //STITCH_C_IMAGE_STITCH_H
