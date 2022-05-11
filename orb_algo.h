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
#ifndef STITCH_C_ORB_ALGO_H
#define STITCH_C_ORB_ALGO_H
class ORBAlgorithm{
public:

    ORBAlgorithm();
    ~ORBAlgorithm();
    //detect feature points
    void detectPoints(const cv::Mat &img1, const cv::Mat &mask1, const cv::Mat &img2, const cv::Mat &mask2,
                      cv::detail::ImageFeatures &features1, cv::detail::ImageFeatures &features2);
    //match feature points and return the match points pair
    void matchPoints(const cv::detail::ImageFeatures &features1, const cv::detail::ImageFeatures &features2
                     , std::vector<cv::DMatch>& matchPairs);
    //use RANSAC to get the homography mat
    void getHomographyMat(const cv::detail::ImageFeatures &features1, const cv::detail::ImageFeatures &features2,
                          std::vector<cv::DMatch>& goodMatches, cv::Mat &homography);
    void getHomographyMat(const cv::Mat &img1, const cv::Mat &img2,
                          const cv::detail::ImageFeatures &features1, const cv::detail::ImageFeatures &features2,
                          std::vector<cv::DMatch>& goodMatches, cv::Mat &homography, std::string& imgParh);
    static bool sortByDistance(const cv::DMatch& match1, const cv::DMatch &match2){
        return match1.distance < match2.distance;
    }
};
#endif //STITCH_C_ORB_ALGO_H
