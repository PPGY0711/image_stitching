#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
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
private:
    cv::Ptr<cv::xfeatures2d::SURF> surfDetector;
    cv::Ptr<cv::ORB> orbDetector;
    cv::Ptr<cv::xfeatures2d::StarDetector> starDetector;
    cv::Ptr<cv::xfeatures2d::MSDDetector> msdDetector;
    cv::Ptr<cv::xfeatures2d::BEBLID> beblidExtractor;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> briefExtractor;
    void initDetectorsAndExtractors(){
        //    -----SURF----
//        cv::Ptr<cv::xfeatures2d::SURF> surfDetector;
        this->surfDetector = cv::xfeatures2d::SURF::create(800);  //800为海塞矩阵阈值，越大越精准
        //    -----ORB------
//        cv::Ptr<cv::ORB> orbDetector;
        this->orbDetector  = cv::ORB::create(8000,1.2);//保留点数
        //    -----STAR-----
//        cv::Ptr<cv::xfeatures2d::StarDetector> starDetector;
        this->starDetector = cv::xfeatures2d::StarDetector::create();
        //    -----MSD-----
//        this->cv::Ptr<cv::xfeatures2d::MSDDetector> msdDetector;
        this->msdDetector = cv::xfeatures2d::MSDDetector::create();

        this->beblidExtractor = cv::xfeatures2d::BEBLID::create(0.75, cv::xfeatures2d::BEBLID::SIZE_256_BITS);

        this->briefExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
    }
public:
    bool init = false;
    enum DetectFlags {
        DETECT_SURF = 0,
        DETECT_ORB = 1,
        DETECT_STAR = 2,
        DETECT_MSD = 3
    };
    enum DescriptorFlags {
        DES_SURF = 0,
        DES_ORB = 1,
        DES_BEBLID = 2,
        DES_BRIEF = 3,
    };
    ORBAlgorithm();
    ORBAlgorithm(bool init);
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
    void matchTwoPic(const cv::Mat &img1, const cv::Mat &mask1, const cv::Mat &img2, const cv::Mat &mask2,
                     int detectFlag, int desFlag, cv::Mat& homography);
};
#endif //STITCH_C_ORB_ALGO_H
