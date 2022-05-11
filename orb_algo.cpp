#include "orb_algo.h"
using namespace cv;
using namespace std;
using namespace cv::detail;

ORBAlgorithm::ORBAlgorithm() {}
ORBAlgorithm::~ORBAlgorithm() {}

//detect feature points from two images
//img1 and img2 are two images to match
//ImageFeatures is a new struct in OpenCV4 containing keypoints and the descriptors of the feature points
void ORBAlgorithm::detectPoints(const Mat &img1, const Mat &mask1, const Mat &img2, const Mat &mask2,
                                ImageFeatures &features1, ImageFeatures &features2) {
    //detect 4000 keypoints, the first image scale is 1.2 and detect 2 scale images
    Ptr<Feature2D> finder;
    finder = ORB::create(10000, 1.2);
    int64 t1 = getTickCount(), t2 = 0, t3 = 0, t4 = 0;
    computeImageFeatures(finder, img1, features1, mask1);
    t2 = getTickCount();
    double cost1 = 1000.0 * (t2-t1)/getTickFrequency();

    t3 = getTickCount();
    computeImageFeatures(finder, img2, features2, mask2);
    t4 = getTickCount();
    double cost2 = 1000.0 * (t4-t3)/getTickFrequency();
    cout << "detectPoints, time cost1: " << cost1 << ", time cost2: " << cost2 << endl;
}

//match feature points and return the match points pair
void ORBAlgorithm::matchPoints(const ImageFeatures &features1, const ImageFeatures &features2,
                               vector<DMatch>& matchPairs) {
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches1to2;
    vector<DMatch> matches2to1;
    vector<DMatch> twoDirectionMatch;
    //match
    matcher->match(features1.descriptors, features2.descriptors, matches1to2);
    matcher->match(features2.descriptors, features1.descriptors, matches2to1);
    //get the intersection of match image1 to image2 and match image2 to image1
    int* flag = new int[features2.descriptors.rows];
    memset(flag,-1,sizeof(int)*features2.descriptors.rows);
    for(size_t i = 0; i < features2.descriptors.rows; ++i){
        flag[matches2to1[i].queryIdx] = matches2to1[i].trainIdx;
    }
    for(size_t i = 0; i < matches1to2.size(); ++i){
        if(flag[matches1to2[i].trainIdx] == matches1to2[i].queryIdx){
            twoDirectionMatch.push_back(matches1to2[i]);
        }
    }
    sort(twoDirectionMatch.begin(), twoDirectionMatch.end(), sortByDistance);
    size_t loop_time = twoDirectionMatch.size();
    if(loop_time >= 500) loop_time = 500;
    for(size_t i = 0; i < loop_time; ++i){
        matchPairs.push_back(twoDirectionMatch[i]);
    }
}

//get homography mat using RANSAC
void ORBAlgorithm::getHomographyMat(const ImageFeatures &features1, const ImageFeatures &features2,
                                    vector<DMatch>& goodMatches, Mat &homography) {
    vector<Point2f> image1Points, image2Points;
    vector<KeyPoint> kp1 = features1.keypoints;
    vector<KeyPoint> kp2 = features2.keypoints;
    for(size_t i = 0; i < goodMatches.size(); i++){
        image1Points.push_back(kp1[goodMatches[i].queryIdx].pt);
        image2Points.push_back(kp2[goodMatches[i].trainIdx].pt);
    }
    Mat inlier_mask;
    homography = findHomography(image2Points, image1Points, RANSAC, 2.5f, inlier_mask);
    cout << homography << endl;
    int i = 0;
    for(vector<DMatch>::iterator iter = goodMatches.begin(); iter != goodMatches.end();){
        if(!inlier_mask.at<uchar>(i)){
            iter = goodMatches.erase(iter);
        }else{
            ++iter;
        }
        i++;
    }
}

//get homography mat using RANSAC
void ORBAlgorithm::getHomographyMat(const Mat& img1, const Mat& img2, const ImageFeatures &features1, const ImageFeatures &features2,
                                    vector<DMatch>& goodMatches, Mat &homography, string& imgPath) {
    vector<Point2f> image1Points, image2Points;
    vector<KeyPoint> kp1 = features1.keypoints;
    vector<KeyPoint> kp2 = features2.keypoints;
    for(size_t i = 0; i < goodMatches.size(); i++){
        image1Points.push_back(kp1[goodMatches[i].queryIdx].pt);
        image2Points.push_back(kp2[goodMatches[i].trainIdx].pt);
    }
    Mat inlier_mask;
    homography = findHomography(image2Points, image1Points, RANSAC, 2.5f, inlier_mask);
    cout << homography << endl;
    int i = 0;
    for(vector<DMatch>::iterator iter = goodMatches.begin(); iter != goodMatches.end();){
        if(!inlier_mask.at<uchar>(i)){
            iter = goodMatches.erase(iter);
        }else{
            ++iter;
        }
        i++;
    }
    cout << "goodMatch num: " << goodMatches.size() << endl;
    Mat matchImg;
    drawMatches(img1, kp1, img2, kp2, goodMatches, matchImg, Scalar(0, 255, 0),Scalar(0,0,255));
    imwrite(imgPath, matchImg);
    imshow("match Result", matchImg);
}