#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
/**
 * This demo shows how BEBLID descriptor can be used with a feature detector (here ORB) to
 * detect, describe and match two images of the same scene.
 */

int main()
{
    //可能需要开始手动处理标出标志线了。
    // Read the input images in grayscale format (CV_8UC1)
    cv::Mat img1 = cv::imread("../img/undist/sharpened_32.bmp");
    cv::Mat img2 = cv::imread("../img/undist/sharpened_33.bmp");
//    GaussianBlur(img1, img1,Size(3, 3), 10);
//    GaussianBlur(img2, img2,Size(3, 3), 10);
    Mat gray1,gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);
    bilateralFilter(img1, gray1, 10, 40, 10);
    bilateralFilter(img2, gray2, 10, 40, 10);
    Mat edge1,edge2;
    //canny
    Canny(gray1, edge1,60,100,3,true);
    Canny(gray2, edge2,60,100,3,true);
    edge1.copyTo(img1);
    edge2.copyTo(img2);
//    gray1.copyTo(img1);
//    gray2.copyTo(img2);
    cv::Mat mask1 = cv::Mat::zeros(img1.size(),img1.type());
    mask1(cv::Rect(img1.cols/2,0,img1.cols/2,2*img1.rows/5)).setTo(255);
    cv::Mat mask2 = cv::Mat::zeros(img1.size(),img1.type());
    mask2(cv::Rect(0,0,img2.cols/2,2*img2.rows/5)).setTo(255);
    cv::imshow("Original image1", img1);
    cv::imshow("Original image2", img2);
    cv::imshow("mask1", mask1);
    cv::imshow("mask2", mask2);
    bool used_beblid = true; // true: beblid  false: brief

    // Create the feature detector, for example ORB
    auto detector = cv::ORB::create(4000, 1.2);

    // Use 32 bytes per descriptor and configure the scale factor for ORB detector
    auto beblid = cv::xfeatures2d::BEBLID::create(0.75, cv::xfeatures2d::BEBLID::SIZE_256_BITS);
    auto brief  = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
    // Detect features in both images
    std::vector<cv::KeyPoint> points1, points2;
    detector->detect(img1, points1,mask1);
    detector->detect(img2, points2,mask2);
    std::cout << "Detected  " << points1.size() << " kps in image1" << std::endl;
    std::cout << "Detected  " << points2.size() << " kps in image2" << std::endl;
    // Describe the detected features i both images
    cv::Mat descriptors1, descriptors2;

    if (used_beblid)
    {
        beblid->compute(img1, points1, descriptors1);
        beblid->compute(img2, points2, descriptors2);
    }
    else
    {
        //detector->compute(img1, points1, descriptors1);
        //detector->compute(img2, points2, descriptors2);
        brief->compute(img1, points1, descriptors1);
        brief->compute(img2, points2, descriptors2);
    }
    std::cout << "Points described" << std::endl;

    // Match the generated descriptors for img1 and img2 using brute force matching
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::cout << "Number of matches: " << matches.size() << std::endl;

    // If there is not enough matches exit
    if (matches.size() < 4)
        exit(-1);

    // Take only the matched points that will be used to calculate the
    // transformation between both images
    std::vector<cv::Point2d> matched_pts1, matched_pts2;
    for (cv::DMatch match : matches)
    {
        matched_pts1.push_back(points1[match.queryIdx].pt);
        matched_pts2.push_back(points2[match.trainIdx].pt);
    }

    // Find the homography that transforms a point in the first image to a point in the second image.
    cv::Mat inliers;
    cv::Mat H = cv::findHomography(matched_pts1, matched_pts2, cv::RANSAC, 3, inliers);
    // Print the number of inliers, that is, the number of points correctly
    // mapped by the transformation that we have estimated
    std::cout << "Number of inliers " << cv::sum(inliers)[0]
              << " ( " << (100.0f * cv::sum(inliers)[0] / matches.size()) << "% )" << std::endl;

    // Convert the image to BRG format from grayscale
    cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);

    // Draw all the matched keypoints in red color
    cv::Mat all_matches_img;
    cv::drawMatches(img1, points1, img2, points2, matches, all_matches_img,
                    CV_RGB(255, 0, 0), CV_RGB(255, 0, 0));  // Red color

    // Draw the inliers in green color
    for (int i = 0; i < matched_pts1.size(); i++)
    {
        if (inliers.at<uchar>(i, 0))
        {
            cv::circle(all_matches_img, matched_pts1[i], 3, CV_RGB(0, 255, 0), 2);
            // Calculate second point assuming that img1 and img2 have the same height
            cv::Point p2(matched_pts2[i].x + img1.cols, matched_pts2[i].y);
            cv::circle(all_matches_img, p2, 3, CV_RGB(0, 255, 0), 2);
            cv::line(all_matches_img, matched_pts1[i], p2, CV_RGB(0, 255, 0), 2);
        }
    }

    // Show and save the result
    cv::imshow("All matches", all_matches_img);
    cv::imwrite("./imgs/inliners_img.jpg", all_matches_img);

    // Transform the first image to look like the second one
    cv::Mat transformed;
    cv::warpPerspective(img1, transformed, H, img2.size());


    cv::imshow("Transformed image", transformed);

    cv::waitKey(0);

    return 0;
}
