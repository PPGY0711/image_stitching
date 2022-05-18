#include "image_stitch.h"
#include "orb_algo.h"
using namespace cv;
using namespace std;
using namespace cv::detail;

#define ROWMASKRATIO (4.5/9.0)
#define COLMASKRATIO (1.0/3.0)
#define CANNYThs1 80
#define CANNYThs2 120

ImageStitcher::ImageStitcher() {
    this->useProject = true;
}
ImageStitcher::ImageStitcher(bool useProject) {
    this->useProject = false;
}
ImageStitcher::~ImageStitcher() {}

static void getCanny(const Mat& src, Mat& dst){
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat edge;
    //canny
    Canny(gray, edge,CANNYThs1,CANNYThs2,3,true);
    edge.copyTo(dst);
}

void ImageStitcher::getHomography(const cv::Mat &src1, const cv::Mat &mask1, const cv::Mat &src2, const cv::Mat &mask2,
                                  cv::Mat &homography, bool flag) {
//    imshow("src1", src1);
//    imshow("mask1", mask1);
//    imshow("src2", src2);
//    imshow("mask2", mask2);
    ORBAlgorithm orb;
    ImageFeatures features1, features2;
    Mat img1 = src1.clone(), img2 = src2.clone();
    Mat newMask1 = mask1.clone(), newMask2 = mask2.clone();
    if(flag){
        // 提取图片边缘用于匹配
        Mat edge1, edge2;
        getCanny(img1, edge1);
        getCanny(img2, edge2);
        edge1.copyTo(img1);
        edge2.copyTo(img2);
    }
    cout << "detecting keypoints..." << endl;
    orb.detectPoints(img1, newMask1, img2, newMask2, features1, features2);
    cout << "num of feature points detected in img1 and img2: "<<features1.keypoints.size() << ", " << features2.keypoints.size() << endl;
    vector<DMatch> matchPairs;
    cout << "matching images..." << endl;
    orb.matchPoints(features1, features2,matchPairs);
    cout << "computing the homography mat..." << endl;
    string path = "../output/canny/matchImage_"+to_string(this->count++)+".png";
    orb.getHomographyMat(img1, img2, features1, features2, matchPairs, homography, path);
}

double ImageStitcher::cylinderProjection(const cv::Mat &src, cv::Mat &dst) {
    int width = src.cols;
    int height = src.rows;
    int centerX = width / 2;
    int centerY = height / 2;
    dst = src.clone();
    double alpha = PI/5;
    double f = width / (2 * tan(alpha / 2));
//    double f = 1.5*width / (2*tan(alpha / 2));
    double theta, pointX, pointY;
    for (int i = 0; i < height; i++) {
        uchar *ptr = dst.ptr<uchar>(i);
        int k = 0;
        for (int j = 0; j < width; j++) {
            theta = asin((j - centerX) / f);
            pointY = f * tan((j - centerX) / f) + centerX;
            pointX = (i - centerY) / cos(theta) + centerY;

            if (pointX >= 0 && pointX <= height && pointY >= 0 && pointY <= width) {
                const uchar *tmp = src.ptr<uchar>(pointX);
                ptr[k] = tmp[(int) pointY * 3];
                ptr[k + 1] = tmp[(int) pointY * 3 + 1];
                ptr[k + 2] = tmp[(int) pointY * 3 + 2];
            } else {
                ptr[k] = 0;
                ptr[k + 1] = 0;
                ptr[k + 2] = 0;
            }
            k += 3;
        }
    }
    return f;
}


void ImageStitcher::reverseCylinderProjection(const Mat &src, Mat &dst) {
    int width = src.cols;
    int height = src.rows;
//    Mat tmpSrc = Mat::zeros(src.rows, newWidth, src.type());
    Mat tmpDst = Mat::zeros(width, height, src.type());
//    cout << adjust.size() << endl;
//    cout << newWidth/2-chooseX << endl;
    // 图像平移或者用ROI
    // 得到中心点坐标之后，将中心点移到视野中央
//    Mat tROI = tmpSrc(Rect(newWidth/2-src.cols/2-1, 0, src.cols, src.rows));
    // 得到中心点坐标之后，将中心点移到视野中央
    int centerX = width / 2;
    int centerY = height / 2;
//    src.convertTo(tROI, tROI.type());
//    imshow("newSrc", tmpSrc);
    double f = this->centerProjectF;
    double theta, pointX, pointY;
    for (int i = 0; i < height; i++) {
        double y = i-centerY;
        for (int j = 0; j < width; j++) {
            double x = j-centerX;
            // 反柱面投影公式
            pointX = f*tan(x/f);
            theta = atan(pointX/f);
            pointY = y/cos(theta);
            pointX += centerX;
            pointY += centerY;
            const uchar *tmp = src.ptr<uchar>(i);
            if (pointX >= 0 && pointX <= width && pointY >= 0 && pointY <= height) {
                uchar *ptr = tmpDst.ptr<uchar>((int)pointX);
                ptr[(int)pointY*3] = tmp[j*3];
                ptr[(int)pointY*3 + 1] = tmp[j*3 + 1];
                ptr[(int)pointY*3 + 2] = tmp[j*3 + 2];
            }
        }
    }
    // 反柱面投影之后需要先顺时针旋转90°再翻转才是原图的拉直效果
    Mat roDst, fliDst;
    rotate(tmpDst, roDst, ROTATE_90_CLOCKWISE);
    flip(roDst, fliDst, 1);
    dst = fliDst.clone();
}


void ImageStitcher::stitchImages(const std::vector<cv::Mat> &srcs, Mat &dst, std::vector<double> cr2r, std::vector<double> cr2l, int c_num) {
    // 图片将从图片序列的中间往两边拼接
    int s_num = (int)srcs.size();
    cout << "num of images: " << s_num << endl;
    Mat p_src1;
    // 从中间位置往外拼
    c_num = c_num > s_num || c_num==0? s_num/2:c_num;
    if(this->useProject){
        this->centerProjectF = cylinderProjection(srcs[c_num], p_src1);
        imshow("before Cylinder Projection1", srcs[c_num]);
        imshow("after Cylinder Projection1", p_src1);
        // 画出柱面投影的中心点，为最终全景图片的反投影提供图形依据
        circle(p_src1, Point(p_src1.cols/2,p_src1.rows/2), 5, Scalar(0,255,0),1);
        Mat revTmp;
        reverseCylinderProjection(p_src1, revTmp);
        imshow("after Reverse Cylinder Projection1", revTmp);
        waitKey(0);
    }else{
        // 若不柱面投影
        srcs[c_num].copyTo(p_src1);
    }
    for(int j = c_num-1; j >= 0; --j){
        // 从中间往左拼
        Mat p_src2;
        if(this->useProject){
            cylinderProjection(srcs[j], p_src2);
        }else{
            srcs[j].copyTo(p_src2);
        }
        int extraCol = p_src2.cols*cr2l[c_num-1-j];
        Mat tp_src1 = Mat::zeros(p_src1.rows, p_src1.cols + extraCol, p_src1.type());
        Mat tROI = tp_src1(Rect(extraCol, 0, p_src1.cols, p_src1.rows));
        p_src1.convertTo(tROI, tROI.type());
        imshow("roiMat", tROI);
        imshow("tp_src1", tp_src1);
        waitKey(0);
        Mat H;
        Mat mask1 = Mat::zeros(tp_src1.size(), CV_8UC1);
//        cout <<"here:" << p_src2.cols*cr2l[c_num-1-j] << ", " << p_src1.cols*cr2l[c_num-1-j] << endl;
        mask1(Rect(extraCol, 0, p_src2.cols*cr2l[c_num-1-j], ROWMASKRATIO*p_src1.rows)).setTo(255);
        Mat mask2 = Mat::zeros(p_src2.size(), CV_8UC1);
        mask2(Rect(p_src2.cols*(1-cr2l[c_num-1-j]), 0, p_src2.cols*cr2l[c_num-1-j], ROWMASKRATIO*p_src2.rows)).setTo(255);
        getHomography(tp_src1, mask1, p_src2, mask2, H, false);
        cout << H << endl;
        Mat tmp;
        warpPerspective(p_src2, tmp, H, Size(tp_src1.cols, tp_src1.rows));
        imshow("tmp after homography", tmp);
        // 图像融合
        for (size_t i = 0; i < tmp.rows; i++)
        {
            uchar* p_roiMat = tp_src1.ptr<uchar>(i);
            uchar* p_tmp = tmp.ptr<uchar>(i);
            for (size_t j = 0; j < tmp.cols * 3; j += 3)
            {
                if (p_roiMat[j] || p_roiMat[j + 1] || p_roiMat[j + 2])
                {
                    if (p_tmp[j] || p_tmp[j + 1] || p_tmp[j + 2])
                    {
                        int dis = j / 3 - extraCol;
                        if (dis < 200 && dis > 0)
                        {
                            double weight = 1.0 * dis / 200;
                            p_tmp[j] = p_roiMat[j] * weight + p_tmp[j] * (1 - weight);
                            p_tmp[j + 1] = p_roiMat[j + 1] * weight + p_tmp[j + 1] * (1 - weight);
                            p_tmp[j + 2] = p_roiMat[j + 2] * weight + p_tmp[j + 2] * (1 - weight);
                        }
                        else if (dis > 0)
                        {
                            p_tmp[j] = p_roiMat[j];
                            p_tmp[j + 1] = p_roiMat[j + 1];
                            p_tmp[j + 2] = p_roiMat[j + 2];
                        }
                    }
                    else
                    {
                        p_tmp[j] = p_roiMat[j];
                        p_tmp[j + 1] = p_roiMat[j + 1];
                        p_tmp[j + 2] = p_roiMat[j + 2];
                    }
                }
            }
        }
        imshow("tmp after blender", tmp);
        waitKey(0);
        Mat tmpROI;
        removeBlackSide(tmp, tmpROI);
        p_src1 = tmpROI;
    }
    dst = p_src1.clone();
    imshow("stitch in the half way", dst);
    waitKey(0);
    for(int i = c_num+1; i < s_num; ++i){
        // 从中间往右拼
        Mat p_src2;
        if(this->useProject){
            cylinderProjection(srcs[i], p_src2);
        }else{
            srcs[i].copyTo(p_src2);
        }
        int extraCol = p_src2.cols*(1-cr2r[i-c_num-1]);
        Mat tp_src1 = Mat::zeros(p_src1.rows, p_src1.cols + extraCol, p_src1.type());
        Mat tROI = tp_src1(Rect(0, 0, p_src1.cols, p_src1.rows));
        p_src1.convertTo(tROI, tROI.type());
        //homography mat
        Mat H;
        Mat mask = Mat::zeros(tp_src1.size(), CV_8UC1);
        mask(Rect(p_src1.cols-p_src2.cols*(cr2r[i-c_num-1]),0,p_src2.cols*(cr2r[i-c_num-1]), ROWMASKRATIO*tp_src1.rows)).setTo(255);
        Mat mask2 = Mat::zeros(p_src2.size(), CV_8UC1);
        mask2(Rect(0, 0, p_src2.cols*cr2r[i-c_num-1], ROWMASKRATIO*p_src2.rows)).setTo(255);
        getHomography(tp_src1, mask, p_src2, mask2, H, false);
        Mat tmp;
        // 将待拼接图像变换到src1所在的坐标系
        warpPerspective(p_src2, tmp, H, Size(tp_src1.cols, tp_src1.rows));
//        imshow("tmp after warp perspective", tmp);
        // 图像融合
        for (int m = 0; m < tmp.rows; m++)
        {
            uchar* p_tp_src1 = tp_src1.ptr<uchar>(m);
            uchar* p_tmp = tmp.ptr<uchar>(m);
            for (int n = 0; n < tmp.cols * 3; n += 3)
            {
                if (p_tp_src1[n] || p_tp_src1[n + 1] || p_tp_src1[n + 2])
                {
                    if (p_tmp[n] || p_tmp[n+1] || p_tmp[n+2])
                    {
//                        int dis = n / 3 - extraCol;
                        int dis = tp_src1.cols - n / 3;
                        if (dis < 200 && dis > 0)
                        {
                            double weight = 1.0 * dis / 200;
                            p_tmp[n] = p_tp_src1[n] * weight + p_tmp[n] * (1 - weight);
                            p_tmp[n + 1] = p_tp_src1[n + 1] * weight + p_tmp[n + 1] * (1 - weight);
                            p_tmp[n + 2] = p_tp_src1[n + 2] * weight + p_tmp[n + 2] * (1 - weight);
                        }
                        else if (dis >= 0)
                        {
                            p_tmp[n] = p_tp_src1[n];
                            p_tmp[n + 1] = p_tp_src1[n + 1];
                            p_tmp[n + 2] = p_tp_src1[n + 2];
                        }
                    }
                    else
                    {
                        p_tmp[n] = p_tp_src1[n];
                        p_tmp[n + 1] = p_tp_src1[n + 1];
                        p_tmp[n + 2] = p_tp_src1[n + 2];
                    }
                }
            }
        }
        imshow("tmp after blender", tmp);
        waitKey(0);
        Mat tmpROI;
        removeBlackSide(tmp, tmpROI);
        p_src1 = tmpROI;
    }
    dst = p_src1.clone();
    imwrite("../output/canny/pano.bmp", dst);
}

Point sp(-1,-1);
int* pos;
static void on_mouse_click(int event, int x, int y, int flags, void* userdata){
    Mat image = *(Mat*)userdata;
    if(event == EVENT_LBUTTONDOWN) {
        // 左键按下
        sp.x = x;
        sp.y = y;
        *pos = x;
        std::cout << "start point: " << sp << ", pos value:" << *pos << std::endl;
    }
}

// 手动选取柱面反投影的中心点位置
void ImageStitcher::chooseCenterPoint(Mat &src, int* centerX) {
    pos = centerX;
    namedWindow("choose center point", WINDOW_AUTOSIZE);
    Mat dst = src.clone();
    // 画一条垂直方向上的中线，指导选点
    line(dst, Point(0, dst.rows/2),Point(dst.cols-1, dst.rows/2), Scalar(0,255,0));
    imshow("choose center point", dst);
    setMouseCallback("choose center point", on_mouse_click, &dst);
//    cout << "choose center pos: (" << sp.x << ", " << sp.y << ")" <<endl;
    // 选取后按任意键一次以结束调用
    waitKey(0);
}

Point p1(-1,-1), p2(-1,-1);
vector<pair<Point, Point>> lines;
Mat pointTmp;
static void on_mouse_click2(int event, int x, int y, int flags, void* userdata){
    Mat image = *(Mat*)userdata;
    Mat tmp;
    pointTmp.copyTo(tmp);
    if(event == EVENT_LBUTTONDOWN) {
        std::cout << "click point: (" << x << "," << y << ")" << std::endl;
        // 左键按下
        if(p1.x == -1 && p1.y == -1){
            p1.x = x;
            p1.y = y;
            circle(tmp, Point(x,y), 5, Scalar(255,0,0), 2, LINE_AA, 0);
        }else if(p2.x == -1 && p2.y==-1){
            p2.x = x;
            p2.y = y;
            circle(tmp, Point(x,y), 5, Scalar(0,0,255), 2, LINE_AA, 0);
            line(tmp, p1, p2, Scalar(0,255,0), 2, LINE_8,0);
            lines.emplace_back(make_pair(p1, p2));
            cout << "Push to vector :" << p1 << ", " << p2 << endl;
            p1.x = p1.y = p2.x = p2.y = -1;
        }
        tmp.copyTo(pointTmp);
        imshow("Point collection", tmp);
    }
}

void ImageStitcher::collectLineSegments(cv::Mat &src, std::vector<std::pair<cv::Point, cv::Point>> &pointPairs) {
    Mat img = src.clone();
    lines.clear();
    p1.x = p1.y = p2.x = p2.y = -1;
    namedWindow("Point collection", WINDOW_AUTOSIZE);
    imshow("Point collection", img);
    img.copyTo(pointTmp);
    setMouseCallback("Point collection", on_mouse_click2, &img);
    cout << "选取点对构造直线方程，计算距离" << endl;
    waitKey(0);
    imwrite("../output/canny/lineSegmentPairs.bmp", pointTmp);
    pointPairs.assign(lines.begin(), lines.end());
}
