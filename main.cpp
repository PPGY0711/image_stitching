/*
 * @author: Guanyan Peng
 * @create date: 2022/5/5
 * @update date: 2022/5/14
 * @project: image stitching and object measurement
 * @ref1: example given by official opencv doc
 * @link: https://docs.opencv.org/4.0.0/d9/dd8/samples_2cpp_2stitching_detailed_8cpp-example.html
 * @ref2: ORBAlgorithm和ImageStitcher类的实现参考：特征点匹配应用——图像拼接的原理与基于OpenCV的实现
 * @link: https://blog.csdn.net/lhanchao/article/details/52974129
 * @ref3: 柱面正投影和反投影
 * @link: https://zh.wikipedia.org/zh-hans/%E6%9F%B1%E9%9D%A2%E6%8A%95%E5%BD%B1%E6%B3%95
 * @ref4: BEBLID描述符
 * @link: https://blog.csdn.net/Small_Munich/article/details/113950115
 */

#include <iostream>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include "image_stitch.h"
using namespace cv;
using namespace std;

#define SQUARE(x) ((x)*(x))
#define CENTER_DIS 0.5
#define SharpenFactor 1     //锐化系数
#define SharpenThs 30       //锐化阈值
//vector<Mat> preprocess;
Point lp(-1,-1),rp(-1,-1);
Mat curImg;
string winName;

static void validCalibration(vector<vector<Point3f>>& objPoints, vector<vector<Point2f>>& imgPoints, vector<Mat>& rvecsMat,
                             vector<Mat>& tvecsMat, Mat& cameraMatrix, Mat& distCoeffs){
    cout << "开始评价标定结果..." << endl;
    for(int i = 0; i < objPoints.size(); i++){
        double total_err = 0.0;            // 所有图像的平均误差的总和
        double err = 0.0;                  // 每幅图像的平均误差
        double totalErr = 0.0;
        double totalPoints = 0.0;
        vector<Point2f> image_points_pro;     // 保存重新计算得到的投影点
        // 通过得到的摄像机内外参数，对角点的空间三维坐标进行重新投影计算
        projectPoints(objPoints[i], rvecsMat[0], tvecsMat[0], cameraMatrix, distCoeffs, image_points_pro);
        err = norm(Mat(imgPoints[i]), Mat(image_points_pro), NORM_L2);
        totalErr += err * err;
        totalPoints += objPoints[i].size();
        err /= objPoints[i].size();
        total_err += err;
        // 重投影误差2：0.149617像素
        // 重投影误差3：0.00382504像素
        cout << "重投影误差2：" << sqrt(totalErr / totalPoints) << "像素" << endl;
        cout << "重投影误差3：" << total_err << "像素" << endl;
        cout << "x = " << cameraMatrix.at<double>(0, 2) << endl;
        cout << "y = " << cameraMatrix.at<double>(1, 2) << endl;
        cout << cameraMatrix << endl << endl;
    }
}

vector<Mat> calibrateAndUndistortImages(Mat& src, vector<Mat>& imgs, double* pixelDis){
    int w = 45, h = 34;
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    vector<KeyPoint> keypoints;
    // 设置SimpleBlobDetector参数，用于检测图像中的圆形
    SimpleBlobDetector::Params params;

    // 改变阈值
    params.minThreshold = 0;
    params.maxThreshold = 255;

    // 根据面积过滤
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 300;

    // 根据Circularity过滤
    params.filterByCircularity = false;
    params.minCircularity = 0.00001;

    // 根据Convexity过滤
    params.filterByConvexity = false;
    params.minConvexity = 0.1;

    // 根据Inertia过滤
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;
#if CV_MAJOR_VERSION < 3   // 如果你使用的是opencv2
    // 使用默认参数设置检测器
    SimpleBlobDetector detector();
    // 您可以这样使用检测器
    detector.detect( im, keypoints);
#else
    // 使用参数设置检测器
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    // SimpleBlobDetector::create 创建一个智能指针
    // 所以你需要使用arrow(->)而不是dot(.)
    detector->detect(gray, keypoints);
    Mat im_with_keypoints;
    drawKeypoints(gray, keypoints, im_with_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // 结果显示
//    imshow("keypoints", im_with_keypoints);
    imwrite("../calib/calib_keypoints.bmp", im_with_keypoints);
//    waitKey(0);
#endif
    // 圆心检测（角点提取）
    CirclesGridFinderParameters parameters;
    parameters.gridType = CirclesGridFinderParameters::SYMMETRIC_GRID;
    vector<Point2f> corners;
    Size board = Size(w, h);
    cout << board.height << ", " << board.width << endl;
    bool found = findCirclesGrid(gray, Size(w, h), corners, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, detector, parameters);
    if(found){
        Mat tmpSrc = src.clone();
        drawChessboardCorners(tmpSrc, board, corners, found);
//        imshow("calibrate", tmpSrc);
        imwrite("../calib/drawCorners.bmp", tmpSrc);
        // 观察了一下原corners是按行存储的，并且是从大到小，这里倒序一下和三维坐标对应
        for(int i = 0, j = corners.size()-1; i < j; ++i,--j){
            Point2f tmp = corners[i];
            corners[i].x = corners[j].x;
            corners[i].y = corners[j].y;
            corners[j].x = tmp.x;
            corners[j].y = tmp.y;
        }
        // 创建角点的三维坐标
        float square_size = 0.5;
        int i,j;
        vector<Point3f> _objPoints;
        vector<double> intervals; // 垂直和水平方向上的像素距离
        for(i = 0; i < h; i++){
            for(j = 0; j < w; j++){
                cout << "(" << corners[i*w+j].x << ", " << corners[i*w+j].y << "); ";
                if(j < w-1){
                    intervals.emplace_back(sqrt(SQUARE(corners[i*w+j].x- corners[i*w+j+1].x) + SQUARE(corners[i*w+j].y- corners[i*w+j+1].y)));
                }
                if(i < h-1){
                    intervals.emplace_back(sqrt(SQUARE(corners[(i+1)*w+j].x- corners[i*w+j].x) + SQUARE(corners[(i+1)*w+j].y- corners[i*w+j].y)));
                }
                _objPoints.push_back(Point3f(i*square_size,j*square_size,0));
            }
            cout << endl;
        }

        // 由于圆心距为0.5，这里计算出平均的像素距离供后续测尺寸
        double totalDis = accumulate(intervals.begin(), intervals.end(), 0.0);
        *pixelDis = CENTER_DIS/ (totalDis/intervals.size());

        // 准备进行相机标定，用于对待拼接进行正畸
        Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));        // 摄像机内参数矩阵
        Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));          // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
        vector<Mat> rvecsMat;                                          // 存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
        vector<Mat> tvecsMat;                                          // 存放所有图像的平移向量，每一副图像的平移向量为一个mat
        string outputFileName;
        Size imageSize(src.cols, src.rows);
        cout << "图片宽：" << imageSize.width << ", 高：" << imageSize.height << endl;
        /* 运行标定函数 */
        vector<vector<Point3f>> objPoints;
        objPoints.push_back(_objPoints);
        vector<vector<Point2f>> imgPoints;
        imgPoints.push_back(corners);
        double rms = calibrateCamera(objPoints, imgPoints, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3);
        cout << "标定完成，RMS：" << rms << "像素" << endl << endl;
        validCalibration(objPoints, imgPoints, rvecsMat, tvecsMat, cameraMatrix, distCoeffs);
        // 正畸
        Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize);
        cout << "得到新内参： " << endl;
        cout << cameraMatrix << endl;
        validCalibration(objPoints, imgPoints, rvecsMat, tvecsMat, newCameraMatrix, distCoeffs);
        vector<Mat> undistImgs;
        int count = 29;
        for(Mat& mat: imgs){
            Mat dst = mat.clone();
            undistort(mat, dst, cameraMatrix, distCoeffs, newCameraMatrix);
            undistImgs.push_back(dst);
//            imshow("undist " + to_string(count), dst);
//            imshow("origin " + to_string(count), mat);
            imwrite("../img/undist/undist_"+to_string(count)+".bmp", dst);
            count++;
        }
        return undistImgs;
    }else{
        cout << "Error: 未检测到角点！" << endl;
        return imgs;
    }
}

static void on_mouse_draw_line(int event, int x, int y, int flags, void* userdata){
    Mat image = *(Mat*)userdata;
    Mat tmp;
    curImg.copyTo(tmp);
    if(event == EVENT_LBUTTONDOWN) {
//        std::cout << "click point: (" << x << "," << y << ")" << std::endl;
        // 左键按下
        if(lp.x == -1 && lp.y == -1){
            lp.x = x;
            lp.y = y;
        }else if(rp.x == -1 && rp.y==-1){
            rp.x = x;
            rp.y = y;
            line(tmp, lp, rp, Scalar(255,255,255), 1, LINE_AA,0);
            lp.x = lp.y = rp.x = rp.y = -1;
        }
        tmp.copyTo(curImg);
        imshow(winName, tmp);
    }
}

void ImproveUSM(const Mat& src, Mat &BlurImg, int Threshold, float Factor, string& path)
{
//    cout << "ImproveUSM" << ", " << src.channels() << endl;
    GaussianBlur(src, BlurImg, Size(9, 9), 0, 0);
    Mat DiffMask, dst;
    DiffMask = Mat::zeros(src.rows, src.cols, src.type());
    if (src.channels() == 1)  //灰度单通道
    {
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                int Value_diff = abs(src.at<uchar>(i, j) - BlurImg.at<uchar>(i, j));
                if (Value_diff < Threshold) //小于阈值说明非边缘，不需要锐化
                    DiffMask.at<uchar>(i, j) = 1;
                else
                    DiffMask.at<uchar>(i, j) = 0;
            }
        }
    }
    else if (src.channels() == 3) //三通道BGR
    {
        int Value_diff[3] = { 0 };
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Value_diff[k] = abs(src.at<Vec3b>(i, j)[k] - BlurImg.at<Vec3b>(i, j)[k]);

                    if (Value_diff[k] < Threshold) //小于阈值说明非边缘，不需要锐化
                        DiffMask.at<Vec3b>(i, j)[k] = 1;
                    else
                        DiffMask.at<uchar>(i, j) = 0;
                }
            }
        }
    }
    addWeighted(src, 1 + Factor, BlurImg, -Factor, 0, dst); //将两幅图按权重系数融合,dst = src + Factor（src - BlurImg）
    src.copyTo(dst, DiffMask);  //将src中DiffMask对应的非0部分复制到dst中
    imshow("ImproveUSM", dst);
    dst.copyTo(curImg);
    cout << "save sharpened dst" << endl;
    imwrite(path, curImg);
}

void preprocessImgs(vector<Mat>& imgs){
    // 预处理：双边滤波+手动描边
    vector<Mat> ret((int)imgs.size());
    int count = 0, no=29;
    for(Mat& mat:imgs){
        // 这里不能转成gray不然画图受影响，先把BGR转到YUV再滤波
        Mat sharpened;
        string path = "../img/undist/sharpened_"+to_string(no)+".bmp";
        ImproveUSM(mat, sharpened, SharpenThs, SharpenFactor, path);
        sharpened = imread(path);
        Mat yuv;
        cvtColor(sharpened, yuv, COLOR_BGR2YCrCb);
        vector<Mat> channels;
        split(yuv, channels);
        Mat filtered;
        bilateralFilter(channels[0], filtered, 5, 20, 20);
        channels[0] = filtered;
        merge(channels, filtered);
        cvtColor(filtered, filtered, COLOR_YCrCb2BGR);
//        ImproveUSM(filtered, sharpened, SharpenThs, SharpenFactor, path);
        winName = "draw lines " + to_string(count);
        namedWindow(winName, WINDOW_AUTOSIZE);
        filtered.copyTo(curImg);
        imshow(winName, filtered);
        setMouseCallback(winName, on_mouse_draw_line, &filtered);
        waitKey(0);
        imwrite("../img/undist/preprocess_"+to_string(no++)+".bmp", curImg);
        count++;
    }
}

void customStitchImages(ImageStitcher& stitcher, vector<Mat>& undistImgs, Mat& dst, Mat& stretch){
    stitcher.stitchImages(undistImgs, dst, (int)undistImgs.size()/2);
    imshow("stitcher", dst);
    int centerX = dst.cols/2, centerY = dst.rows/2;
    int chooseX = -1;
    stitcher.chooseCenterPoint(dst, &chooseX);
    cout << centerX << ", " << centerY << "," << chooseX <<  endl;
    // 让选择的中心点成为中心，图像需要平移
    int displacement = centerX-chooseX;
    int newWidth = (abs(displacement)+centerX)*2;
    Mat adjust = Mat::zeros(dst.rows, newWidth, dst.type());
//    cout << adjust.size() << endl;
//    cout << newWidth/2-chooseX << endl;
    // 图像平移或者用ROI
    // 得到中心点坐标之后，将中心点移到视野中央
    Mat tROI = adjust(Rect(displacement>0?newWidth/2-chooseX-1:0, 0, dst.cols, dst.rows));
    dst.convertTo(tROI, tROI.type());
    Mat tmpAd = adjust.clone();
    rectangle(tmpAd, Rect(displacement>0?newWidth/2-chooseX:0, 0, dst.cols, dst.rows), Scalar(0,255,0),2,8,0);
    line(tmpAd, Point(0,adjust.rows/2), Point(adjust.cols-1,adjust.rows/2), Scalar(255,0,0));
    line(tmpAd, Point(adjust.cols/2,0), Point(adjust.cols/2,adjust.rows-1), Scalar(0,0,255));
    line(tmpAd, Point(chooseX,0), Point(chooseX,adjust.rows-1), Scalar(255,0,255));
    line(tmpAd, Point(adjust.cols-chooseX-1,0), Point(adjust.cols-chooseX-1,adjust.rows-1), Scalar(255,0,255));
    imshow("adjust", tmpAd);
    waitKey(0);
    Mat linear;
    stitcher.reverseCylinderProjection(adjust, linear);
    stitcher.removeBlackSide(linear, linear);
    imshow("after lined", linear);
    imwrite("../output/canny/stretch.bmp", linear);
    imwrite("../output/canny/adjust.bmp", tmpAd);
    stretch = linear.clone();
}

void stitchImages(Mat& dst){
    string img_name;
    vector<Mat> marked;
    ifstream in("../path2.txt");
    while(getline(in, img_name))
    {
        Mat img = imread(img_name);
        marked.push_back(img);
    }
    // 31-33图片正畸后拼接
    Mat tmpDst, stretch;
    ImageStitcher stitcher;
    stitcher.count = 0;
    customStitchImages(stitcher, marked, tmpDst, stretch);
    stretch.copyTo(dst);
}

void testPrecision(Mat& stretch, double pixelDis){
    // 在最终拉直的图片上选点做直线，计算直线方程，设置鼠标和键盘的回调
    Mat backUp = stretch.clone();
    vector<pair<Point,Point>> pairs;
//     在主线程写回调总是报SIGBUS错误，改到了类里面写函数解决了，很玄学...
    ImageStitcher stitcher;
    stitcher.collectLineSegments(backUp, pairs);
    vector<double> vpos;
    for(int i = 0; i < pairs.size(); i++){
        cout << pairs[i].first << ", " << pairs[i].second << endl;
        vpos.emplace_back((static_cast<double>(pairs[i].first.x)+static_cast<double>(pairs[i].second.x))/2);
    }
    // 步长为1计算0.5cm距离的精度
    double len1 = 0.5*10, len2 = 1.0*10;
    vector<double> error1, error2;
    double tS1, tS2, tE1, tE2; // 像素距离和，误差和
    tS1 = tS2 = tE1 = tE2 = 0;
    for(int i = 0; i < vpos.size()-1; i++){
        double pixelSub = vpos[i+1]-vpos[i];
        double disMeasure = pixelSub*pixelDis;
        tS1 += disMeasure;
        error1.emplace_back(disMeasure-len1);
        cout << "第" << i << "个1.0间隔的长度测量值为：" << disMeasure << "mm，测量误差为: " << disMeasure-len1 << "mm." << endl;
    }
    cout << tS1 << endl;
    tE1 = (tS1-(len1*(vpos.size()-1)))/(vpos.size()-1);
    cout << "测量0.5cm的总误差" << tE1 << endl;
    // 步长为2计算1.0cm距离的精度
    for(int i = 0; i < vpos.size()-2; i++){
        double pixelSub = vpos[i+2]-vpos[i];
        double disMeasure = pixelSub*pixelDis;
        tS2 += disMeasure;
        error2.emplace_back(disMeasure-len2);
        cout << "第" << i << "个1.0间隔的长度测量值为：" << disMeasure << "mm，测量误差为: " << disMeasure-len2 << "mm." << endl;
    }
    tE2 = (tS2-(len2*(vpos.size()-2)))/(vpos.size()-2);
    cout << "测量1.0cm的总误差" << tE2 << endl;
}

Point rulerLeft(-1,-1), rulerRight(-1,-1);
pair<Point,Point> rulerLine;
Mat tmpImg;
string tmpWinName;
void on_mouse_choose_points(int event, int x, int y, int flags, void* userdata){
    Mat image = *(Mat*)userdata;
    Mat tmp;
    tmpImg.copyTo(tmp);
    if(event == EVENT_LBUTTONDOWN) {
//        std::cout << "click point: (" << x << "," << y << ")" << std::endl;
        // 左键按下
        if(rulerLeft.x == -1 && rulerLeft.y == -1){
            rulerLeft.x = x;
            rulerLeft.y = y;
            circle(tmp, rulerLeft, 2, Scalar(0,255,0), 1, LINE_AA, 0);
        }else if(rulerRight.x == -1 && rulerRight.y==-1){
            rulerRight.x = x;
            rulerRight.y = y;
            circle(tmp, rulerLeft, 2, Scalar(0,0,255), 1, LINE_AA, 0);
            line(tmp, rulerLeft, rulerRight, Scalar(255,0,0), 1, LINE_AA,0);
            rulerLine = make_pair(rulerLeft, rulerRight);
            rulerLeft.x = rulerLeft.y = rulerRight.x = rulerRight.y = -1;
        }
        tmp.copyTo(tmpImg);
        imshow(tmpWinName, tmp);
    }
}

static void my_rotate(const Mat& image, Mat& dst, int lineCenterHeight, double angle){
    Mat tmpDst, M;
    int w = image.cols, h = image.rows;
    M = getRotationMatrix2D(Point(w/2,h/2), angle, 1.0); //scale参数可以控制放缩倍数
    double cos = abs(M.at<double>(0,0)), sin = abs(M.at<double>(0,1));
    int nw = cos*w+sin*h, nh = sin*w+cos*h;
    // 求取旋转之后图片的实际大小（视野里如果包含所有图像的话应该是多大）
    // 宽高有变化，中心会偏移，这里是调整中心的位置
    M.at<double>(0,2) += (nw/2-w/2);
    M.at<double>(1,2) += (nh/2-h/2);
    warpAffine(image, tmpDst, M, Size(nw,nh), INTER_LINEAR, 0, Scalar(0,0,0));
    imshow("rotate", tmpDst);
    tmpDst.copyTo(dst);
//    int transy = lineCenterHeight-h/2;
//    cout << "transY: " << transy << endl;
//    Mat transDst;
//    Mat M3 = (cv::Mat_<double>(2, 3) << 1.0, 0, 0, 0, 1, -transy);
//    cout << M3 << endl;
//    warpAffine(tmpDst, transDst, M3, Size(tmpDst.cols, tmpDst.rows));
//    imshow("transy", transDst);
//    transDst.copyTo(dst);
}

static void calLineExp(Point p1, Point p2, double* angle){
    // 计算直线的一般式方程Ax+By+C=0及直线与x轴正方形的夹角（顺时针）
    /*
         A = Y2 - Y1
         B = X1 - X2
         C = X2*Y1 - X1*Y2
    */
    Vec3d coef;
    Point2f p1f = static_cast<Point2f>(p1);
    Point2f p2f = static_cast<Point2f>(p2);
    coef[0] = p2f.y-p1f.y;
    coef[1] = p1f.x-p2f.x;
    coef[2] = p2f.x*p1f.y-p2f.y*p1f.x;
    if(coef[0] == 0) *angle = 0;
    else if(coef[1] == 0) *angle = PI/2;
    else{
        *angle=atan((-coef[0])/(-coef[1])); // -PI/2 ~ PI/2
    }
}

void readOriginImgs(vector<Mat>& imgs){
    // 读取本地图片并存到向量中
    string img_name;
    ifstream fin("../path.txt");
    int seq = 1;
    while(getline(fin, img_name))
    {
        Mat img = imread(img_name);
        if(img_name == "../img/origin/30.bmp"){
            // 30那一张需要单独调一下亮度，稍微变暗（白色部分调低一些）
            vector<Mat> channels;
            Mat yuv;
            cvtColor(img, yuv, COLOR_BGR2YCrCb);
            split(yuv, channels);
            Mat image = channels[0].clone();
            for(int row = 0; row < image.rows; ++row){
                uchar* currentRow = image.ptr<uchar>(row);
                for(int col = 0; col < image.cols; ++col){
                    // 原来是存的uchar类型，这里会进行隐式转换
                    int pv = image.at<uchar>(row,col);
                    if(pv >= 250){
                        image.at<uchar>(row,col) = pv-35;
                    }else
                        image.at<uchar>(row,col) = saturate_cast<uchar>(pv-25);
                }
            }
            channels[0] = image;
            Mat darken;
            merge(channels, yuv);
            cvtColor(yuv, darken, COLOR_YCrCb2BGR);
            darken.copyTo(img);
        }
        tmpWinName = "Choose Ruler Border";
        // 将所有图片中的尺子边缘保持在水平，选出尺子两端的点计算直线方程和倾角，通过旋转变化得到新的图
        namedWindow(tmpWinName, WINDOW_AUTOSIZE);
        img.copyTo(tmpImg);
        setMouseCallback(tmpWinName, on_mouse_choose_points, &tmpImg);
        imshow(tmpWinName, tmpImg);
        waitKey(0);
        // 根据Point点对计算
        double angle = 0;
        calLineExp(rulerLine.first, rulerLine.second, &angle);
        cout << "i: " << seq << ", rotate angle: " << angle << endl;
        if(angle != 0){
            // 需要逆时针/顺时针旋转angle角度
            Mat roMat;
            int ch = (rulerLine.first.y+rulerLine.second.y)>>1;
            my_rotate(img, roMat, ch, -angle*180/PI);
            roMat.copyTo(img);
        }
        imgs.push_back(img);
    }
    int num_images = imgs.size();    //图像数量
    cout<<"图像读取完毕，数量为"<<num_images<<endl;
}

int main(int argc, char** argv){
    double pixelDis = 0.0146816;

//    vector<Mat> imgs;
//    readOriginImgs(imgs);
//    // 相机标定并对图像做正畸
//    Mat calib = imread("../calib/big.bmp");
//    vector<Mat> undistImgs = calibrateAndUndistortImages(calib, imgs, &pixelDis);
//    cout << "平均像素距离：" << pixelDis << "mm" << endl;
////    一次性的过程，预处理正畸之后的图片，人工画出标志线
//    preprocessImgs(undistImgs);

    Mat stretch;
    stitchImages(stretch);
    testPrecision(stretch, pixelDis);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

/*
 * 1.为什么30和31不能拼：重叠部分太少了，因此要计算一下如果要能够拼接，至少需要有30%的重叠，这样的话镜头的视野只能移动多长的距离？
 * 2.单独拼接29-30，31-33，需要实现图片从柱状投影之后的拉直效果done
 * 3.尝试Canny先做预处理之后，用边缘检测的图去做特征点匹配+拼接
 * */