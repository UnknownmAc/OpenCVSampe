//
//  main.cpp
//  PoseDetector2
//
//  Created by niskumar on 16/12/15.
//  Copyright (c) 2015 niskumar. All rights reserved.
//

#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#define WINDOW_NAME "PoseDetectorWindow"
#define WINDOW_NAME_2 "PoseDetectorWindow2"

void drawOptFlowMap (const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at< cv::Point2f>(y, x);
            
            line(cflowmap, cvPoint(x,y), cvPoint(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            
            circle(cflowmap, cvPoint(cvRound(x+fxy.x), cvRound(y+fxy.y)), 2, color, -1);
        }
}

void ApplyGaussianBlur(const cv::Mat src, cv::Mat &dst)
{
    for (int i=1; i<31; i=i+2)
    {
        // smooth the image in the "src" and save it to "dst"
        //cv::blur(src, dst, cvSize(3,3));
        
        // Gaussian smoothing
        cv::GaussianBlur( src, dst, cvSize( 3, 3 ), 0, 0 );
    }
}


int main(int argc, const char * argv[])
{
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
    
    char key;
    cvNamedWindow("Camera_Output", 1);    //Create window
    
    IplImage* firstFrame = NULL;
    IplImage* lastFrame = NULL;
    
    std::vector<IplImage*> frameVector;
    
    while(1){ //Create infinte loop for live streaming
        
        IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
        cvShowImage("Camera_Output", frame);   //Show image frames on created window
        
        frameVector.push_back(cvCloneImage(frame));
        
        key = cvWaitKey(30);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    
#if 0 //blur chalaana h toh
    std::vector<cv::Mat> frameVectorMat;
    for(int i = 0; i < frameVector.size(); i++)
        frameVectorMat.push_back(frameVector[i]);
    
    for(int i=0;i < 1;i++)
    {
        cv::Mat dst = frameVector[i];
        ApplyGaussianBlur(frameVectorMat[i], dst);
        
        
        IplImage tempImage = dst;
        
        cvShowImage("Blur waala output", &tempImage);
        
        frameVector[i] = &tempImage;
    }
#endif

    cvReleaseCapture(&capture); //Release capture.

    lastFrame = frameVector[frameVector.size() - 1];
    
    std::vector<cv::Mat> flowVector;
    
    for(int i=0; i < 1; i++)
    {
        firstFrame = frameVector[i];
        
        IplImage *testInputGray = cvCreateImage(cvGetSize(lastFrame), IPL_DEPTH_8U, 1);
        
        cvCvtColor(firstFrame, testInputGray, CV_RGB2GRAY);
        IplImage* prevEdgeImage = cvCloneImage(testInputGray);
        //cvCanny(testInputGray, prevEdgeImage, 0, 200);
        

        cvCvtColor(lastFrame, testInputGray, CV_RGB2GRAY);
        IplImage* nextEdgeImage = cvCloneImage(testInputGray);
        //cvCanny(testInputGray, nextEdgeImage, 0, 200);
        
        cv::Mat prevEdgeImageMat = prevEdgeImage;
        cv::Mat nextEdgeImageMat = nextEdgeImage;
        
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prevEdgeImageMat, nextEdgeImageMat, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      
        flowVector.push_back(flow);
        
        cvReleaseImage(&testInputGray);
#if 0
        cv::Mat cflow;
        cvtColor(prevEdgeImageMat, cflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 16, CV_RGB(0, 255, 0));
#endif
    }
    
    if(flowVector.size() > 0)
    {
        float displacementMatrix[flowVector[0].rows][flowVector[0].cols];
        
        for(int i = 0; i < flowVector[0].rows; i++)
            for(int j = 0; j < flowVector[0].cols; j++)
                displacementMatrix[i][j] = 0.0f;
        
        float displacementMaxm = 0.0f;
        
        for(int i = 0; i < flowVector.size(); i++)
        {
            cv::Mat flow = flowVector[i];
            
            for(int y = 0; y < flow.rows; y += 1)
                for(int x = 0; x < flow.cols; x += 1)
                {
                    const cv::Point2f& fxy = flow.at< cv::Point2f>(y, x);
                    
                    cv::Point init_pos(x, y);
                    cv::Point final_pos(cvRound(x+fxy.x), cvRound(y+fxy.y));
                    
                    cv::Point diff = final_pos - init_pos;
                    
                    displacementMatrix[y][x] += 100*(diff.x*diff.x + diff.y*diff.y);
                }
            
        }
        
        for(int i = 0; i < flowVector[0].rows; i++)
            for(int j = 0; j < flowVector[0].cols; j++)
            {
                displacementMatrix[i][j] /= flowVector.size();
            }
        
        for(int i = 0; i < flowVector[0].rows; i++)
            for(int j = 0; j < flowVector[0].cols; j++)
            {
                if(displacementMatrix[i][j] > displacementMaxm)
                    displacementMaxm = displacementMatrix[i][j];
            }

        IplImage* tempImage = cvCreateImage(cvGetSize(lastFrame), IPL_DEPTH_8U, 1);
        cv::Mat depthMap = tempImage;
        
        for(int y=0;y<depthMap.rows;y++)
        {
            for(int x=0;x<depthMap.cols;x++)
            {
                
                char color = (char) (255.0 - std::fabs(((displacementMatrix[y][x] * 255.0) / (displacementMaxm))));
                // set pixel
                depthMap.at<char>(cvPoint(x,y)) = color;
            }
        }
#if 0
        cv::Mat source = frameVector[0];
        cv::Point2f src_center(source.cols/2.0F, source.rows/2.0F);
        cv::Mat rot_mat = getRotationMatrix2D(src_center, 30, 1.0);
        cv::Mat dst;
        warpAffine(source, dst, rot_mat, source.size());
#endif
        
#if 1
        cvNamedWindow(WINDOW_NAME_2);
        imshow(WINDOW_NAME_2, depthMap);
#endif
        
     //   cvNamedWindow(WINDOW_NAME_2);
      //  imshow(WINDOW_NAME_2, dst);
    }
    
    cvWaitKey();
    
    return 0;
}
