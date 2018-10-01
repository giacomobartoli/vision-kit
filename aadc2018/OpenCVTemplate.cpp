/*********************************************************************
Copyright (c) 2018
Audi Autonomous Driving Cup. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3.  All advertising materials mentioning features or use of this software must display the following acknowledgement: ?This product includes software developed by the Audi AG and its contributors for Audi Autonomous Driving Cup.?
4.  Neither the name of Audi nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AUDI AG AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL AUDI AG OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************/
#define PI 3.1415926
#include "stdafx.h"
#include "OpenCVTemplate.h"
#include "ADTF3_OpenCV_helper.h"
#include "playground.h"

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdio>
//#include <ctime>
#include "stdafx.h"
#include "OpenCVTemplate.h"
#include "ADTF3_OpenCV_helper.h"
#include <thread>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <regex>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;



ADTF_TRIGGER_FUNCTION_FILTER_PLUGIN(CID_COPENCVTEMPLATE_DATA_TRIGGERED_FILTER,
                                    "OpenCV Template",
                                    cOpenCVTemplate,
                                    adtf::filter::pin_trigger({ "in" }));

cOpenCVTemplate::cOpenCVTemplate()
{
    //Register Properties
    RegisterPropertyVariable("ROIOffsetX [Pixel]",      m_ROIOffsetX);
    RegisterPropertyVariable("ROIOffsetY [Pixel]",      m_ROIOffsetY);
    RegisterPropertyVariable("ROIWidth [Pixel]",        m_ROIWidth);
    RegisterPropertyVariable("ROIHeight [Pixel]",       m_ROIHeight);
    RegisterPropertyVariable("detectionLines",  m_detectionLines);
    RegisterPropertyVariable("minLineWidth [Pixel]", m_minLineWidth);
    RegisterPropertyVariable("maxLineWidth [Pixel]", m_maxLineWidth);
    RegisterPropertyVariable("minLineContrast", m_minLineContrast);
    RegisterPropertyVariable("thresholdImageBinarization", m_thresholdImageBinarization);
    RegisterPropertyVariable("left extreme of horizontal part of search area for right lane", m_horizontalLeft_RightLane);
    RegisterPropertyVariable("right extreme of horizontal part of search area for right lane", m_horizontalRight_RightLane);
    RegisterPropertyVariable("upper extreme of vertical part of search area for right lane", m_verticalUp_RightLane);
    RegisterPropertyVariable("lower extreme of vertical part of search area for right lane", m_verticalDown_RightLane);
    RegisterPropertyVariable("left extreme while searching for middle line", m_midhorizontalCompleteLeft);
    RegisterPropertyVariable("right extreme while searching for middle line", m_midhorizontalCompleteRight);
    RegisterPropertyVariable("upper extreme while searching for middle line", m_midverticalCompleteUp);
    RegisterPropertyVariable("lower extreme while searching for middle line", m_midverticalCompleteDown);
    RegisterPropertyVariable("top left X-point for bird's eye", m_topLeftX);
    RegisterPropertyVariable("top right X-point for bird's eye", m_topRightX);
    RegisterPropertyVariable("bottom left X-point for bird's eye", m_bottomLeftX);
    RegisterPropertyVariable("bottom right X-point for bird's eye", m_bottomRightX);
    RegisterPropertyVariable("top left Y-point for bird's eye", m_topLeftY);
    RegisterPropertyVariable("top right Y-point for bird's eye", m_topRightY);
    RegisterPropertyVariable("bottom left Y-point for bird's eye", m_bottomLeftY);
    RegisterPropertyVariable("bottom right Y-point for bird's eye", m_bottomRightY);
    RegisterPropertyVariable("resolution of bird's eye view X", m_destResX);
    RegisterPropertyVariable("resolution of bird's eye view Y", m_destResY);
    RegisterPropertyVariable("follow the point that is displaced by this much from lane", m_displacement);
    RegisterPropertyVariable("threshold for going off road", m_OffRoadThreshold);
    RegisterPropertyVariable("threshold for going out of lane", m_OutOfLaneThreshold);
    RegisterPropertyVariable("which lane are you in when you start driving?", m_currentLane);

    //create and set inital input format type
    m_InPinVideoFormat.m_strFormatName = ADTF_IMAGE_FORMAT(RGB_24);
    adtf::ucom::object_ptr<IStreamType> pTypeInput = adtf::ucom::make_object_ptr<cStreamType>(stream_meta_type_image());
    set_stream_type_image_format(*pTypeInput, m_InPinVideoFormat);
    //Register input pin
    Register(m_oReaderVideo, "in", pTypeInput);

    //Register output pins
    adtf::ucom::object_ptr<IStreamType> pTypeOutput = adtf::ucom::make_object_ptr<cStreamType>(stream_meta_type_image());
    set_stream_type_image_format(*pTypeOutput, m_OutPinVideoFormat);
    Register(m_oWriterVideo, "out", pTypeOutput);

    //register callback for type changes
    m_oReaderVideo.SetAcceptTypeCallback([this](const adtf::ucom::ant::iobject_ptr<const adtf::streaming::ant::IStreamType>& pType) -> tResult
    {
        return ChangeType(m_oReaderVideo, *pType.Get());
    });

    //Register output pins for bird eye
    adtf::ucom::object_ptr<IStreamType> pTypeOutputBirdEye = adtf::ucom::make_object_ptr<cStreamType>(stream_meta_type_image());
    set_stream_type_image_format(*pTypeOutputBirdEye, m_OutPinVideoFormatBirdEye);
    Register(m_oWriterVideoBirdEye, "out_bird_eye", pTypeOutputBirdEye);

    //Register output pins for normal video
    adtf::ucom::object_ptr<IStreamType> pTypeOutputNormal = adtf::ucom::make_object_ptr<cStreamType>(stream_meta_type_image());
    set_stream_type_image_format(*pTypeOutputNormal, m_OutPinVideoFormatNormal);
    Register(m_oWriterVideoNormal, "normal", pTypeOutputNormal);

    object_ptr<IStreamType> pTypeUpdateManeuver;
    if IS_OK(adtf::mediadescription::ant::create_adtf_default_stream_type_from_service("tSignalValue", pTypeUpdateManeuver, m_UpdateManeuverFactory))
    {
        adtf_ddl::access_element::find_index(m_UpdateManeuverFactory, cString("ui32ArduinoTimestamp"), m_ddlUpdateManeuver.timeStamp);
        adtf_ddl::access_element::find_index(m_UpdateManeuverFactory, cString("f32Value"), m_ddlUpdateManeuver.value);
    }
    else
    {
        LOG_WARNING("No media description for tSignalValue found!");
    }

    Register(m_oUpdateManeuver, "Update_Maneuver" , pTypeUpdateManeuver);

    object_ptr<IStreamType> pTypeWarning;
    if IS_OK(adtf::mediadescription::ant::create_adtf_default_stream_type_from_service("tSignalValue", pTypeWarning, m_WarningFactory))
    {
        adtf_ddl::access_element::find_index(m_WarningFactory, cString("ui32ArduinoTimestamp"), m_ddlWarning.timeStamp);
        adtf_ddl::access_element::find_index(m_WarningFactory, cString("f32Value"), m_ddlWarning.value);
    }
    else
    {
        LOG_WARNING("No media description for tSignalValue found!");
    }

    Register(m_oWarning, "Warning" , pTypeWarning);

    // Set dirs variables
    ROOTDIR = "../";
    LABELS = "/home/aadc/Downloads/tf-audi/speed/vision/AADC_labels_map.pbtxt";
    GRAPH = "/home/aadc/Downloads/tf-audi/speed/vision/vision_inference_graph_18.pb";

    // Set input & output nodes names
    inputLayer = "image_tensor:0";
    outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

// Load and initialize the model from .pb file
//string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
//LOG(INFO) << "graphPath:" << graphPath;
//Status loadGraphStatus = LoadGraph(graphPath, &session);


// Load labels map from .pbtxt file
   labelsMap = std::map<int,std::string>();
   labelsMap.insert(pair<int, string>(1, "emergency_car"));
   labelsMap.insert(pair<int, string>(2, "car"));
   labelsMap.insert(pair<int, string>(3, "adult"));
   labelsMap.insert(pair<int, string>(4, "child"));

   tempBBOX_X0_FRONT = 0;
   tempBBOX_Y0_FRONT = 0;
   tempBBOX_WIDTH_FRONT = 0;
   tempBBOX_HEIGHT_FRONT = 0;

    sawAdultFront = false;
    sawChildFront = false;
    sawCarFront = false;
    sawEmergencyFront = false;

    Status loadGraphStatus = LoadGraph(GRAPH, &session);
    if (!loadGraphStatus.ok()) {
        LOG_ERROR("loadgraph(): ERROR");
    }
    else
        LOG_INFO("loadGraph(): frozen graph loaded");
}

tResult cOpenCVTemplate::Configure()
{
    //get clock object
    RETURN_IF_FAILED(_runtime->GetObject(m_pClock));
    
    RETURN_NOERROR;
}

tResult cOpenCVTemplate::ChangeType(adtf::streaming::cDynamicSampleReader& inputPin,
    const adtf::streaming::ant::IStreamType& oType)
{
    if (oType == adtf::streaming::stream_meta_type_image())
    {
        adtf::ucom::object_ptr<const adtf::streaming::IStreamType> pTypeInput;
        // get pType from input reader
        inputPin >> pTypeInput;
        adtf::streaming::get_stream_type_image_format(m_InPinVideoFormat, *pTypeInput);

        //set also output format
        adtf::streaming::get_stream_type_image_format(m_OutPinVideoFormat, *pTypeInput);
        adtf::streaming::get_stream_type_image_format(m_OutPinVideoFormatBirdEye, *pTypeInput);
        adtf::streaming::get_stream_type_image_format(m_OutPinVideoFormatNormal, *pTypeInput);
        //we always have a grayscale output image
        m_OutPinVideoFormat.m_strFormatName = ADTF_IMAGE_FORMAT(GREYSCALE_8);
        // and set pType also to samplewriter
        m_oWriterVideo << pTypeInput;
        m_oWriterVideoBirdEye << pTypeInput;
        m_oWriterVideoNormal << pTypeInput;
    }
    else
    {
        RETURN_ERROR(ERR_INVALID_TYPE);
    }

    RETURN_NOERROR;
}

tResult cOpenCVTemplate::Process(tTimeStamp tmTimeOfTrigger)
{
    object_ptr<const ISample> pReadSample;
    tSignalValue WarningSignal;

    WarningSignal.f32Value = 0;

    tempBBOX_X0_FRONT = 0;
    tempBBOX_Y0_FRONT = 0;
    tempBBOX_WIDTH_FRONT = 0 ;
    tempBBOX_HEIGHT_FRONT = 0;

    sawAdultFront = false;
    sawChildFront = false;
    sawCarFront = false;
    sawEmergencyFront = false;

    if (IS_OK(m_oReaderVideo.GetNextSample(pReadSample)))
    {
        object_ptr_shared_locked<const ISampleBuffer> pReadBuffer;

        Mat outputImageBirdEye(m_destResY, m_destResX, CV_8UC3);
        //lock read buffer
        if (IS_OK(pReadSample->Lock(pReadBuffer)))
        {
            //create a opencv matrix from the media sample buffer
            Mat src(cv::Size(m_InPinVideoFormat.m_ui32Width, m_InPinVideoFormat.m_ui32Height),
                CV_8UC3, (uchar*)pReadBuffer->GetPtr());

            Point2f src_vertices[4];
            src_vertices[0] = Point(m_topLeftX, m_topLeftY);
            src_vertices[1] = Point(m_topRightX, m_topRightY);
            src_vertices[2] = Point(m_bottomRightX, m_bottomRightY);
            src_vertices[3] = Point(m_bottomLeftX, m_bottomRightY);

            Point2f dst_vertices[4];
            dst_vertices[0] = Point(0, 0);
            dst_vertices[1] = Point(m_destResX, 0);
            dst_vertices[2] = Point(m_destResX, m_destResY);
            dst_vertices[3] = Point(0, m_destResY);

            Mat M = getPerspectiveTransform(src_vertices, dst_vertices);

            warpPerspective(src, outputImageBirdEye, M, outputImageBirdEye.size(), INTER_LINEAR);

            pReadBuffer->Unlock();
        }

        //THIS IS WHERE I GET STUFF FROM FRONT CAMERA

        Mat outputImageNormal;
        //lock read buffer
        if (IS_OK(pReadSample->Lock(pReadBuffer)))
        {
            //create a opencv matrix from the media sample buffer
            Mat src(cv::Size(m_InPinVideoFormat.m_ui32Width, m_InPinVideoFormat.m_ui32Height),
                CV_8UC3, (uchar*)pReadBuffer->GetPtr());
            outputImageNormal = src;

            //draw the circles for bird's eye configuration
//            circle(outputImageNormal, Point(m_topLeftX, m_topLeftY), 5, Scalar(0, 0, 255), -1, 8);
//            circle(outputImageNormal, Point(m_topRightX, m_topRightY), 5, Scalar(0, 0, 255), -1, 8);
//            circle(outputImageNormal, Point(m_bottomLeftX, m_bottomLeftY), 5, Scalar(0, 0, 255), -1, 8);
//            circle(outputImageNormal, Point(m_bottomRightX, m_bottomRightY), 5, Scalar(0, 0, 255), -1, 8);
            pReadBuffer->Unlock();
        }

        Mat outputImage;
        //lock read buffer
        if (IS_OK(pReadSample->Lock(pReadBuffer)))
        {
            //create a opencv matrix from the media sample buffer
            Mat inputImage(cv::Size(m_InPinVideoFormat.m_ui32Width, m_InPinVideoFormat.m_ui32Height),
                CV_8UC3, (uchar*)pReadBuffer->GetPtr());


            cvtColor(inputImage, outputImage, COLOR_BGR2GRAY);
            threshold(outputImage, outputImage, m_thresholdImageBinarization, 255, THRESH_BINARY);// Generate Binary Image

            // Detect Lines
            // here we store the pixel lines in the image where we search for lanes
            vector<tInt> detectionLines;
            // here we have all the detected line points
            vector<Point> detectedLinePoints;

            m_LaneRoiRight = cv::Rect2f(static_cast<tFloat32>(m_horizontalLeft_RightLane), static_cast<tFloat32>(m_verticalUp_RightLane), static_cast<tFloat32>(m_horizontalRight_RightLane - m_horizontalLeft_RightLane), static_cast<tFloat32>(m_verticalDown_RightLane - m_verticalUp_RightLane));
            m_LaneRoiLeft = cv::Rect2f(static_cast<tFloat32>(m_midhorizontalCompleteLeft), static_cast<tFloat32>(m_midverticalCompleteUp), static_cast<tFloat32>(m_midhorizontalCompleteRight - m_midhorizontalCompleteLeft), static_cast<tFloat32>(m_midverticalCompleteDown - m_midverticalCompleteUp));

            //calculate the detectionlines in image
            getDetectionLines(detectionLines);

            RETURN_IF_FAILED(findLinePoints(detectionLines, outputImage, detectedLinePoints));           

            cvtColor(outputImage, outputImage, COLOR_GRAY2RGB);
            // draw ROI
            rectangle(outputImage, m_LaneRoi, Scalar(255), 10, 8, 0);
            rectangle(outputImage, m_LaneRoiRight, Scalar(255), 3, 8, 0);
            rectangle(outputImage, m_LaneRoiLeft, Scalar(165, 42, 42), 3, 8, 0);
            // draw detection lines
            for (vector<tInt>::const_iterator it = detectionLines.begin(); it != detectionLines.end(); it++)
            {
                line(outputImage, Point(m_ROIOffsetX, *it), Point(m_ROIOffsetX + m_ROIWidth, *it), Scalar(255, 255, 0), 2, 8);
            }
            // show Min and Max line width which is searched.
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 2;
            int thickness = 3;
            int lengthOfLine = 100;
            {
                string textMax = "MaxLineWidth";

                int baselineMax = 0;
                Size textSizeMax = getTextSize(textMax, fontFace,
                                            fontScale, thickness, &baselineMax);
                baselineMax += thickness;

                // Place Text
                Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

                // ... and the baseline first
                rectangle(outputImage, textOrgMax + Point(textSizeMax.width, thickness),
                     textOrgMax + Point(textSizeMax.width, thickness) + Point(lengthOfLine, -m_maxLineWidth),
                     Scalar(0, 255, 0),CV_FILLED);

                // then put the text itself
                putText(outputImage, textMax, textOrgMax, fontFace, fontScale,
                        Scalar(0, 255, 0), thickness, 8);

                string textMin = "MinLineWidth";

                int baselineMin = 0;
                Size textSizeMin = getTextSize(textMax, fontFace,
                                               fontScale, thickness, &baselineMin);
                baselineMin += thickness;

                // Place Text
                Point textOrgMin(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)) + std::max(textSizeMin.height, static_cast<int>(m_minLineWidth)) +10);

                // ... and the baseline first
                rectangle(outputImage, textOrgMin + Point(textSizeMin.width, thickness),
                          textOrgMin + Point(textSizeMin.width, thickness) + Point(lengthOfLine, -m_minLineWidth),
                          Scalar(3, 242, 175), CV_FILLED);

                // then put the text itself
                putText(outputImage, textMin, textOrgMin, fontFace, fontScale,
                        Scalar(3, 242, 175), thickness, 8);
            }

            m_mostFarLeft = 9999;

            //iterate found points for drawing
            for (vector<Point>::size_type i = 0; i != detectedLinePoints.size(); i++)
            {
                circle(outputImage,
                    detectedLinePoints.at(i),
                    5,
                    Scalar(0, 0, 255),
                    -1,
                    8);

                //detect line points in the right rectangle
                if ((detectedLinePoints.at(i).x >= m_horizontalLeft_RightLane) && (detectedLinePoints.at(i).x <= m_horizontalRight_RightLane))
                {
                    if ((detectedLinePoints.at(i).y >= m_verticalUp_RightLane) && (detectedLinePoints.at(i).y <= m_verticalDown_RightLane))
                    {
                        if (detectedLinePoints.at(i).x <= m_mostFarLeft)
                        {
                            m_mostFarLeft = detectedLinePoints.at(i).x;
                            m_mostFarLeftY = detectedLinePoints.at(i).y;
                        }
                    }
                }

                //detect line points in the left rectangle
//                if ((detectedLinePoints.at(i).x >= m_midhorizontalCompleteLeft) && (detectedLinePoints.at(i).x <= m_midhorizontalCompleteRight))
//                {
//                    if ((detectedLinePoints.at(i).y >= m_midverticalCompleteUp) && (detectedLinePoints.at(i).y <= m_midverticalCompleteDown))
//                    {
//                        m_numberPointsLeft++;
//                    }
//                }
            }

            LOG_INFO("I saw a white spot in search area at %i", m_mostFarLeft);
            circle(outputImage, Point(m_mostFarLeft, m_mostFarLeftY), 5, Scalar(0, 128, 0), -1, 8);

            //This is the point my car will follow
            circle(outputImage, Point(m_mostFarLeft - m_displacement, m_mostFarLeftY), 5, Scalar(255, 192, 203), -1, 8);

            //Here I draw the two spots I set as limits for car to decide if it is going off road or out of lane
            circle(outputImage, Point(m_horizontalLeft_RightLane + m_OffRoadThreshold, m_mostFarLeftY), 5, Scalar(192, 192, 192), -1, 8);
            circle(outputImage, Point(m_horizontalRight_RightLane - m_OutOfLaneThreshold, m_mostFarLeftY), 5, Scalar(192, 192, 192), -1, 8);

            tInt32 AA = m_horizontalLeft_RightLane + m_OffRoadThreshold;
            tInt32 BB = m_horizontalRight_RightLane - m_OutOfLaneThreshold;

            LOG_INFO(cString::Format("The point for off road limit is %i", AA));
            LOG_INFO(cString::Format("The point for out of lane limit is %i", BB));

            if (m_mostFarLeft <= (m_horizontalLeft_RightLane + m_OffRoadThreshold))
            {
                LOG_INFO("You are going off road [RIGHT_LANE]!");
                WarningSignal.f32Value = tFloat32(2);
            }
            else if (m_mostFarLeft >= (m_horizontalRight_RightLane - m_OutOfLaneThreshold))
            {
                LOG_INFO("You are going out of lane [RIGHT_LANE]!");
                WarningSignal.f32Value = tFloat32(1);
            }

            //draw the circles for right lane
            circle(outputImage, Point(m_horizontalLeft_RightLane, m_verticalUp_RightLane), 5, Scalar(255, 0, 0), -1, 8);
            circle(outputImage, Point(m_horizontalRight_RightLane, m_verticalUp_RightLane), 5, Scalar(255, 0, 0), -1, 8);
            circle(outputImage, Point(m_horizontalLeft_RightLane, m_verticalDown_RightLane), 5, Scalar(255, 0, 0), -1, 8);
            circle(outputImage, Point(m_horizontalRight_RightLane, m_verticalDown_RightLane), 5, Scalar(255, 0, 0), -1, 8);

            //draw the circles for middle lane
            circle(outputImage, Point(m_midhorizontalCompleteLeft, m_midverticalCompleteUp), 5, Scalar(165, 42, 42), -1, 8);
            circle(outputImage, Point(m_midhorizontalCompleteRight, m_midverticalCompleteUp), 5, Scalar(165, 42, 42), -1, 8);
            circle(outputImage, Point(m_midhorizontalCompleteLeft, m_midverticalCompleteDown), 5, Scalar(165, 42, 42), -1, 8);
            circle(outputImage, Point(m_midhorizontalCompleteRight, m_midverticalCompleteDown), 5, Scalar(165, 42, 42), -1, 8);

            LOG_INFO(cString::Format("The warning signal is %f", WarningSignal.f32Value));

//            if (m_numberPointsLeft > m_numberPointsRight)
//            {
//                LOG_INFO("You are in the LEFT LANE! Turn right now!");
//                LOG_INFO(cString::Format("Number of points to your right are %i", m_numberPointsRight));
//                LOG_INFO(cString::Format("Number of points to your left are %i", m_numberPointsLeft));
//                m_prevLaneStatus = m_myLaneStatus;
//                m_myLaneStatus = 1;
//                WarningSignal.f32Value = tFloat32(1);
//            }
//            else
//            {
//                LOG_INFO("You are in RIGHT LANE. Everything is fine!");
//                LOG_INFO(cString::Format("Number of points to your right are %i", m_numberPointsRight));
//                LOG_INFO(cString::Format("Number of points to your left are %i", m_numberPointsLeft));
//                m_prevLaneStatus = m_myLaneStatus;
//                m_myLaneStatus = 0;
//            }

//            if ((m_numberPointsLeft == 0) && (m_numberPointsRight == 0))
//            {
//                LOG_INFO("I can't see any lane! I will make a decision based on the lane I was in previously");
//                if (m_prevLaneStatus == 0)
//                {
//                    LOG_INFO("I was in the RIGHT LANE previously. So, I will turn left and hopefully find some lane!");
//                    WarningSignal.f32Value = tFloat32(2);
//                }
//                else if (m_prevLaneStatus == 1)
//                {
//                    LOG_INFO("I was in the LEFT LANE previously. So, I will turn right and hopefully find some lane!");
//                    WarningSignal.f32Value = tFloat32(1);
//                }
//            }

            RETURN_IF_FAILED(TransmitWarning(WarningSignal));

            pReadBuffer->Unlock();

        }

        //Write processed Image to Output Pin
        if (!outputImage.empty())
        {
            //update output format if matrix size does not fit to
            if (outputImage.total() * outputImage.elemSize() != m_OutPinVideoFormat.m_szMaxByteSize)
            {
                setTypeFromMat(m_oWriterVideo, outputImage);
            }
            // write to pin
            writeMatToPin(m_oWriterVideo, outputImage, m_pClock->GetStreamTime());
        }

        //Write processed bird eye Image to Output Pin
        if (!outputImageBirdEye.empty())
        {
            //update output format if matrix size does not fit to
            if (outputImageBirdEye.total() * outputImageBirdEye.elemSize() != m_OutPinVideoFormatBirdEye.m_szMaxByteSize)
            {
                setTypeFromMat(m_oWriterVideoBirdEye, outputImageBirdEye);
            }
            // write to pin
            writeMatToPin(m_oWriterVideoBirdEye, outputImageBirdEye, m_pClock->GetStreamTime());
        }

        LOG_INFO("VISIONE INIZIO");
        Mat pippo = outputImageNormal;
        resize(outputImageNormal, pippo, Size(300, 300), 0, 0, INTER_CUBIC);
        //cvtColor(pippo, pippo, CV_BGR2RGB);
        imwrite( "vista.jpg", pippo );
        int ciccio = findObjects(pippo, 0);

        // show Min and Max line width which is searched.
        int myFont = FONT_HERSHEY_SIMPLEX;
        double scaleOfFont = 2;
        int myThickness = 3;
        int lengthOfTheLine = 100;

        if ((sawAdultFront == true) && (tempBBOX_X0_FRONT <= 0.5))
        {
            string adultText = "I saw an adult to my right";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawAdultFront == true) && (tempBBOX_X0_FRONT > 0.5))
        {
            string adultText = "I saw an adult to my left";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawChildFront == true) && (tempBBOX_X0_FRONT <= 0.5))
        {
            string adultText = "I saw a child to my right";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawChildFront == true) && (tempBBOX_X0_FRONT > 0.5))
        {
            string adultText = "I saw a car to my left";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawCarFront == true) && (tempBBOX_X0_FRONT <= 0.5))
        {
            string adultText = "I saw a car to my right";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawCarFront == true) && (tempBBOX_X0_FRONT > 0.5))
        {
            string adultText = "I saw an emergency vehicle to my left";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawEmergencyFront == true) && (tempBBOX_X0_FRONT <= 0.5))
        {
            string adultText = "I saw an emergency vehicle to my right";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

        if ((sawEmergencyFront == true) && (tempBBOX_X0_FRONT > 0.5))
        {
            string adultText = "I saw an emergency vehicle to my left";

            int baselineMax = 0;
            Size textSizeMax = getTextSize(adultText, myFont,
                                        scaleOfFont, myThickness, &baselineMax);
            baselineMax += myThickness;

            // Place Text
            Point textOrgMax(0, std::max(textSizeMax.height, static_cast<int>(m_maxLineWidth)));

            // ... and the baseline first
            rectangle(outputImageNormal, textOrgMax + Point(textSizeMax.width, myThickness),
                 textOrgMax + Point(textSizeMax.width, myThickness) + Point(lengthOfTheLine, -m_maxLineWidth),
                 Scalar(0, 255, 0),CV_FILLED);

            // then put the text itself
            putText(outputImageNormal, adultText, textOrgMax, myFont, scaleOfFont,
                    Scalar(0, 255, 0), myThickness, 8);
        }

//        m_TheBoundingBox = cv::Rect2f(static_cast<tFloat32>(tempBBOX_X0), static_cast<tFloat32>(tempBBOX_Y0), static_cast<tFloat32>(tempBBOX_WIDTH), static_cast<tFloat32>(tempBBOX_HEIGHT));
//        rectangle(outputImageNormal, m_TheBoundingBox, Scalar(255), 10, 8, 0);

        //Write processed normal Image to Output Pin
        if (!outputImageNormal.empty())
        {
            //update output format if matrix size does not fit to
            if (outputImageNormal.total() * outputImageNormal.elemSize() != m_OutPinVideoFormatNormal.m_szMaxByteSize)
            {
                setTypeFromMat(m_oWriterVideoNormal, outputImageNormal);
            }
            // write to pin
            writeMatToPin(m_oWriterVideoNormal, outputImageNormal, m_pClock->GetStreamTime());
        }

    }

    RETURN_NOERROR;
}

tResult cOpenCVTemplate::findLinePoints(const vector<tInt>& detectionLines, const cv::Mat& image, vector<Point>& detectedLinePoints)
{
    RETURN_IF_FAILED(checkRoi());
    //iterate through the calculated horizontal lines
    for (vector<tInt>::const_iterator nline = detectionLines.begin(); nline != detectionLines.end(); nline++)
    {
        uchar ucLastVal = 0;

        // create vector with line data
        const uchar* p = image.ptr<uchar>(*nline, m_ROIOffsetX);
        std::vector<uchar> lineData(p, p + m_ROIWidth);

        tBool detectedStartCornerLine = tFalse;
        tInt columnStartCornerLine = 0;

        for (std::vector<uchar>::iterator lineIterator = lineData.begin(); lineIterator != lineData.end(); lineIterator++)
        {
            uchar ucCurrentVal = *lineIterator;
            tInt currentIndex = tInt(std::distance(lineData.begin(), lineIterator));
            //look for transition from dark to bright -> start of line corner
            if ((ucCurrentVal - ucLastVal) > m_minLineContrast)
            {
                detectedStartCornerLine = tTrue;
                columnStartCornerLine = currentIndex;
            }//look for transition from bright to dark -> end of line
            else if ((ucLastVal - ucCurrentVal) > m_minLineContrast && detectedStartCornerLine)
            {
                //we already have the start corner of line, so check the width of detected line
                if ((abs(columnStartCornerLine - currentIndex) > m_minLineWidth)
                    && (abs(columnStartCornerLine - currentIndex) < m_maxLineWidth))
                {
                    detectedLinePoints.push_back(Point(tInt(currentIndex - abs(columnStartCornerLine - currentIndex) / 2 +
                                                 m_ROIOffsetX), *nline));

                    detectedStartCornerLine = tFalse;
                    columnStartCornerLine = 0;
                }
            }
            //we reached maximum line width limit, stop looking for end of line
            if (detectedStartCornerLine &&
                abs(columnStartCornerLine - currentIndex) > m_maxLineWidth)
            {
                detectedStartCornerLine = tFalse;
                columnStartCornerLine = 0;
            }
            ucLastVal = ucCurrentVal;
        }
    }

    RETURN_NOERROR;
}

tResult cOpenCVTemplate::getDetectionLines(vector<tInt>& detectionLines)
{
    tInt distanceBetweenDetectionLines = m_ROIHeight / (m_detectionLines + 1);

    for (int i = 1; i <= m_detectionLines; i++)
    {
        detectionLines.push_back(m_ROIOffsetY + i * distanceBetweenDetectionLines);
    }
    RETURN_NOERROR;
}

tResult cOpenCVTemplate::checkRoi(void)
{
    // if width or heigt are not set ignore the roi
    if (static_cast<tFloat32>(m_ROIWidth) == 0 || static_cast<tFloat32>(m_ROIHeight) == 0)
    {
        LOG_ERROR("ROI width or height is not set!");
        RETURN_ERROR_DESC(ERR_INVALID_ARG, "ROI width or height is not set!");
    }

    //check if we are within the boundaries of the image
    if ((static_cast<tFloat32>(m_ROIOffsetX) + static_cast<tFloat32>(m_ROIWidth)) > m_InPinVideoFormat.m_ui32Width)
    {
        LOG_ERROR("ROI is outside of image");
        RETURN_ERROR_DESC(ERR_INVALID_ARG, "ROI is outside of image");
    }

    if ((static_cast<tFloat32>(m_ROIOffsetY) + static_cast<tFloat32>(m_ROIHeight)) > m_InPinVideoFormat.m_ui32Height)
    {
        LOG_ERROR("ROI is outside of image");
        RETURN_ERROR_DESC(ERR_INVALID_ARG, "ROI is outside of image");
    }

    //create the rectangle
    m_LaneRoi = cv::Rect2f(static_cast<tFloat32>(m_ROIOffsetX), static_cast<tFloat32>(m_ROIOffsetY), static_cast<tFloat32>(m_ROIWidth), static_cast<tFloat32>(m_ROIHeight));


    RETURN_NOERROR;
}

//Here you write the outputs to the output pins
tResult cOpenCVTemplate::TransmitWarning(tSignalValue WarningSignal)
{
    object_ptr<ISample> pWriteSample;

    //Here you allocate a media sample which has a return type of tResult
    //Just check if tResult is valid or not
    RETURN_IF_FAILED(alloc_sample(pWriteSample))
    {
        auto oCodec = m_WarningFactory.MakeCodecFor(pWriteSample);

        RETURN_IF_FAILED(oCodec.SetElementValue(m_ddlWarning.timeStamp, WarningSignal.ui32ArduinoTimestamp));
        RETURN_IF_FAILED(oCodec.SetElementValue(m_ddlWarning.value, WarningSignal.f32Value));
    }

    m_oWarning << pWriteSample << flush << trigger;
    RETURN_NOERROR;
}


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status cOpenCVTemplate::ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

Status cOpenCVTemplate::readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();

    // Search entry patterns of type 'item { ... }' and parse each of them
    smatch matcherEntry;
    smatch matcherId;
    smatch matcherName;
    const regex reEntry("item \\{([\\S\\s]*?)\\}");
    const regex reId("[0-9]+");
    const regex reName("\'.+\'");
    string entry;

    auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = sregex_iterator();

    int id;
    string name;
    for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
        matcherEntry = *i;
        entry = matcherEntry.str();
        regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = stoi(matcherId[0].str());
        else
            continue;
        regex_search(entry, matcherName, reName);
        if (!matcherName.empty())
            name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
        else
            continue;
        labelsMap.insert(pair<int, string>(id, name));
        LOG_INFO("ITEM AGGIUNTO");
    }
    return Status::OK();
}

Status cOpenCVTemplate::ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
//Status cOpenCVTemplate::ReadTensorFromImageFile(const string& file_name, const int input_height,
//                               const int input_width, const float input_mean,
//                               const float input_std,
//                               std::vector<Tensor>* out_tensors) {


Status cOpenCVTemplate::ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  // auto float_caster =
  //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  // auto resized = ResizeBilinear(
  //     root, dims_expander,
  //     Const(root.WithOpName("size"), {input_height, input_width}));


  // Subtract the mean and divide by the scale.
  // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
  //     {input_std});


  //cast to int
  //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status cOpenCVTemplate::LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

Status cOpenCVTemplate::readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return Status::OK();
}

//whichCamera = 0 means front, 1 means rear
int cOpenCVTemplate::findObjects(Mat img_matrix, int whichCamera)
{
//Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
//if (!readLabelsMapStatus.ok()) {
//    LOG_ERROR("readLabelsMapFile(): ERROR");
//    return -1;
//} else
//    LOG_INFO("readLabelsMapFile() labels map loaded");

Tensor tensor;
std::vector<Tensor> outputs;

tensorflow::TensorShape shape = tensorflow::TensorShape();
shape.AddDim(1);
shape.AddDim(img_matrix.rows);
shape.AddDim(img_matrix.cols);
shape.AddDim(3);

//cvtColor(img_matrix, img_matrix, COLOR_BGR2RGB);

// Convert mat to tensor
tensor = Tensor(tensorflow::DT_FLOAT, shape);
Status readTensorStatus = readTensorFromMat(img_matrix, tensor);
if (!readTensorStatus.ok()) {
    LOG_ERROR("Mat->Tensor conversion failed");
    return -1;
}
else
{
    LOG_INFO("MAT -> TENSOR CONVERSION IS SUCCESS!");
}

// Run the graph on tensor
outputs.clear();
Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
if (!runStatus.ok()) {
    LOG_ERROR("Running model failed");
    return -1;
}
else {
    LOG_INFO("RUNNING MODEL SUCCESS");
}

// Extract results from the outputs vector
tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();

// TRYING TO PRINT TENSOR VALUES
//auto output_c = outputs[1].scalar<float>();
LOG_INFO("LOOOKK HEREEEE");
// Print the results
std::cout << outputs[1].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
//std::cout << output_c() << "\n"; // 30
//ofstream myfile;
//myfile.open ("DETECTIONS.txt");

//if class is 1 then emergency, 2 then car, 3 then adult, 4 then child

for(size_t i = 0; i < numDetections(0) && i < 20;++i)
{
  if(scores(i) > 0.3)
  {
    LOG(ERROR) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);

//    myfile << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
//    tempBBOX_X0 = ((tFloat32)boxes(0,i,0)) * (tFloat32)img_matrix.cols;
//    tempBBOX_Y0 = ((tFloat32)boxes(0,i,1)) * (tFloat32)img_matrix.rows;
//    tempBBOX_WIDTH = ((tFloat32)boxes(0,i,2)) * (tFloat32)img_matrix.cols;
//    tempBBOX_HEIGHT = ((tFloat32)boxes(0,i,3)) * (tFloat32)img_matrix.rows;

        tempBBOX_X0_FRONT = ((tFloat32)boxes(0,i,0));
        tempBBOX_Y0_FRONT = ((tFloat32)classes(i));
        LOG_INFO(cString::Format("THE X0 IS %f IN FRONT CAMERA", tempBBOX_X0_FRONT));

        if (tempBBOX_Y0_FRONT == 1)
        {
            sawEmergencyFront = true;
        }
        else if (tempBBOX_Y0_FRONT == 2)
        {
            sawCarFront = true;
        }
        else if (tempBBOX_Y0_FRONT == 3)
        {
            sawAdultFront = true;
        }
        else if (tempBBOX_Y0_FRONT == 4)
        {
            sawChildFront = true;
        }

//    tempBBOX_Y0 = ((tFloat32)boxes(0,i,0)) * 960;
//    tempBBOX_X0 = ((tFloat32)boxes(0,i,1)) * 1280;
//    tempBBOX_WIDTH = ((tFloat32)boxes(0,i,2)) * 1280;
//    tempBBOX_HEIGHT = ((tFloat32)boxes(0,i,3)) * 960;
//    tempBBOX_WIDTH = tempBBOX_WIDTH - tempBBOX_X0;
//    tempBBOX_HEIGHT = tempBBOX_HEIGHT - tempBBOX_Y0;
    //LOG_INFO(cString::Format("THE X0 IS %f", tempBBOX_X0));
    //LOG_INFO(cString::Format("THE Y0 IS %f", tempBBOX_Y0));
    //LOG_INFO(cString::Format("THE WIDTH IS %f", tempBBOX_WIDTH));
    //LOG_INFO(cString::Format("THE HEIGHT IS %f", tempBBOX_HEIGHT));
  }

}
//myfile.close();
//chrono::seconds dura(5);
//this_thread::sleep_for(dura);

//vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
//for (size_t i = 0; i < goodIdxs.size(); i++)
//    LOG(INFO) << "class:" << labelsMap[classes(goodIdxs.at(i))];

// Draw bboxes and captions
//cvtColor(img_matrix, img_matrix, COLOR_BGR2RGB);
//drawBoundingBoxesOnImage(img_matrix, scores, classes, boxes, labelsMap, goodIdxs);

return 0;
}

vector<size_t> cOpenCVTemplate::filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);

    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
}
}

    void cOpenCVTemplate::drawBoundingBoxesOnImage(Mat &image,
                                  tensorflow::TTypes<float>::Flat &scores,
                                  tensorflow::TTypes<float>::Flat &classes,
                                  tensorflow::TTypes<float,3>::Tensor &boxes,
                                  map<int, string> &labelsMap,
                                  vector<size_t> &idxs) {
        int temp = (int)idxs.size();
        for (int j = 0; j < temp; j++)
            drawBoundingBoxOnImage(image,
                                   boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                                   boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                                   scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
    }

    double cOpenCVTemplate::IOU(Rect2f box1, Rect2f box2) {

        float xA = max(box1.tl().x, box2.tl().x);
        float yA = max(box1.tl().y, box2.tl().y);
        float xB = min(box1.br().x, box2.br().x);
        float yB = min(box1.br().y, box2.br().y);

        float intersectArea = abs((xB - xA) * (yB - yA));
        float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

        return 1. * intersectArea / unionArea;
    }

    void cOpenCVTemplate::drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled) {
        cv::Point tl, br;
        if (scaled) {
            tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
            br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
        } else {
            tl = cv::Point((int) xMin, (int) yMin);
            br = cv::Point((int) xMax, (int) yMax);
        }
        cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

        // Ceiling the score down to 3 decimals (weird!)
        float scoreRounded = floorf(score * 1000) / 1000;
        string scoreString = to_string(scoreRounded).substr(0, 5);
        string caption = label + " (" + scoreString + ")";

        // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
        int fontCoeff = 12;
        cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
        cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
        cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
        cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
    }

