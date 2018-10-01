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


#pragma once

//*************************************************************************************************
#define CID_COPENCVTEMPLATE_DATA_TRIGGERED_FILTER "opencv_template.filter.user.aadc.cid"

using namespace adtf_util;
using namespace ddl;
using namespace adtf::ucom;
using namespace adtf::base;
using namespace adtf::streaming;
using namespace adtf::mediadescription;
using namespace adtf::filter;
using namespace std;
using namespace cv;


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


/*! the main class of the open cv template. */
class cOpenCVTemplate : public cTriggerFunction
{
private:
    /*! Media description for variable sent to speed controller telling it to move to next maneuver. */
    struct tUpdateManeuver
    {
        tSize timeStamp;
        tSize value;
    } m_ddlUpdateManeuver;

    /*! The signal data sample factory. It is used for reading out the samples */
    adtf::mediadescription::cSampleCodecFactory m_UpdateManeuverFactory;

    /*! Tell the car that it is going off road or out of lane. */
    struct tWarning
    {
        tSize timeStamp;
        tSize value;
    } m_ddlWarning;

    /*! The signal data sample factory. It is used for reading out the samples */
    adtf::mediadescription::cSampleCodecFactory m_WarningFactory;

    //Properties
    /*! Offset of the ROI in the Stream*/
    adtf::base::property_variable<int> m_ROIOffsetX = 0;
    /*! Offset of the ROI in the Stream*/
    adtf::base::property_variable<int> m_ROIOffsetY = 600;
    /*! Width of the ROI*/
    adtf::base::property_variable<int> m_ROIWidth = 1280;
    /*! Height of the ROI*/
    adtf::base::property_variable<int> m_ROIHeight = 200;
    /*! number of detection lines searched in ROI */
    adtf::base::property_variable<int> m_detectionLines = 20;
    /*! Minimum Line Width in Pixel */
    adtf::base::property_variable<int> m_minLineWidth = 10;
    /*! Maximum Line Width in Pixel */
    adtf::base::property_variable<int> m_maxLineWidth = 30;
    /*! Mimimum line contrast in gray Values */
    adtf::base::property_variable<int> m_minLineContrast = 50;
    /*! Threshold for image binarization */
    adtf::base::property_variable<int> m_thresholdImageBinarization = 180;
    /*! Thresholds for search area (basically, this is the area I will look for the white spots of a lane) */
    adtf::base::property_variable<int> m_horizontalLeft_RightLane = 1000;
    adtf::base::property_variable<int> m_horizontalRight_RightLane = 1400;
    adtf::base::property_variable<int> m_verticalUp_RightLane = 400;
    adtf::base::property_variable<int> m_verticalDown_RightLane = 600;
    /*! Search for middle line here. */
    adtf::base::property_variable<int> m_midhorizontalCompleteLeft = 500;
    adtf::base::property_variable<int> m_midhorizontalCompleteRight = 600;
    adtf::base::property_variable<int> m_midverticalCompleteUp = 400;
    adtf::base::property_variable<int> m_midverticalCompleteDown = 500;
    /*! Left and right thresholds for going off road and out of lane */
    adtf::base::property_variable<int> m_OffRoadThreshold = 100;
    adtf::base::property_variable<int> m_OutOfLaneThreshold = 100;
    /*! Points to get birds eye perspective */
    adtf::base::property_variable<int> m_topLeftX = 0;
    adtf::base::property_variable<int> m_topRightX = 1280;
    adtf::base::property_variable<int> m_bottomLeftX = 0;
    adtf::base::property_variable<int> m_bottomRightX = 1280;
    adtf::base::property_variable<int> m_topLeftY = 0;
    adtf::base::property_variable<int> m_topRightY = 0;
    adtf::base::property_variable<int> m_bottomLeftY = 960;
    adtf::base::property_variable<int> m_bottomRightY = 960;
    adtf::base::property_variable<int> m_destResX = 800;
    adtf::base::property_variable<int> m_destResY = 600;
    /*! You want to get the point your car will follow by simply adding this value to the detected line point */
    adtf::base::property_variable<int> m_displacement = 100;
    /*! Keep track of which lane you are in. 0 for right and 1 for left */
    adtf::base::property_variable<int> m_currentLane = 0;

    //Pins
    /*! Reader for the video. */
    cPinReader m_oReaderVideo;
    /*! Writer for the video. */
    cPinWriter m_oWriterVideo;
    /*! Writer for the bird eye video. */
    cPinWriter m_oWriterVideoBirdEye;
    /*! Writer for the normal video. */
    cPinWriter m_oWriterVideoNormal;
    /*! Writer that tells speed controller to switch to next maneuver. */
    cPinWriter m_oUpdateManeuver;
    /*! Writer that tells speed controller that it might be going off road or out of lane. */
    cPinWriter m_oWarning;

    //Variable that will consider only the most far-left point when in right lane
    tInt32 m_mostFarLeft;
    tInt32 m_mostFarLeftY;
    //Number of points to left and right of car, comparison will tell car whether it is in left or right lane
    tInt32 m_numberPointsRight;
    tInt32 m_numberPointsLeft;

    //Your changing lane status is stored here at all times
    tInt32 m_myLaneStatus;
    //Store your previous lane status
    tInt32 m_prevLaneStatus;

    //Values for bbox
    tFloat32 tempBBOX_X0_FRONT;
    tFloat32 tempBBOX_Y0_FRONT;
    tFloat32 tempBBOX_WIDTH_FRONT;
    tFloat32 tempBBOX_HEIGHT_FRONT;
    //bools for each object
    tBool sawAdultFront;
    tBool sawChildFront;
    tBool sawCarFront;
    tBool sawEmergencyFront;

     //Stream Formats
     /*! The input format */
     adtf::streaming::tStreamImageFormat m_InPinVideoFormat;
     /*! The output format */
     adtf::streaming::tStreamImageFormat m_OutPinVideoFormat;
     /*! The output format for bird's eye view */
     adtf::streaming::tStreamImageFormat m_OutPinVideoFormatBirdEye;
     /*! The output format for normal view */
     adtf::streaming::tStreamImageFormat m_OutPinVideoFormatNormal;

     /*! The clock */
     object_ptr<adtf::services::IReferenceClock> m_pClock;

     /*! lane detection roi bounding rectangle */
     cv::Rect m_LaneRoi = cv::Rect();
     cv::Rect m_LaneRoiRight = cv::Rect();
     cv::Rect m_LaneRoiLeft = cv::Rect();
     cv::Rect m_TheBoundingBox = cv::Rect();

     tensorflow::Status loadGraphStatus;

     std::unique_ptr<tensorflow::Session> session;

     // Set dirs variables
     string ROOTDIR = "../";
     string LABELS = "/home/aadc/Downloads/tf-audi/speed/vision/AADC_labels_map.pbtxt";
     string GRAPH = "/home/aadc/Downloads/tf-audi/speed/vision/vision_inference_graph_18.pb";

     string inputLayer;
     vector<string> outputLayer;

     std::map<int, std::string> labelsMap;


public:

    /*! Default constructor. */
    cOpenCVTemplate();


    /*! Destructor. */
    virtual ~cOpenCVTemplate() = default;

    /**
    * Overwrites the Configure
    * This is to Read Properties prepare your Trigger Function
    */
    tResult Configure() override;
    /**
    * Overwrites the Process
    * You need to implement the Reading and Writing of Samples within this function
    * MIND: Do Reading until the Readers queues are empty or use the IPinReader::GetLastSample()
    * This Function will be called if the Run() of the TriggerFunction was called.
    */
    tResult Process(tTimeStamp tmTimeOfTrigger) override;

    tResult TransmitWarning(tSignalValue WarningSignal);

    /* tf methods */

    int findObjects(Mat img_matrix, int whichCamera);

    tensorflow::Status readTensorFromMat(const Mat &mat, tensorflow::Tensor &outTensor);

    tensorflow::Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap);


    tensorflow::Status LoadGraph(const string& graph_file_name,
                     std::unique_ptr<tensorflow::Session>* session);

    tensorflow::Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                                   const int input_width, const float input_mean,
                                   const float input_std,
                                   std::vector<tensorflow::Tensor>* out_tensors);

    static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                                 tensorflow::Tensor* output);

    tensorflow::Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                          size_t* found_label_count);

    vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                               tensorflow::TTypes<float, 3>::Tensor &boxes,
                               double thresholdIOU, double thresholdScore);

    void drawBoundingBoxesOnImage(Mat &image,
                                  tensorflow::TTypes<float>::Flat &scores,
                                  tensorflow::TTypes<float>::Flat &classes,
                                  tensorflow::TTypes<float,3>::Tensor &boxes,
                                  map<int, string> &labelsMap,
                                  vector<size_t> &idxs);
    double IOU(Rect2f box1, Rect2f box2);

    void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled=true);

private:

    /*!
     * Searches for the first line points.
     *
     * \param           detectionLines      The detection lines.
     * \param           image               The image.
     * \param [in,out]  detectedLinePoints  The detected line points.
     *
     * \return  Standard Result Code.
     */
    tResult findLinePoints(const vector<tInt>& detectionLines, const cv::Mat& image, vector <Point>& detectedLinePoints);

    /*!
     * Gets detection lines.
     *
     * \param [in,out]  detectionLines  The detection lines.
     *
     * \return  Standard Result Code.
     */
    tResult getDetectionLines(vector<tInt>& detectionLines);

    /*!
    * Change type.
    *
    * \param [in,out]  inputPin    The input pin.
    * \param           oType       The type.
    *
    * \return  Standard Result Code.
    */
    tResult ChangeType(adtf::streaming::cDynamicSampleReader& inputPin,
        const adtf::streaming::ant::IStreamType& oType);

    /*!
    * Checks if the ROI is within the Image boundaries
    *
    *
    *
    * \return  Standard Result Code.
    */
    tResult checkRoi(void);
};


//*************************************************************************************************

