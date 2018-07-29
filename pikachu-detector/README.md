![welcome](pikachu.png)

## Pikachu Detector 
Have you ever wondered how to exploit Deep Learning and Google Vision Kit for detecting Pikachu? Well, in that case this project might be interesting for you.

### Files:
* pikachu_object_detection.py: a small library that process .protobinary file
* custom_pikachu_detector.py: a ready to go python executable. Its aim is to reveal the camera, capture the input video and apply inference.
* pikachu_detector.protobinary: a tf frozen graph. It has been trained over a pre-trained coco model, with 10k iterations.


### Deploy:

* Connect through ssh to your Vision Kit
* Move to /home/AIY-projects/src/example/vision
* Clone this repo
* Type `./custom_pikachu_detector` from a shell


### Results:

![Pikachu Detector](test.jpg)

### References:
* [Custom Vision Training AIY](https://cogint.ai/custom-vision-training-on-the-aiy-vision-kit/) by Chard Hart