## Google Vision Kit 


**Vision demo**

**version**: April 2018


|Name|Description|Considerations|Improvements|
|---|---|---|---|
|Image Classification Camera|object classification using camera|Low confidence. See results: [https://gist.github.com/giacomobartoli/a29b066dfefb4af8ac474329c7d2b52b]()|Adding a bounding box framing objects (ex: YOLO)|
|Image Classification | Given an input image this demo classifies objects in it. There are two models available for image classification task: MobileNet based (image classification.MOBILENET), which has **59.9%** top-1 accuracy on ImageNet; SqueezeNet based (image_classification.SQUEEZENET), which has **45.3%** top-1 accuracy on ImageNet.| Still to week. I tried using /img/room.jpg and it only found a bookcase and a window shade. However, the picture contains also plants, books, a sofa, a carpet, woods. | Improve accuracy |
|Dish Classifier| Given an input image this demo can classify food. Ex: piadina, napolean pizza with prosciutto, frozen yogurt| It's incredibly accurate. It's even able to distinguish composed food. | Make it in real time |
|Dish Detection| Same as dish classifier but it provides detection (bbox)| It's incredibly accurate. | Make it in real time |
|Joy Detector| Led button lights up depending on how much a detected face is happyor not. Ex: happy face => yellow led.| It works very well. | What about multiple faces? It would be nice even to label detected emotions. |
|Face Camera Trigger| It takes a picture as soon as a face is detected | Pretty clever. However, it does not recognize a face if the person is a little bit turned from behind. | --- |
|Leds example| | | |
|Face Detection| Given an input image, this script face detection confidence and joy score. Ex: ./face_detection.py --input faces.jpg face_score=0.987305, joy_score=0.003976, bbox=(722.0, 152.0, 672.0, 672.0) | Multiple faces are allowed. For each detected face the algorithm returns: 1.face_score, 2.joy_score, 3.bbox| |
|Face Detection Camera| Framing  faces through camera| Multple faces are allowed. However, the frame is lagging a little bit. | Improve the FPS rate or even test tinyYOLO |
|MobileNet base classifier| | | |
|Object detection| The object detection demo takes an image and checks whether it’s a cat, dog, or person. | It does not work on very simple tasks. Given an image with a god and a cat I got only one dog with 70% accuracy. See /img folder for the images with whom I tried. | Extended this demo to other objects. This can be good for fining tuning and specific scenarios, eg: person detection on unmanned vechicles. |
|Nature explorer| Based on a subset of Visipedia. It can recognize insects, plants and bird. | This seems to be the only model based on MobileNet v1, input=192| |