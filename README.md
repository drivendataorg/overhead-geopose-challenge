[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/geopose-homepage.png)
<figcaption>Images shown are from the public <a href="https://ieee-dataport.org/open-access/urban-semantic-3d-dataset">Urban Semantic 3D Dataset</a>, provided courtesy of DigitalGlobe</figcaption>

# <Overhead Geopose Challenge>

## Goal of the Competition

Overhead satellite imagery provides critical time-sensitive information for use areas like disaster response, navigation, and security. Most current methods for using aerial imagery assume images are taken from directly overhead, or “near-nadir”. However, the first images available are often taken from an angle, or are “oblique”. Effects from these camera orientations complicate useful tasks like change detection, vision-aided navigation, and map alignment.

In this challenge, participants made satellite imagery taken from a significant angle more useful for time-sensitive applications like disaster and emergency response

## What's in this Repository

This repository contains code from winning competitors in the [Overhead Geopose Challenge](https://www.drivendata.org/competitions/78/overhead-geopose-challenge/).

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Prediction Contest

All of the models below are based on the solution provided in the benchmark blog post: **["Overhead Geopose Challenge - Benchmark"](https://www.drivendata.co/blog/overhead-geopose-benchmark/)**

| Place | Team or User | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----- | ------------ | ------------ | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | selim_sef    | 0.902184     | 0.902459      | An EfficientNet V2 L encoder is used instead of the Resnet34 encoder because it has a huge capacity and is less prone to overfitting. The decoder is a UNet with more filters and additional convolution blocks for better handling of fine grained details. MSE loss would produce imbalance for different cities, depending on building heights. The model is trained with an R2 loss for AGL/MAG outputs, which reflects the final competition metric and is more robust to noisy training data. |
| 2     | bloodaxe     | 0.889567     | 0.890993      | I’ve trained a bunch of UNet-like models and averaged their predictions. Sounds simple, yet I used quite heavy encoders (B6 & B7) and custom-made decoders to produce very accurate height map predictions at original resolution. Another crucial part of the solution was extensive custom data augmentation for height, orientation, scale, gsd, and image RGB values.                                                                                                                           |
| 3     | o__@         | 0.882882     | 0.882801      | I ensembled the VFlow-UNet model using a large input resolution and a large backbone without downsampling. Better results were obtained when the model was trained on all images from the training set. The test set contains images of the same location as the images in the training set. This overlap was identified by image matching to improve the prediction results.                                                                                                                       |
| 4     | kbrodt       | 0.872775     | 0.873057      | The model uses a UNet architecture with various encoders (efficientnet-b{6,7} and senet154), and has only one above ground level (AGL) head and two heads in the bottleneck for scale and angle. The features are a random 512x512 crop of an aerial image, the city's one hot encoding, and ground sample distance (GSD). The model is trained with mean squared error (MSE) loss function for all targets (AGL, scale, angle) using AdamW optimizer with 1e-4 learning rate.                      |


Additional solution details can be found in the `reports` folder inside the directory for each submission.

### Model Write-up Bonus

| Team or User | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------ | ------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bloodaxe     | 0.889567     | 0.890993      | See the "Prediction Contest" section above                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| chuchu       | 0.856847     | 0.855636      | We conducted an empirical upper bound analysis, which suggested that the main errors are from height prediction and the rest are from angle prediction. To overcome the bottlenecks we proposed HR-VFLOW, which takes HRNet  as backbone and adopts simple multi-scale fusion as multi-task decoders to predict height, magnitude, angle, and scale simultaneously. To handle the height variance, we first pretrained the model on all four cities and then transfered the pretrained model to each specific city for better city-wise performance. |
| vecxoz       | 0.852948     | 0.851828      | {model explanation}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |





