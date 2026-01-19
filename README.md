# Python-projects


This is my Masters project which I completed for my Theoretical Physics course at Durham University.

PROJECT OVERVIEW
This project applies deep-learning–based object detection to automated platelet recognition and morphological analysis in microscopy images. Using a PyTorch-based Mask R-CNN pipeline, the model detects individual platelets and extracts quantitative shape metrics relevant to biomedical analysis.

COPYRIGHT:
Due to copyright restrictions, I cannot upload all the pictures that I have used for the model training apart from two.

TECHNICAL REQUIREMENTS:
  -pytorch
  -torchvision
  -torch
  -torchsnippets
  -pycocotools
  -NVIDIA CUDA
  (Optional for faster training):
  -NVIDIA GPU (e.g. GTX 2080).
  
  
HOW TO TRAIN THE MODEL:
To train the model, you will need to have images and their annotations ready. The file, responsible for training, is called model_training_script.py. In the same working directory you will need to have engine.py, transform.py, coco_eval.py, coco_utils.py, and utils.py. They are provided in the "Masters project" and are a part of the official PyTorch object detection tutorial, and werenot authored by me. To find out more, visit https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html or check out model_training_script.py. The annotations were made using Label Studio. Make sure that you have an NVIDIA GPU and that you have downloaded NVIDIA's CUDA, otherwise, the training is going to take much longer. In addition, beware of using too many images in a single batch, as it will eat up VRAM. The process becomes unstable if more than 10 are used.

HOW TO USE THE TRAINED MODEL (model.pth):
To use the trained model, which is outside the "Masters project" folder, you will need to download it and put it in the working directory. There are 2 images available for the demonstration in "images_for_illustration" folder. The file that generates predictions is "object_recognition.py" which will require "statistics_visualisation.py" and "score_analysis.py" files to properly function and create charts displaying distribution of various platelet metrics. These morphology metrics are area, perimeter, circularity, eccentricity, aspect ratio, solidity, and anisotropy. A platelet thickness estimation algorithm was prototyped but excluded from the final pipeline due to sensitivity to noise. I recommend using a Spyder IDE as it allows accessing variables directly after they have been generated.

SKILLS DEMONSTRATED
  -PyTorch & Torchvision model training
  -Object annotation (COCO format)
  -Dataset annotation (Label Studio)
  -Quantitative image analysis
  -Performance/debugging considerations (VRAM, hyperparameter variation)
