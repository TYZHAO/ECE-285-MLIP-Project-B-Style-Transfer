# ECE285 Final Project B - Style Transfer
## Description
This is project Style Transfer developed by team Team Name. This branch is the implementation of Cycle-GAN.

## Requirement
Please make sure the data set is at /datasets/ee285f-public/wikiart/wikiart/ directory. To 
run the demo, you will only need the timg.jpg (as your original image) in the root folder of the project. To test on different styles, just modify the address of the styleImg. Make sure the target style images are in the same directory of notebook file and run the demo files.

## Code organization
<pre>
cyclegan---------------------------------Contains all the codes.
  dataset.py-----------------------------Contains the dataset function
  models.py------------------------------Contains all models
  run.py---------------------------------Manage the training
  test.py--------------------------------Generte result based on saved checkpoints 
  train.ipynb----------------------------ipython version of run.py
  train.py-------------------------------Contains an experiment class
  utils.py-------------------------------Other functions
datasets---------------------------------folder contains all the datasets
</pre>

## Enjoy!
                              
