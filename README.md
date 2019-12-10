# ECE285 Final Project B - Style Transfer
## Description
This is project Style Transfer developed by team Team Name. This branch is the implementation of Cycle-GAN.

## Requirement
To run the demo, open cyclegan/demo.ipynb with jupyter notebook on a GPU node. It will automatically generate the transfer results of all 4 differenet styles mentioned in the report. Note that every checkpoint in cyclegan/output is needed.
To train the model, run cyclegan/run.py in terminal or run cyclegan/train.ipynb in jupyter notebook on a GPU node. Note that all training data are refined, so please download data in datasets/. If we upload all data for all 4 styles the repo would be too large, so we only upload the Monet one. A log file will be generated during training.

## Code organization
<pre>
cyclegan---------------------------------Contains all the codes.
  output---------------------------------Folder contains all pretrained models and generated output images.
  demo.ipynb-----------------------------Demo of Cycle-GAN 
  dataset.py-----------------------------Contains the dataset function
  models.py------------------------------Contains all models
  run.py---------------------------------Manage the training
  test.py--------------------------------Generte result based on saved checkpoints 
  train.ipynb----------------------------ipython version of run.py
  train.py-------------------------------Contains an experiment class
  utils.py-------------------------------Other functions
  
datasets---------------------------------Folder contains all the datasets
  field_select---------------------------All selected field photographs
  image_monet_field----------------------All selected monet paintings
  field_data-----------------------------Training photographs list
  monet_field_data-----------------------Training paintings list
  test-----------------------------------Folder contains all testing data
    cb-----------------------------------Cubism paintings
    monet--------------------------------Monet paintings
    vg-----------------------------------Van Gogh paintings
    point--------------------------------Pointillism paintings
    photo--------------------------------Landscape photographs
</pre>
                              
