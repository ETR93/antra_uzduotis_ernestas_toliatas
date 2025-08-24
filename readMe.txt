To run script python (>=3.10) is needed to installed

Run commands to install needed packages:
pip3 install torch torchvision
pip3 install numpy

NOTE:
If IDE is being used you can add these packages via package manager, look for package
* torch
* numpy

Usage:
TO GENERATE NEW MODEL
There will already trained model file control__nn.pth. To generate new model delete the old file and run command
python ModelTrain.py . There will be asked amount of training sample (was tested on 200) and target value (was tested on 5)

RUN VALUES PREDICTION
When the model is generated (file control__nn.pth) to run values predictions run command python Main.py

NOTE
Script only will work on Windows. On linux or mac was not tested.
