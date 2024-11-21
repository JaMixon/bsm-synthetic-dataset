#!/usr/bin/bash

cd # change directory to the directory housing the python run files

python3 'densenet-models-synth.py'
python3 'densenet-models-combi.py'

python3 'inceptionv3-models-synth.py'
python3 'inceptionv3-models-combi.py'

python3 'resnet50-models-synth.py'
python3 'resnet50-models-combi.py'

python3 'vgg16-models-synth.py'
python3 'vgg16-models-combi.py'
