# bsm-ds-augment

## Set up instructions

The dataset used can be found at https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/YHKWMR
The dataset files will need to be stored in a data folder accessible by python
It is highly recommended to set up virtual environments for each stage of the process: dataset-split, lora-fine-tune and comp-vis-models as there could be (probably are) conflicting library requirements.
WandB was used for reporting during LoRA training. If this isn't desired, remove the "report_to" flag from the launch-fine-tuning.sh script

### Dataset Split

Once the dataset has been downloaded from the link above, the python files should be executed in the following order:

- train-test-split.py
- resize-images.py
- create_caption_file.py

The data paths will need to be filled in for each python file before running.

Python 3.11 or later is recommended.

### LoRA Fine Tuning

The Stable Diffusion 2 model can be found at https://huggingface.co/stabilityai/stable-diffusion-2

A Hugging Face read token will need to be generated for use in bringing in the model during LoRA fine tuning. 

- Install python 3.11
- Create virtual environment
- Import LoRA from source

        git clone https://github.com/huggingface/diffusers

        cd diffusers

        pip install .

        cd examples/text_to_image

        pip install -r requirements.txt

Accelerate will need to be configured by running the `accelerate config` command.

More information on LoRA through Hugging Face's Diffusers library can be found at https://huggingface.co/docs/diffusers/training/lora

### Model Training and Testing

Python 3.11 in a virtual environment is recommended.

Run the requirements file in the comp-vis-models directory: `pip install -r requirements.txt`

Add the appropriate data paths to each python file (or point the files to a parameters file).

Run the models-run.sh file with `bash models-run.sh` or `./models-run.sh` (if avaiable as an executable script).
