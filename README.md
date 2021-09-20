# SimFish

A simulated zebrafish larvae for a 2D environment created using deep Q learning. Developed from project at https://bitbucket.org/azylbertal/simfish/src/master/

To run:
  1. Install dependencies using "pip install requirements.txt"
  2. Compile cython module using "python setup.py build_ext --inplace" from the Tools directory.  
  3. Run the "run.py" file, from which the chosen environment name and trial number can be adjusted.
  4. This will create the necessary files (given a bespoke "create_configuration_{environment}.py" file)
  5. To test the environment, run the "run_environment_test.py" file. Note that for Linux this will require installation of tkinter, which can be done using "sudo apt-get install python3-tk"

To view graphs, install tensorboard and run 'tensorboard --logdir="./Output/{trial_directory}/logs"'

Uses the following packages:
* Tensorflow==2.3.1/1.15.0
* numpy==1.19.2
* h5py==2.10.0
* matplotlib==3.3.2
* pymunk==5.7 - MUST be version 5.7
* skimage==0.17.2 - install as "scikit-image"
* pandas==1.1.4
* moviepy==1.0.3
* Cython==0.29.21
* seaborn==0.11.0

## Run Configurations

The run.py file contains example configurations which specify parameters for either mode: training, or experimental.

The different modes take different configuration parameters.

## Simulation Configurations

These can be found in the Configurations directory, along with python scripts that produce them. These are either single configurations for the assays, or scaffold configurations, which generate sequences of configurations used for training.


