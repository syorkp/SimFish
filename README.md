# SimFish

A simulated zebrafish larvae for a 2D environment created using deep Q learning. Developed from project at https://bitbucket.org/azylbertal/simfish/src/master/

To run:
  1. Install dependencies using "pip install requirements.txt"
  2. Compile cython module using "python setup.py build_ext --inplace" from the Tools directory.  
  3. Run the "run.py" file, from which the chosen environment name and trial number can be adjusted.
  4. This will create the necessary files (given a bespoke "create_configuration_{environment}.py" file)
  5. To test the environment, run the "run_environment_test.py" file. Note that for Linux this will require installation of tkinter, which can be done using "sudo apt-get install python3-tk"

To view graphs, install tensorboard and run 'tensorboard --logdir="./Output/{trial_directory}/logs"'

## Run Configurations

The run.py file contains example configurations which specify parameters for either mode: training, or experimental.

The different modes take different configuration parameters.

## Simulation Configurations

These can be found in the Configurations directory, along with python scripts that produce them. These are either single configurations for the assays, or scaffold configurations, which generate sequences of configurations used for training.


