# Setup

* Create a new conda environment using the environment.yml file.

* Activate your environment and execute this commands:
  * pip install --force-reinstall charset-normalizer==3.1.0
  * pip install --force-reinstall numpy==1.23.0

* Put the train and test datasets in the root directory (or change the paths in the notebook).

* Change the comment in the first line of the ground truth to: filename,key,ts_num,ts_denom,tempo (NO //)

* Execute the Jupyter notebook.