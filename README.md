# Modelbuilder
Celery Python worker code wrapped on top of DSG workbench
For build and deployment guide, 
please refer to [here](https://github.home.247-inc.net/advancedprototypes/modelbuilder-web-service/wiki/Build-and-Deployment-Guide)

Modelbuilder is the actual model to which the training data is fed. It is majorly python based code.
It's main tasks are 
1) Normalization of the dataset.
2) Training
3) Accuracy calculation of the model.

using web2nl.

After building the model it generates the final resultant file.

## Pre-requisites

* Python (Refer [Python Doc](https://www.python.org/downloads/) for installation)
* Redis (Refer [Redis Doc](https://redis.io/download) for installation)

## Service Installation Guide

Note: 
* Execute all the below commands from project's root directory.
* Start the redis sentinel and slave as root:
    - Command for starting the sentinel in linux:
        * redis-server /etc/redis/redis-sentinel.conf --sentinel</br>
    - Command for starting the redis slave:
        * redis-server /etc/redis/slave.conf </br>

Step 1: Install package dependancies </br>

`pip install -r requirements.txt`

Step 2: Edit the celery config file (File location: {root directory}/src/celeryconfig.py) with correct sentinel address and master</br>
Step 3: Go inside src folder. </br>
* Source the env-setup script to set the environment variables using </br>
(Note: Please change the values in env-setup.sh according to your environment)

`source env-setup.sh`

* Start the celery worker using the command </br>

`celery -A Task worker --loglevel=info -Ofair`