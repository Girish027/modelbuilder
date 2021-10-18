import json
import os

with open(os.environ.get('MODELBUILDER_WORKER_CONFIG'), 'r') as jsonfile:
    data = json.load(jsonfile)

