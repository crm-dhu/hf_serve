#!/bin/bash 

export FLASK_APP=unmask
python3 -m flask run --port 8081 --host=0.0.0.0
