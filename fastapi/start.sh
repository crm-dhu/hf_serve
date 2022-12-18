#!/bin/bash 

python3 -m uvicorn unmask:app --reload --port 8081 --host=0.0.0.0
