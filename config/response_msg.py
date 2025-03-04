import json 
import os

file = open(os.getcwd() + '/response_msg.json')
msg = json.load(file)