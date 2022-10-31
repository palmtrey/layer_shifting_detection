# Title: twitch_stream.py
# Purpose: To capture live video streams from Twitch.tv as proof of concept for capturing Ultimaker
#   video feeds.
# Author: Cameron Palmer, campalme@clarkson.edu
# Last Modified: September 11th, 2022
# 
# Code modified from https://medium.com/swlh/scraping-live-stream-video-with-python-b7154b1fedde

import json
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from browsermobproxy import Server
import datetime
import sys

url = "https://twitch.tv"

#start proxy server
server = Server('/home/cameron/Documents/browsermob-proxy')
server.start()
proxy = server.create_proxy()

# selenium arguments
options = webdriver.ChromeOptions()
options.add_argument('headless')
# options.add_argument("--window-size=1920,1080");
options.add_argument("--proxy-server={0}".format(proxy.proxy))

caps = DesiredCapabilities.CHROME.copy()
caps['acceptSslCerts'] = True
caps['acceptInsecureCerts'] = True

proxy.new_har("twitch",options={'captureHeaders': True,'captureContent':True,'captureBinaryContent': True}) # tag network logs 
              
driver = webdriver.Chrome('chromedriver',options=options,desired_capabilities=caps)
driver.get(url)

# Start capturing
filename = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")+'.mpeg'

fetched = []
for ent in proxy.har['log']['entries']:
    _url = ent['request']['url']
    _response = ent['response']
    
    #make sure havent already downloaded this piece
    if _url in fetched:
        continue
        
    if _url.endswith('.ts'):
        #check if this url had a valid response, if not, ignore it
        if 'text' not in _response['content'].keys() or not _response['content']['text']:
            continue
            
        print(_url+'\n')
        r1 = requests.get(_url, stream=True)
        if(r1.status_code == 200 or r1.status_code == 206):

            # re-open output file to append new video
            with open(filename,'ab') as f:
                data = b''
                for chunk in r1.iter_content(chunk_size=1024):
                    if(chunk):
                        data += chunk
                f.write(data)
                fetched.append(_url)
        else:
            print("Received unexpected status code {}".format(r1.status_code))   