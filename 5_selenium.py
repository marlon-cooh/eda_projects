#!/usr/bin/env python3

from selenium import webdriver # type: ignore
import pandas as pd
import os

def testing():
    path = "/usr/bin/chromedriver"
    try:
        if os.path.exists(path):
            
            cService = webdriver.ChromeService(executable_path = path)
            driver = webdriver.Chrome(service = cService)
            driver.get("https://www.facebook.com/dannia.okampo")
            print(driver.title)
            driver.quit()
    except AttributeError:
        print("This caremonda raises 'str' object has no attribute 'capabilities'")
    

def run():
    testing()
    
if __name__ == "__main__":
    run()