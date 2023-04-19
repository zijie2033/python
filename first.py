from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from time import sleep
import json
option = webdriver.ChromeOptions()
option.add_experimental_option('detach',True)
option.add_argument("--disable-notifications")
option.add_experimental_option("excludeSwitches", ["enable-automation","enable-logging"])
option.add_experimental_option('useAutomationExtension', False)
option.add_experimental_option("prefs", {"profile.password_manager_enabled": False, "credentials_enable_service": False})

s= Service('.\chromedriver.exe')
chromee=webdriver.Chrome(service=s,options=option)  
chromee.get("https://courseselection.ntust.edu.tw")
username = chromee.find_element(By.NAME,'UserName')
password = chromee.find_element(By.NAME,'Password')
btnlogin = chromee.find_element(By.NAME,'btnLogIn')
username.send_keys('B11007050')
password.send_keys('zxc507ASDFG...')
btnlogin.click()
chromee.implicitly_wait(5)

