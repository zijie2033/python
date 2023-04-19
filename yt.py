from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import selenium
import time
import pytube 
from pytube import YouTube
string=input("請輸入要搜尋的YT影片:")
option = webdriver.ChromeOptions()
option.add_experimental_option('detach',True)
option.add_argument("--disable-notifications")
option.add_experimental_option("excludeSwitches", ["enable-automation","enable-logging"])
option.add_experimental_option('useAutomationExtension', False)
option.add_experimental_option("prefs", {"profile.password_manager_enabled": False, "credentials_enable_service": False})
s= Service('.\chromedriver.exe')
chromee=webdriver.Chrome(service=s,options=option)  
chromee.get("https://www.youtube.com/results?search_query="+string) 
last=None
url=[]
count = 1
title=chromee.find_elements(By.ID,'thumbnail')
print(title)
for s in title:
    vid=s.get_attribute('href')
    if vid == last :
        continue
    if re.search('shorts',vid):
        continue
    yt = YouTube(vid)
    print(str(count)+'.'+yt.title)
    count=count+1
    url.append(vid)
if count==1:
    print('搜尋失敗')
    exit()
choose = int(input("請選擇要下載哪部影片(1~"+str(count-1)+")"))-1
yt=YouTube(url[choose])
mp4list=yt.streams.filter(progressive=False,file_extension='mp4')
mp4=mp4list.order_by('resolution').desc().first()
file=mp4.download()
print('下載成功')





