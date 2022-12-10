from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import urllib.request
import os

search = "방 사진"   # 이미지 이름
count = 200    # 크롤링할 이미지 개수
saveurl = "./room_image/"  # 이미지들을 저장할 폴더 주소

## 셀레니움으로 구글 이미지 접속 후 이미지 검색

options = webdriver.ChromeOptions()
options.headless = True
options.add_argument("window-size=1920x1080")

chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('detach', True)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl")
elem = driver.find_element(By.NAME,"q")
elem.send_keys(search)

elem.send_keys(Keys.RETURN)

# 페이지 끝까지 스크롤 내리기
SCROLL_PAUSE_TIME = 1
# 스크롤 깊이 측정하기
last_height = driver.execute_script("return document.body.scrollHeight")

# 스크롤 끝까지 내리기

while True:

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # 페이지 로딩 기다리기
    time.sleep(SCROLL_PAUSE_TIME)
    # 더 보기 요소 있을 경우 클릭하기

    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:

        try:
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click()

        except:
            break

    last_height = new_height

#이미지 찾고 다운받기
images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")

for i in range(count):

    try:
        images[i].click() # 이미지 클릭
        time.sleep(1)

        imgUrl = driver.find_element(By.CSS_SELECTOR,".n3VNCb").get_attribute("src")
        urllib.request.urlretrieve(imgUrl, saveurl + str(i) + ".jpg")    # 이미지 다운

    except:
        pass
driver.close()