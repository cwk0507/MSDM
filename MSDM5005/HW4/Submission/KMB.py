from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from datetime import datetime

chromepath = "C:\Python\Python38\chromedriver.exe"
KMB_url = "https://search.kmb.hk/KMBWebSite/index.aspx?lang=tc"

driver = webdriver.Chrome(chromepath)
driver.get(KMB_url)
WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "imgRouteSearchIcon")))

routes = ['269D','E34B','968']
stop_time = []

for route in routes:
    route_icon_path  = '//*[@id="imgRouteSearchIcon"]'
    route_icon = driver.find_element_by_xpath(route_icon_path)
    route_icon.click()
    time.sleep(2)
    
    route_textbox_path = '//*[@id="txtRoute"]'
    route_textbox = driver.find_element_by_xpath(route_textbox_path)
    route_textbox.clear()
    route_textbox.send_keys(route)
    time.sleep(2)
    
    search_button_path = '//*[@id="routeSearchButton"]'
    search_button = driver.find_element_by_xpath(search_button_path)
    search_button.click()
    time.sleep(2)
    
    origin = driver.find_element_by_xpath('//*[@id="busDetailsOrigin"]').text
    destination = driver.find_element_by_xpath('//*[@id="busDetailsDest"]').text
    
    stops_path = '/html/body/div[1]/div[2]/div[5]/table[2]/tbody/tr[4]/td/table/tbody/tr/td/div/div/div[1]/table/tbody/tr'
    stops = driver.find_elements_by_xpath(stops_path)
    for i in range(len(stops)):
        try:
            driver.execute_script("return arguments[0].scrollIntoView();", stops[i])
            stop_name = stops[i].find_element_by_class_name('routeStopStartBulletTextTD').text
        except:
            try:
                stop_name = stops[i].find_element_by_class_name('stopTd').text
            except:
                try:
                    stop_name = stops[i].find_element_by_class_name('routeStopEndBulletTextTD').text
                except:
                    continue
        print(f'Fetching {route} - {stop_name}')
        stops[i].click()
        time.sleep(2)
        time_path = '//*[@id="mapDiv_root"]/div[3]/div[1]/div[2]/div/table/tbody/tr[5]/td/table/tbody/tr'
        while True:
            try:    
                times = driver.find_elements_by_xpath(time_path)
                break
            except:
                continue
        c=0
        while times[1].text == '暫時無法提供' and c<10:
            c+=1
            driver.find_element_by_xpath('//*[@id="mapDiv_root"]/div[3]/div[1]/div[2]/div/table/tbody/tr[5]/td/table/tbody/tr[1]/td[2]/img').click()
            time.sleep(2)
            while True:
                try:
                    times = driver.find_elements_by_xpath(time_path)
                    break
                except:
                    continue
        t=[]
        for j in range(1,len(times)):
            t.append(times[j].text)
        stop_time.append([route,origin,destination,stop_name,datetime.now().strftime('%Y-%m-%d %H:%M:%S')]+[tt for tt in t])
    
    driver.find_element_by_xpath('//*[@id="p2pBusRouteDetailStartEndTr"]/td/table/tbody/tr[2]/td[4]/img').click()
    time.sleep(2)
    
    origin = driver.find_element_by_xpath('//*[@id="busDetailsOrigin"]').text
    destination = driver.find_element_by_xpath('//*[@id="busDetailsDest"]').text
    
    stops_path = '/html/body/div[1]/div[2]/div[5]/table[2]/tbody/tr[4]/td/table/tbody/tr/td/div/div/div[1]/table/tbody/tr'
    stops = driver.find_elements_by_xpath(stops_path)
    for i in range(len(stops)):
        try:
            driver.execute_script("return arguments[0].scrollIntoView();", stops[i])
            stop_name = stops[i].find_element_by_class_name('routeStopStartBulletTextTD').text
        except:
            try:
                stop_name = stops[i].find_element_by_class_name('stopTd').text
            except:
                try:
                    stop_name = stops[i].find_element_by_class_name('routeStopEndBulletTextTD').text
                except:
                    continue
        print(f'Fetching {route} - {stop_name}')
        stops[i].click()
        time.sleep(2)
        time_path = '//*[@id="mapDiv_root"]/div[3]/div[1]/div[2]/div/table/tbody/tr[5]/td/table/tbody/tr'
        while True:
            try:    
                times = driver.find_elements_by_xpath(time_path)
                break
            except:
                continue
        c=0
        while times[1].text == '暫時無法提供' and c<10:
            c+=1
            driver.find_element_by_xpath('//*[@id="mapDiv_root"]/div[3]/div[1]/div[2]/div/table/tbody/tr[5]/td/table/tbody/tr[1]/td[2]/img').click()
            time.sleep(2)
            while True:
                try:
                    times = driver.find_elements_by_xpath(time_path)
                    break
                except:
                    continue
        t=[]
        for j in range(1,len(times)):
            t.append(times[j].text)
        stop_time.append([route,origin,destination,stop_name,datetime.now().strftime('%Y-%m-%d %H:%M:%S')]+[tt for tt in t])
    

driver.close()    
header = ['Route','Origin','Destination','Stop_Name','Scraping Time','ETA1','ETA2','ETA3']
df = pd.DataFrame(stop_time, columns = header)
df.to_csv(f'kmb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',index=False)
