import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

account = "113101032"
passwd = "Shiromine941207"

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get("https://cos.nycu.edu.tw/#/redirect/cosnew")
time.sleep(3)
username = driver.find_element(By.NAME, "account")
password = driver.find_element(By.NAME, "password")
username.send_keys(account)
password.send_keys(passwd)
password.send_keys(Keys.RETURN)
wait = WebDriverWait(driver, 10)
wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.label__text[href='#/redirect/cosnew']"))).click()
time.sleep(2)
tabs = driver.window_handles
driver.switch_to.window(tabs[-1])
time.sleep(1)
burger = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.navbar-burger")))
driver.execute_script("arguments[0].click();", burger)
time.sleep(2)
wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "填寫問卷"))).click()
time.sleep(2)
while True:
    try:
        fill_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "填寫")]')))
        fill_button.click()
        time.sleep(2)

        table_rows = driver.find_elements(By.XPATH, '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[1]/table/tr')

        for i in range(2,len(table_rows)+1):
            xpath = f'//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[1]/table/tr[{i}]/td[3]/input'
            wait.until(EC.element_to_be_clickable((By.XPATH, xpath))).click()

        choice_xpaths = [
            '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[2]/table/tr[3]/td[2]/label[1]',
            '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[2]/table/tr[5]/td[2]/label[1]',
            '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[2]/table/tr[7]/td[2]/label[2]/input',
            '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[2]/table/tr[9]/td[2]/label[1]',
            '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/div[2]/table/tr[11]/td[2]/label[2]',
        ]
        for xpath in choice_xpaths:
            wait.until(EC.element_to_be_clickable((By.XPATH, xpath))).click()
            time.sleep(0.5)

        submit_xpath = '//*[@id="questionnairefilling"]/section/div/div/div[2]/form/button[3]'
        wait.until(EC.element_to_be_clickable((By.XPATH, submit_xpath))).click()
        time.sleep(5)
        driver.refresh()
        time.sleep(5)

    except:
        print("error")
        break
print("🎉 所有問卷處理完畢")