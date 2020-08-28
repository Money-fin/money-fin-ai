import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
import re
import time

DATA = "D:/datasets/moneyfin/SET_00013_20200101-20200630.csv"
TARGET_DATA_PATH = "D:/datasets/moneyfin/text/"
CHROME_DRIVER = "bin/chromedriver.exe"


class Crawler():
    
    def __init__(self):
        self.urls, self.labels = self._read_csv()

    def _get_options(self):
        options = Options()
        options.headless = True
        return options

    def _create_browser(self):
        options = self._get_options()
        browser = webdriver.Chrome(CHROME_DRIVER, options=options)
        return browser

    def _read_csv(self):
        data = pd.read_csv(DATA)
        urls = data["URL_ADDR"].values
        labels = data["SCR_CTGO_NM"].values

        return urls, labels

    def crawl(self):
        browser = self._create_browser()

        for i, url in enumerate(self.urls):
            if i < 18182:
                continue

            try:
                print(url)
                save_path = TARGET_DATA_PATH + f"{i:08d}.txt"
                browser.get(url)

                label = self.labels[i].strip()
                if label == "긍정":
                    label = 1
                    continue
                elif label == "부정":
                    label = 0
                else:
                    continue
                
                contents = browser.find_element_by_css_selector("#news_read")
            except:
                continue

            with open(save_path, "w", encoding="utf8") as f:
                f.write(str(label) + "\n")
                f.write(contents.text)

            time.sleep(1)

        browser.close()


def main():
    crawler = Crawler()
    crawler.crawl()

if __name__ == "__main__":
    main()
