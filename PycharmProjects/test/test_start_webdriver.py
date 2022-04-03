import time
from selenium.webdriver.common.by import By
from selenium import webdriver

# Инициализируем драйвер браузера. После этой команды мы должны увидеть новое открытое окно браузера
driver = webdriver.Chrome()
# команда time.sleep устанавливает паузу в 5 секунд, чтобы мы успели увидеть, что происходит в браузере
time.sleep(5)

# Метод get сообщает браузеру, что нужно открыть сайт по указанной ссылке
driver.get("https://yandex.ru/")
time.sleep(5)

# Метод find_element позволяет найти нужный элемент на сайте, указав путь к нему.
textarea = driver.find_element(By.ID,"text")

# Напишем текст который, будем искать
textarea.send_keys("CPU")
time.sleep(5)

# Найдем кнопку для выполнения поиска
submit_button = driver.find_element(By.CLASS_NAME, 'search2__button')

# Скажем драйверу, что нужно нажать на кнопку. После этой команды мы должны увидеть страницу с инфой которую искали
submit_button.click()
time.sleep(5)

# Выведем ссылки, где есть название - CPU...
pages = driver.find_elements(By.PARTIAL_LINK_TEXT, 'CPU')
#hrefs_pages = driver.find_elements(By.XPATH, '//a[@href]') # получаем все ссылки на странице
for href in pages:
    print(href.get_attribute('href'))

# Перейдем по ссылке
# textarea = driver.find_element(By.LINK_TEXT,"Центральный процессор — Википедия").click()

# После выполнения всех действий мы должны не забыть закрыть окно браузера
# driver.quit()
