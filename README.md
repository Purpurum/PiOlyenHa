<div align="center">
  <img src="https://i.ibb.co/N9Sk9WX/baner2.png" alt="banner2" border="0" /></a>
</div>

## <div align="center">Стэк технологий📑</div>
<div align="center">
  <a href="https://www.python.org/doc/"><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"></a>
  <a href="https://pytorch.org/docs/stable/index.html"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"></a>
  <a href="https://opencv.github.io/cvat/docs/"><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"></a>
  <br>
  <a href="https://github.com/ultralytics/ultralytics?tab=readme-ov-file"><img src="https://img.shields.io/badge/Ultralytics-YOLOv8-purple.svg"></a>
  <a href="https://github.com/kivy/kivy"><img src="https://img.shields.io/badge/Kivy-cross platform GUI-green.svg"></a>
</div>

## <div align="center">О нашем решении📝</div>
<p>
Наше решение данного кейса является десктопным приложением, классифицирующим три вида животных: Кабаргу, Благородного оленя, Косулю.
Это приложение должно повысить удобство и скорость обработки изображений, получаемых с автоматических фотоловушек, работая в автономном режиме на PC оператора, без доступа в интернет. 
</p>

## <div align="center">Быстрый старт🎢</div>

####  Запуск приложения
<p>
  Скомпирилованное .exe приложение находится в папке relise. Для запуска приложения необходимо скачать архив этого проекта, и запустить .exe файл.
</p>
  
####  Альтернативный запуск через консоль
<p>
  Если вы решили попробовать нескомпилированное приложение, то вам необходимо:<br>
  &ensp; 1. Установить Python версии не меньше 3.9<br>
  &ensp; 2. В cmd "pip install -r requirements.txt"<br>
  &ensp; 3. Запустить PingApp.py<br>
</p>

#### Как это работает?
<p>
  После запуска приложения, пользователь видит понятный и интуитивынй итерфейс. Сценарий использования приложения такой:<br>
  &ensp; 1. Запустив приложение пользователь нажимает на кнопку "Выбрать фотоархив", в открывшемся проводнике он выбирает архив или папку с собранными данными и нажимает "Сохранить".<br>
  &ensp; 2. После выбора директории фотографий с фотографиями, пользователь нажимает кнопку "Сканировать фотографии".<br>
  &ensp; 3. Далее немного подождав (все зависит от объема данных), в центре экрана приложения отображается статистика по обработанному фотосету.<br>
  &ensp; 4. Затем у пользователя есть возможность сохранить результаты обработки фотографии моделью, с помощью кнопки "Сохранить отчет".
</p> 

#### Результат обработки
<p>
  ГРФИК
</p> 
</details>

## <div align="center">Детекция и классификация📸</div>
<p>
  Данное решение содержит два типа сетей: детектор и классификаторы.
  В качестве детектора ипользуется бесклассовая модель Yolov8n.
  В качестве классификаторов используется ансамбль ResNet50xt.
</p>
<div align="center">

  #### Схема работы приложения
  <p>
    <img src="res/sch.png" border="0" /></a>
  </p>
  <img src="" width="500" height="500"/>
</div>

## <div align="center">Результат работы моделей🔮</div>
<p>
  ТУТ СКОР
</p>

## <div align="center">Демонстрация работы🎞</div>
<p>
  ГИФКА
</p>

<div align="center">
  <img src="res/expl.gif" border="0" /></a>
</div>
