# FAQ Telegram-бот на Python

### УрФУ ДПО "Программирование нейронных сетей"

Команда **K6**

**Состав команды:**
* Шустик Б. А.
* Торопов А. А.
* Хабибулин Т. Р.
* Михайленко К. А.
* Косолапов С. А.

### Локальная разработка

Для локальной разработки необходимо установить зависимости

```
pip intall -r requirements.txt
```

Создать файл `.env` по аналогии с файлом `.env.example` и ввести туда токен своего бота.

После чего можно запустить бота
```
python3 -m src.main
```

### Запуск в Docker-контейнера

Для запуска в Docker-контейнере необходимо ввести команду 
```
docker compose up
```

Если не нужно отслеживать логи проекта и запустить его в фоне, то нужно добавить флаг `-d`