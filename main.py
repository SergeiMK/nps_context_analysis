#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# ЗА ПРЕДЕЛАМИ АНКЕТЫ: КАК ЗВЕЗДЫ, ПОГОДА И НОВОСТИ ВЛИЯЮТ НА NPS
#
# Бинарная модель для предсказания детракторов (NPS 0-6)
# Модель: CatBoostClassifier
# Валидация: Time-based StratifiedGroupKFold
#
# Автор: Сергей Колмаков
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import gc
import re
import os
import joblib
import json
import numpy as np
import pandas as pd
import ephem
import holidays
from functools import lru_cache
from tqdm import tqdm
from itertools import combinations
from pathlib import Path
from io import StringIO
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier, Pool

# Опциональные библиотеки для расширенного функционала
try:
    from meteostat import Point, Daily
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False
    print("[WARN] Библиотека 'meteostat' не установлена. Погодные признаки, кроме длины дня, не будут рассчитаны. Установите: pip install meteostat")

try:
    from timezonefinder import TimezoneFinder
    TF = TimezoneFinder()
    TIMEZONEFINDER_AVAILABLE = True
except Exception:
    TF = None
    TIMEZONEFINDER_AVAILABLE = False
    print("[WARN] Библиотека 'timezonefinder' не установлена. Будет использован резервный словарь таймзон.")

try:
    import psutil
    # Ограничиваем использование RAM (60% от доступной), чтобы избежать сбоев на больших данных
    USED_RAM_LIMIT = int(psutil.virtual_memory().total * 0.60)
except Exception:
    USED_RAM_LIMIT = None

# ============================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ============================================================

# --- Пути к файлам ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Входные данные
# TODO: Укажите правильное имя вашего исходного файла с данными!
SOURCE_DATA_PATH = DATA_DIR / "your_source_dataset.csv"
EVENTS_TSV_PATH = DATA_DIR / "events.tsv"
KP_INDEX_PATH = DATA_DIR / "kp_index.json"
AP_INDEX_PATH = DATA_DIR / "ap_index.json"

# Выходные файлы
ENRICHED_DATA_SAVE_PATH = BASE_DIR / "enriched_nps_data.csv.gz"
MODEL_SAVE_PATH = BASE_DIR / 'catboost_detractor_model.joblib'

# --- Параметры моделирования и обогащения ---
RANDOM_SEED = 42
N_SPLITS_CV = 8

# --- Константы для астрологических расчетов ---
ASTRO_TENSION_WINDOW = [5]
PLANETS_TO_TRACK = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
ASPECT_PLANETS = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
PERSONAL_PLANETS = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars']
HEAVY_PLANETS = ['Saturn', 'Uranus', 'Neptune', 'Pluto']


# ============================================================
# СПРАВОЧНИКИ (Регионы, Таймзоны, Координаты)
# ============================================================
REGION_GENT_TO_NOM = {
    'Московской области': 'Московская область', 'Ленинградской области': 'Ленинградская область', 'Республике Татарстан': 'Республика Татарстан',
    'Свердловской области': 'Свердловская область', 'Краснодарском крае': 'Краснодарский край', 'Ростовской области': 'Ростовская область',
    'Новосибирской области': 'Новосибирская область', 'Красноярском крае': 'Красноярский край', 'Воронежской области': 'Воронежская область',
    'Самарской области': 'Самарская область', 'Пермском крае': 'Пермский край', 'Нижегородской области': 'Нижегородская область',
    'Саратовской области': 'Саратовская область', 'Иркутской области': 'Иркутская область', 'Челябинской области': 'Челябинская область',
    'Алтайском крае': 'Алтайский край', 'Республике Башкортостан': 'Республика Башкортостан', 'Кемеровской области': 'Кемеровская область — Кузбасс',
    'Оренбургской области': 'Оренбургская область', 'Ставропольском крае': 'Ставропольский край', 'Приморском крае': 'Приморский край',
    'Хабаровском крае': 'Хабаровский край', 'Калининградской области': 'Калининградская область', 'Тюменской области': 'Тюменская область',
    'Ханты-Мансийском АО': 'Ханты-Мансийский автономный округ — Югра', 'Орловской области': 'Орловская область', 'Республике Карелия': 'Республика Карелия',
    'Удмуртской Республике': 'Удмуртская Республика', 'Волгоградской области': 'Волгоградская область', 'Республике Дагестан': 'Республика Дагестан',
    'Ярославской области': 'Ярославская область', 'Тверской области': 'Тверская область', 'Архангельской области': 'Архангельская область',
    'Омской области': 'Омская область', 'Республике Бурятия': 'Республика Бурятия', 'Республике Саха (Якутия)': 'Республика Саха (Якутия)',
    'Республике Мордовия': 'Республика Мордовия', 'Ивановской области': 'Ивановская область', 'Белгородской области': 'Белгородская область',
    'Тульской области': 'Тульская область', 'Забайкальском крае': 'Забайкальский край', 'Владимирской области': 'Владимирская область',
    'Тамбовской области': 'Тамбовская область', 'Пензенской области': 'Пензенская область', 'Ульяновской области': 'Ульяновская область',
    'Вологодской области': 'Вологодская область', 'Кировской области': 'Кировская область', 'Брянской области': 'Брянская область',
    'Чеченской республике': 'Чеченская Республика', 'Липецкой области': 'Липецкая область', 'Чувашской Республике': 'Чувашская Республика',
    'Курской области': 'Курская область', 'Рязанской области': 'Рязанская область', 'Калужской области': 'Калужская область',
    'Томской области': 'Томская область', 'Смоленской области': 'Смоленская область', 'Астраханской области': 'Астраханская область',
    'Республике Коми': 'Республика Коми', 'Курганской области': 'Курганская область', 'Мурманской области': 'Мурманская область',
    'Амурской области': 'Амурская область', 'Кабардино-Балкарской Республике': 'Кабардино-Балкарская Республика',
    'Республике Марий Эл': 'Республика Марий Эл', 'Псковской области': 'Псковская область', 'Новгородской области': 'Новгородская область',
    'Костромской области': 'Костромская область', 'Ямало-Ненецком АО': 'Ямало-Ненецкий автономный округ', 'Северной Осетии-Алании': 'Республика Северная Осетия — Алания',
    'Республике Хакасия': 'Республика Хакасия', 'Сахалинской области': 'Сахалинская область', 'Карачаево-Черкесской Республике': 'Карачаево-Черкесская Республика',
    'Республике Ингушетия': 'Ингушетия', 'Камчатском крае': 'Камчатский край', 'Республике Тыва': 'Республика Тыва',
    'Республике Калмыкия': 'Республика Калмыкия', 'Республике Алтай': 'Республика Алтай', 'Магаданской области': 'Магаданская область'
}

RU_TZ_BY_REGION = {
    'Москва': 'Europe/Moscow', 'Санкт-Петербург': 'Europe/Moscow', 'Калининградская область': 'Europe/Kaliningrad',
    'Московская область': 'Europe/Moscow', 'Ленинградская область': 'Europe/Moscow', 'Республика Карелия': 'Europe/Moscow',
    'Республика Коми': 'Europe/Moscow', 'Архангельская область': 'Europe/Moscow', 'Мурманская область': 'Europe/Moscow',
    'Вологодская область': 'Europe/Moscow', 'Новгородская область': 'Europe/Moscow', 'Псковская область': 'Europe/Moscow',
    'Тверская область': 'Europe/Moscow', 'Ярославская область': 'Europe/Moscow', 'Костромская область': 'Europe/Moscow',
    'Ивановская область': 'Europe/Moscow', 'Владимирская область': 'Europe/Moscow', 'Рязанская область': 'Europe/Moscow',
    'Тульская область': 'Europe/Moscow', 'Калужская область': 'Europe/Moscow', 'Смоленская область': 'Europe/Moscow',
    'Брянская область': 'Europe/Moscow', 'Орловской области': 'Europe/Moscow', 'Курская область': 'Europe/Moscow',
    'Белгородская область': 'Europe/Moscow', 'Воронежская область': 'Europe/Moscow', 'Липецкая область': 'Europe/Moscow',
    'Тамбовская область': 'Europe/Moscow', 'Нижегородская область': 'Europe/Moscow', 'Кировская область': 'Europe/Moscow',
    'Пензенская область': 'Europe/Moscow', 'Республика Мордовия': 'Europe/Moscow', 'Республика Марий Эл': 'Europe/Moscow',
    'Чувашская Республика': 'Europe/Moscow', 'Республика Татарстан': 'Europe/Moscow', 'Республика Калмыкия': 'Europe/Moscow',
    'Ростовская область': 'Europe/Moscow', 'Краснодарский край': 'Europe/Moscow', 'Ставропольский край': 'Europe/Moscow',
    'Ингушетия': 'Europe/Moscow', 'Кабардино-Балкарская Республика': 'Europe/Moscow', 'Карачаево-Черкесская Республика': 'Europe/Moscow',
    'Республика Северная Осетия — Алания': 'Europe/Moscow', 'Чеченская Республика': 'Europe/Moscow', 'Республика Дагестан': 'Europe/Moscow',
    'Волгоградская область': 'Europe/Volgograd', 'Самарская область': 'Europe/Samara', 'Удмуртская Республика': 'Europe/Samara',
    'Астраханская область': 'Europe/Astrakhan', 'Саратовская область': 'Europe/Saratov', 'Ульяновская область': 'Europe/Ulyanovsk',
    'Республика Башкортостан': 'Asia/Yekaterinburg', 'Пермский край': 'Asia/Yekaterinburg', 'Свердловская область': 'Asia/Yekaterinburg',
    'Челябинская область': 'Asia/Yekaterinburg', 'Курганская область': 'Asia/Yekaterinburg', 'Тюменская область': 'Asia/Yekaterinburg',
    'Ханты-Мансийский автономный округ — Югра': 'Asia/Yekaterinburg', 'Ямало-Ненецкий автономный округ': 'Asia/Yekaterinburg',
    'Оренбургская область': 'Asia/Yekaterinburg', 'Омская область': 'Asia/Omsk', 'Новосибирская область': 'Asia/Novosibirsk',
    'Томская область': 'Asia/Tomsk', 'Кемеровская область — Кузбасс': 'Asia/Novokuznetsk', 'Алтайский край': 'Asia/Barnaul',
    'Республика Алтай': 'Asia/Barnaul', 'Красноярский край': 'Asia/Krasnoyarsk', 'Республика Хакасия': 'Asia/Krasnoyarsk',
    'Республика Тыва': 'Asia/Krasnoyarsk', 'Иркутская область': 'Asia/Irkutsk', 'Республика Бурятия': 'Asia/Irkutsk',
    'Забайкальский край': 'Asia/Chita', 'Амурская область': 'Asia/Yakutsk', 'Республика Саха (Якутия)': 'Asia/Yakutsk',
    'Приморский край': 'Asia/Vladivostok', 'Хабаровский край': 'Asia/Vladivostok', 'Сахалинская область': 'Asia/Sakhalin',
    'Магаданская область': 'Asia/Magadan', 'Камчатский край': 'Asia/Kamchatka'
}

REGION_COORDS_MASTER = {
    'Москва': [(55.7558, 37.6173), (55.8279, 37.6378), (55.5451, 37.5494), (55.9851, 37.2091)],
    'Санкт-Петербург': [(59.9343, 30.3351), (60.0513, 30.3300), (59.7214, 30.4140), (60.0070, 29.7710)],
    'Калининградская область': [(54.7101, 20.5101), (55.1583, 21.9333), (54.3167, 20.4833), (54.6333, 22.7000)],
    'Белгородская область': [(50.5950, 36.5870), (51.2180, 37.8680), (50.6000, 35.2920), (50.1830, 36.9000)],
    'Брянская область': [(53.2430, 34.3640), (53.7910, 33.2840), (52.5710, 32.0390), (52.5690, 35.1580)],
    'Владимирская область': [(56.1290, 40.4060), (56.4280, 38.8680), (55.5800, 42.0640), (56.3280, 41.3120)],
    'Ивановская область': [(56.9970, 40.9760), (57.4660, 41.9160), (56.5910, 40.3540), (57.3400, 43.1970)],
    'Калужская область': [(54.5150, 36.2610), (55.2200, 36.6340), (54.0620, 34.2980), (53.9670, 36.1770)],
    'Костромская область': [(57.7670, 40.9260), (58.5520, 46.4670), (59.2000, 42.2830), (57.8000, 40.1330)],
    'Курская область': [(51.7380, 36.1850), (52.3330, 35.2830), (51.2290, 34.4530), (51.5830, 37.6000)],
    'Липецкая область': [(52.6090, 39.5980), (53.2500, 38.9330), (52.0330, 40.1000), (52.5830, 37.8830)],
    'Орловская область': [(52.9690, 36.0680), (53.3330, 37.0000), (52.3670, 35.2670), (52.7170, 37.8670)],
    'Рязанская область': [(54.6260, 39.7360), (55.0990, 41.4160), (53.9500, 39.4830), (54.2670, 38.7170)],
    'Смоленская область': [(54.7820, 32.0450), (55.6500, 33.4670), (53.9170, 31.3330), (54.4000, 33.2500)],
    'Тамбовская область': [(52.7210, 41.4520), (53.2330, 41.4330), (51.8170, 42.7170), (52.6170, 40.0500)],
    'Тульская область': [(54.1930, 37.6170), (54.0680, 38.9660), (53.2500, 36.5670), (53.9170, 38.3000)],
    'Ярославская область': [(57.6260, 39.8840), (58.0500, 38.8330), (57.1830, 38.8500), (56.7330, 38.9670)],
    'Республика Адыгея': [(44.6098, 40.1005), (44.991, 40.178), (44.293, 40.105)],
    'Республика Ингушетия': [(43.1670, 44.8130), (43.2170, 44.7670), (43.3170, 45.0000), (42.8670, 45.0170)],
    'Кабардино-Балкарская Республика': [(43.4840, 43.6070), (43.6000, 44.0500), (43.9000, 43.1330), (43.3170, 42.5000)],
    'Карачаево-Черкесская Республика': [(44.2250, 42.0460), (44.2230, 41.5900), (43.5670, 41.2670), (43.7830, 42.4500)],
    'Республика Северная Осетия — Алания': [(43.0240, 44.6810), (43.3670, 44.3170), (42.8170, 44.6000), (42.7170, 43.8000)],
    'Чеченская Республика': [(43.3180, 45.6940), (43.7000, 46.5670), (43.4170, 44.8330), (42.7170, 46.0330)],
    'Республика Марий Эл': [(56.6340, 47.8960), (56.3170, 49.9830), (56.7170, 46.5500), (56.0830, 47.9000)],
    'Республика Мордовия': [(54.1800, 45.1860), (54.6330, 46.4170), (54.7000, 43.2170), (53.8000, 44.4500)],
    'Чувашская Республика': [(56.1330, 47.2500), (55.5000, 47.4670), (55.8500, 46.3330), (55.4330, 48.2500)],
    'Новгородская область': [(58.5210, 31.2750), (58.3830, 33.3000), (57.6500, 30.3170), (57.9830, 34.0000)],
    'Псковская область': [(57.8190, 28.3310), (56.2830, 29.9830), (57.3330, 27.8170), (58.7330, 29.1830)],
    'Московская область': [(55.7558, 37.6173), (56.315, 38.136), (55.432, 38.764), (55.602, 36.855), (56.716, 37.202), (54.887, 37.478), (55.150, 39.516), (56.000, 35.933), (55.933, 38.983)],
    'Ленинградская область': [(59.9343, 30.3351), (60.716, 33.541), (59.383, 31.983), (59.750, 29.083), (60.533, 28.750), (58.733, 31.100), (59.417, 33.550), (61.000, 30.250), (59.083, 35.100)],
    'Архангельская область': [(64.5401, 40.5433), (61.261, 46.653), (65.85, 47.78), (66.08, 43.19), (62.36, 37.6), (71.63, 52.48), (80.75, 58.05), (63.93, 41.76), (63.07, 47.59)],
    'Вологодская область': [(59.220, 39.891), (60.75, 38.56), (59.13, 36.43), (60.03, 46.30), (58.98, 35.08), (60.91, 46.31), (58.58, 40.16), (60.13, 35.11), (58.9, 43.1)],
    'Мурманская область': [(68.970, 33.074), (67.58, 30.68), (67.62, 38.03), (66.53, 34.57), (69.2, 30.81), (66.07, 39.8), (68.43, 35.9), (67.14, 32.41)],
    'Республика Карелия': [(61.785, 34.346), (66.3, 31.5), (62.7, 36.5), (61.3, 30.6), (64.21, 30.35), (61.8, 32.76), (63.16, 32.25), (61.7, 36.4)],
    'Республика Коми': [(61.673, 50.809), (67.49, 64.03), (63.56, 53.70), (60.15, 49.58), (65.01, 57.23), (59.58, 49.65), (66.04, 60.15), (62.3, 47.9), (63.2, 59.1)],
    'Краснодарский край': [(45.0355, 38.9753), (43.5855, 39.7203), (44.7244, 37.7678), (45.0178, 41.1236), (46.4069, 38.2736), (43.912, 40.205), (45.922, 40.589), (44.298, 37.319), (44.594, 39.083)],
    'Астраханская область': [(46.349, 48.040), (48.06, 46.36), (45.7, 47.16), (47.2, 48.8), (47.18, 46.85), (46.2, 49.1), (48.5, 47.5), (45.9, 48.5)],
    'Волгоградская область': [(48.708, 44.514), (50.08, 45.4), (48.4, 43.5), (50.5, 42.1), (47.8, 46.8), (51.18, 44.28), (49.6, 46.5), (47.78, 44.83)],
    'Ростовская область': [(47.235, 39.713), (49.03, 42.45), (46.5, 40.2), (48.7, 38.9), (49.9, 41.3), (46.8, 42.7), (47.5, 38.3), (46.4, 38.9)],
    'Нижегородская область': [(56.296, 43.936), (57.6, 45.9), (55.5, 43.4), (55.3, 45.9), (58.05, 44.9), (56.7, 42.2), (55.0, 42.1), (56.9, 45.1)],
    'Оренбургская область': [(51.768, 55.097), (51.22, 58.56), (52.42, 52.16), (51.48, 61.42), (53.5, 52.8), (50.8, 52.1), (50.7, 56.4), (52.6, 56.5)],
    'Пензенская область': [(53.195, 45.019), (53.8, 46.3), (52.7, 43.1), (54.1, 43.4), (52.5, 45.7), (53.2, 42.6), (53.9, 44.6), (52.9, 44.0)],
    'Самарская область': [(53.195, 50.106), (54.4, 51.3), (52.5, 50.3), (53.1, 48.4), (54.1, 49.6), (52.3, 51.7), (53.5, 51.5), (52.9, 51.9)],
    'Саратовская область': [(51.533, 46.034), (52.3, 47.8), (51.2, 43.8), (52.8, 45.2), (50.7, 48.3), (52.0, 49.5), (51.6, 48.8), (50.6, 44.8)],
    'Республика Татарстан': [(55.796, 49.108), (55.61, 52.31), (54.89, 48.36), (55.20, 50.62), (56.4, 50.4), (54.8, 52.1), (54.1, 49.5), (55.9, 47.5)],
    'Ульяновская область': [(54.314, 48.403), (54.2, 46.4), (53.1, 47.9), (53.5, 49.6), (55.0, 47.2), (53.9, 50.0), (52.8, 46.9), (54.8, 49.5)],
    'Пермский край': [(58.010, 56.250), (59.39, 57.00), (57.65, 55.28), (60.41, 56.51), (60.2, 59.1), (57.1, 56.9), (58.6, 54.6), (59.2, 54.8), (56.5, 55.3)],
    'Республика Башкортостан': [(54.735, 55.957), (53.58, 58.63), (56.08, 54.76), (52.70, 55.03), (55.9, 58.1), (52.9, 53.4), (53.6, 53.6), (55.0, 53.8)],
    'Свердловская область': [(56.838, 60.605), (57.77, 63.30), (59.64, 59.95), (56.50, 58.25), (58.7, 61.6), (56.4, 63.8), (58.1, 58.1), (56.1, 59.1)],
    'Тюменская область': [(57.153, 65.535), (56.15, 69.48), (58.48, 68.25), (56.48, 64.03), (58.3, 70.3), (57.5, 62.5), (55.6, 66.8), (55.8, 68.9)],
    'Ханты-Мансийский автономный округ — Югра': [(61.002, 69.018), (60.93, 76.55), (62.13, 66.08), (59.61, 63.58), (63.3, 78.4), (59.9, 71.4), (62.2, 74.5), (61.25, 63.3)],
    'Челябинская область': [(55.159, 61.402), (53.4, 59.1), (54.1, 62.9), (54.7, 59.1), (56.0, 59.7), (53.1, 61.4), (54.8, 63.0), (55.7, 60.5)],
    'Ямало-Ненецкий автономный округ': [(66.535, 66.614), (66.08, 81.25), (70.13, 72.48), (65.25, 78.41), (72.5, 78.5), (63.9, 74.5), (68.1, 74.6), (64.9, 68.4)],
    'Алтайский край': [(53.347, 83.778), (52.6, 81.1), (51.5, 84.4), (53.9, 85.9), (53.4, 79.8), (52.0, 85.8), (51.1, 82.2), (52.9, 86.1)],
    'Иркутская область': [(52.286, 104.281), (56.26, 101.63), (55.08, 98.91), (57.33, 108.11), (59.5, 109.3), (51.8, 104.8), (58.8, 114.9), (54.1, 106.1)],
    'Кемеровская область — Кузбасс': [(55.355, 86.087), (53.75, 87.11), (56.1, 87.9), (54.2, 86.7), (56.8, 86.1), (52.9, 87.2), (55.2, 88.8), (53.2, 85.6)],
    'Новосибирская область': [(55.008, 82.935), (54.31, 78.33), (56.33, 84.18), (55.43, 75.25), (53.9, 81.8), (56.6, 77.8), (54.5, 84.3), (55.9, 80.4)],
    'Омская область': [(54.988, 73.368), (56.8, 71.8), (53.9, 74.6), (58.3, 74.3), (55.5, 70.4), (53.4, 72.0), (57.5, 75.2), (56.3, 75.4)],
    'Томская область': [(56.501, 84.992), (59.1, 77.1), (57.8, 82.9), (56.9, 86.8), (60.9, 78.3), (55.9, 84.3), (57.4, 86.1), (58.5, 88.2)],
    'Республика Алтай': [(51.958, 85.960), (49.7, 87.6), (50.9, 84.7), (52.3, 87.3), (50.3, 88.6), (49.4, 85.9), (51.4, 87.2), (50.4, 86.4)],
    'Республика Бурятия': [(51.834, 107.584), (55.8, 109.3), (50.3, 106.4), (53.5, 112.5), (50.4, 103.7), (56.4, 113.5), (52.3, 110.8), (51.5, 116.1)],
    'Республика Тыва': [(51.719, 94.445), (50.6, 90.5), (50.0, 95.9), (53.2, 95.0), (50.7, 97.5), (52.4, 96.1), (50.2, 92.1), (51.5, 96.5)],
    'Республика Хакасия': [(53.719, 91.442), (52.8, 89.8), (54.4, 90.1), (54.5, 91.6), (51.9, 90.3), (53.4, 89.1), (53.1, 91.0)],
    'Красноярский край': [(56.0105, 92.8526), (69.353, 88.202), (58.461, 92.170), (53.721, 91.442), (56.497, 86.148), (66.52, 100.19), (55.355, 95.707), (61.685, 95.839), (53.94, 95.75)],
    'Амурская область': [(50.291, 127.527), (55.1, 128.5), (53.9, 124.7), (49.4, 130.4), (51.7, 123.9), (52.8, 132.8), (54.6, 120.9), (50.5, 125.8)],
    'Забайкальский край': [(52.033, 113.500), (53.65, 119.63), (50.53, 116.85), (50.88, 108.76), (56.3, 120.5), (50.2, 119.2), (54.3, 116.7), (51.7, 110.1)],
    'Камчатский край': [(53.045, 158.650), (58.65, 161.30), (55.93, 161.98), (51.53, 156.63), (62.5, 166.1), (53.9, 155.8), (57.1, 156.8), (55.2, 155.6)],
    'Магаданская область': [(59.557, 150.808), (62.5, 152.3), (63.0, 147.2), (61.9, 156.1), (64.8, 158.4), (59.6, 145.4), (62.7, 160.0), (60.1, 153.7)],
    'Приморский край': [(43.1332, 131.9113), (45.1667, 136.8833), (44.0500, 132.8833), (42.8500, 131.1400), (42.823, 132.91), (46.13, 134.75), (44.58, 135.53), (43.36, 134.69), (43.76, 135.2)],
    'Сахалинская область': [(46.959, 142.738), (50.9, 143.1), (45.9, 142.1), (46.6, 150.9), (53.3, 142.9), (45.1, 147.8), (48.5, 142.7), (43.8, 145.5)],
    'Хабаровский край': [(48.472, 135.088), (52.26, 136.53), (50.25, 136.91), (59.35, 140.48), (56.34, 138.16), (49.03, 140.25), (46.96, 138.25), (51.48, 140.73), (54.08, 132.55)],
    'Республика Саха (Якутия)': [(62.0282, 129.7331), (67.450, 133.383), (70.633, 118.267), (59.367, 112.567), (71.626, 128.869), (63.033, 118.300), (64.567, 143.217), (56.650, 124.700), (69.410, 147.92)],
}

def region_to_tz(region_name: str) -> str:
    """Определяет таймзону по названию региона, используя timezonefinder или резервный словарь."""
    if TIMEZONEFINDER_AVAILABLE and TF is not None:
        coords_list = REGION_COORDS_MASTER.get(region_name)
        if coords_list:
            lat, lon = coords_list[0]
            tz = TF.timezone_at(lat=lat, lon=lon)
            if tz:
                return tz
    # Резервный вариант, если timezonefinder не сработал
    return RU_TZ_BY_REGION.get(region_name, 'Europe/Moscow')


# ============================================================
# ==== БЛОК 1: АСТРОЛОГИЯ ====
# ============================================================
@lru_cache(maxsize=10000)
def get_planet_details(planet_name, ephem_date):
    planet_map = {'Sun': ephem.Sun, 'Moon': ephem.Moon, 'Mercury': ephem.Mercury, 'Venus': ephem.Venus, 'Mars': ephem.Mars, 'Jupiter': ephem.Jupiter, 'Saturn': ephem.Saturn, 'Uranus': ephem.Uranus, 'Neptune': ephem.Neptune, 'Pluto': ephem.Pluto}
    planet_obj = planet_map[planet_name](); planet_obj.compute(ephem_date)
    ecl = ephem.Ecliptic(planet_obj); lon = ecl.lon
    speed = 0.0
    if planet_name not in ['Sun', 'Moon']:
        planet_obj.compute(ephem_date - 0.5); lon1 = ephem.Ecliptic(planet_obj).lon
        planet_obj.compute(ephem_date + 0.5); lon2 = ephem.Ecliptic(planet_obj).lon
        speed = np.degrees(lon2 - lon1)
        if speed > 180: speed -= 360
        elif speed < -180: speed += 360
    signs = ['Овен','Телец','Близнецы','Рак','Лев','Дева','Весы','Скорпион','Стрелец','Козерог','Водолей','Рыбы']
    sign_name = signs[int(np.degrees(lon) // 30)]
    return {'speed': float(speed), 'sign': sign_name, 'lon': lon}

@lru_cache(maxsize=50000)
def get_astro_cats_only(date: pd.Timestamp, tz: str):
    local_noon_utc = date.replace(hour=12).tz_localize(tz).tz_convert('UTC')
    d = ephem.Date(local_noon_utc.to_pydatetime()); d_prev = ephem.Date(d - 1)
    moon_details = get_planet_details('Moon', d)
    moon = ephem.Moon(d); pct = float(moon.phase)
    prev_new_moon, next_new_moon = ephem.previous_new_moon(d), ephem.next_new_moon(d)
    age = d - prev_new_moon
    lunar_day_num = int(np.floor(age)) + 1
    def moon_phase_name_from_pct(p):
        if p < 5 or p > 95: return 'Новолуние'
        if p < 45: return 'Растущая'
        if p < 55: return 'Полнолуние'
        return 'Убывающая'
    def bin_lunar_day(n):
        if n <= 7: return '1-7'
        if n <= 14: return '8-14'
        if n <= 21: return '15-21'
        return '22-30'
    moon_phase_cat = moon_phase_name_from_pct(pct)
    lunar_day_cat = bin_lunar_day(lunar_day_num)
    is_hecate = 1 if (next_new_moon - d) < 2.5 else 0
    moon_lon_in_sign = np.degrees(moon_details['lon']) % 30
    is_moon_voc_proxy = 1 if moon_lon_in_sign > 27 else 0
    sun_details = get_planet_details('Sun', d); sun_sign_cat = sun_details['sign']
    solar_eclipses_utc = pd.to_datetime(['2022-10-25','2023-04-20','2024-04-08','2024-10-02'], utc=True)
    lunar_eclipses_utc = pd.to_datetime(['2022-05-16','2022-11-08','2023-05-05','2023-10-28','2024-03-25','2024-09-18'], utc=True)
    try:
        solar_dates = set(solar_eclipses_utc.tz_convert(tz).date)
        lunar_dates = set(lunar_eclipses_utc.tz_convert(tz).date)
    except Exception:
        solar_dates = set(solar_eclipses_utc.date)
        lunar_dates = set(lunar_eclipses_utc.date)
    local_date = date.date()
    is_solar_in_window = any(abs((local_date - dd).days) <= 3 for dd in solar_dates)
    is_lunar_in_window = any(abs((local_date - dd).days) <= 3 for dd in lunar_dates)
    solar_ecl_cat = 'да' if is_solar_in_window else 'нет'
    lunar_ecl_cat = 'да' if is_lunar_in_window else 'нет'
    is_retro_any, is_station_any, is_ingress_any = 0, 0, 0
    is_retro_mercury = 0
    for p_name in PLANETS_TO_TRACK:
        today, yesterday = get_planet_details(p_name, d), get_planet_details(p_name, d_prev)
        if today['speed'] < 0:
            is_retro_any = 1
            if p_name == 'Mercury': is_retro_mercury = 1
        if (today['speed'] < 0) != (yesterday['speed'] < 0): is_station_any = 1
        if today['sign'] != yesterday['sign']: is_ingress_any = 1
    hard_aspects_count, personal_hard, weighted_hard_score = 0, 0, 0
    hard_aspect_to_mars = 0
    planet_positions = {p: get_planet_details(p, d) for p in ASPECT_PLANETS}
    for p1, p2 in combinations(ASPECT_PLANETS, 2):
        angle = abs(np.degrees(planet_positions[p1]['lon'] - planet_positions[p2]['lon']))
        angle = min(angle, 360 - angle)
        orb = 8 if ('Sun' in [p1,p2] or 'Moon' in [p1,p2]) else 6
        if abs(angle - 180) <= orb or abs(angle - 90) <= orb:
            hard_aspects_count += 1
            if p1 in PERSONAL_PLANETS and p2 in PERSONAL_PLANETS: personal_hard += 1
            weight = 2 if (p1 in HEAVY_PLANETS or p2 in HEAVY_PLANETS) else 1
            weighted_hard_score += weight
            if 'Mars' in [p1, p2]: hard_aspect_to_mars = 1
    interaction_merc_mars = 1 if (is_retro_mercury == 1 and hard_aspect_to_mars == 1) else 0
    def yesno(x): return 'да' if x == 1 else 'нет'
    def bin_hard(n): return '0' if n == 0 else ('1' if n == 1 else '>=2')
    def bin_weighted_hard(s):
        if s == 0: return '0'
        if s <= 2: return '1-2'
        if s <= 4: return '3-4'
        return '5+'
    return {
        'знак_Солнца': sun_sign_cat, 'фаза_луны': moon_phase_cat, 'lunar_day_cat': lunar_day_cat,
        'hard_aspects_cat': bin_hard(hard_aspects_count), 'pp_hard_aspects_cat': bin_hard(personal_hard),
        'is_retrograde_any_cat': yesno(is_retro_any), 'is_station_any_cat': yesno(is_station_any),
        'is_ingress_any_cat': yesno(is_ingress_any), 'солнечное_затмение_cat': solar_ecl_cat,
        'лунное_затмение_cat': lunar_ecl_cat, 'is_hecate_moon_cat': yesno(is_hecate),
        'astro_weighted_aspects_cat': bin_weighted_hard(weighted_hard_score),
        'astro_moon_voc_proxy_cat': yesno(is_moon_voc_proxy),
        'astro_interaction_merc_mars': yesno(interaction_merc_mars),
    }


# ============================================================
# ==== БЛОК 2: КАЛЕНДАРЬ ====
# ============================================================
CALENDAR_COMPACT_CAT = [
    'cal_day_type3','cal_weekday_4','cal_week_of_month','cal_month_phase3',
    'cal_quarter','cal_season','cal_holiday_type4','cal_holiday_window5',
    'cal_long_weekend','cal_school_break_ext', 'cal_payday_proximity', 'cal_is_blue_monday'
]

def add_calendar_compact_features(df: pd.DataFrame, date_col_local='day_local', date_col_msk='msk_day') -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col_msk] if date_col_msk in out.columns else out[date_col_local], errors='coerce')
    def map_weekday_bucket(x):
        if pd.isna(x): return 'NA'
        x = int(x)
        if x in (0,1,2): return 'пн-ср'
        if x in (3,4):   return 'чт-пт'
        if x == 5:       return 'сб'
        return 'вс'
    out['cal_weekday_4'] = d.dt.dayofweek.map(map_weekday_bucket).astype('category')
    first_dom = d - pd.to_timedelta(d.dt.day - 1, unit='D')
    wom = 1 + ((d.dt.day + first_dom.dt.dayofweek - 1) // 7)
    out['cal_week_of_month'] = wom.clip(lower=1, upper=5).map(lambda x: f'W{int(x)}' if pd.notna(x) else 'NA').astype('category')
    def month_phase(day):
        if pd.isna(day): return 'NA'
        day = int(day)
        if day <= 10: return 'начало'
        if day <= 20: return 'середина'
        return 'конец'
    out['cal_month_phase3'] = d.dt.day.map(month_phase).astype('category')
    out['cal_quarter'] = d.dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'}).astype('category')
    out['cal_season']  = d.dt.month.map(lambda m: 'Зима' if m in (12,1,2) else ('Весна' if m in (3,4,5) else ('Лето' if m in (6,7,8) else 'Осень'))).astype('category')
    cal_dates = pd.date_range(d.min(), d.max(), freq='D') if d.notna().any() else pd.DatetimeIndex([])
    if len(cal_dates) > 0:
        ru_hol = holidays.Russia(years=sorted(cal_dates.year.unique()))
        cal_df = pd.DataFrame(index=cal_dates); cal_df.index.name = 'cal_day'
        cal_df['is_holiday'] = cal_df.index.to_series().map(lambda x: 1 if x.date() in ru_hol else 0).astype(int)
        cal_df['hol_name'] = cal_df.index.to_series().map(lambda x: ru_hol.get(x.date(), '')).astype(str)
        def map_holiday_type(name):
            if not name or str(name).strip()=='':
                return 'нет'
            low = str(name).lower()
            majors = ['новый год', '23 февраля', '8 марта', '1 мая', 'день победы', '9 мая', 'день россии', '12 июня', '4 ноября']
            for m in majors:
                if m in low: return 'фед_ключевой'
            if 'пасх' in low: return 'религ_пасха'
            return 'прочий'
        cal_df['hol_type4'] = cal_df['hol_name'].map(map_holiday_type).astype('category')
        cal_df['i'] = np.arange(len(cal_df))
        cal_df['prev_h_i'] = np.where(cal_df['is_holiday']==1, cal_df['i'], np.nan); cal_df['prev_h_i'] = cal_df['prev_h_i'].ffill()
        cal_df['d_prev'] = (cal_df['i'] - cal_df['prev_h_i']).fillna(9999)
        cal_df['next_h_i'] = np.where(cal_df['is_holiday']==1, cal_df['i'], np.nan); cal_df['next_h_i'] = cal_df['next_h_i'].bfill()
        cal_df['d_next'] = (cal_df['next_h_i'] - cal_df['i']).fillna(9999)
        def nearest_delta(row):
            if row['is_holiday'] == 1: return 0
            a, b = row['d_prev'], row['d_next']
            return int(a) if abs(a) <= abs(b) else int(b)
        def prox_cat(v):
            if v == 0:  return 'H0'
            elif v == -1: return 'H-1'
            elif v == -2: return 'H-2'
            elif v == 1:  return 'H+1'
            elif v == 2:  return 'H+2'
            else:        return 'нет_рядом'
        cal_df['holiday_prox'] = cal_df.apply(nearest_delta, axis=1).map(prox_cat).astype('category')
        is_weekend = (cal_df.index.dayofweek >= 5).astype(int)
        cal_df['is_nonwork'] = np.maximum(is_weekend, cal_df['is_holiday'])
        end3 = cal_df['is_nonwork'].rolling(3, min_periods=3).sum().eq(3)
        long_any = end3 | end3.shift(1, fill_value=False) | end3.shift(2, fill_value=False)
        cal_df['long_weekend'] = long_any.map({True:'да', False:'нет'}).astype('category')
        cal_df['blue_monday'] = np.where(
            (cal_df.index.dayofweek == 0) & (cal_df['long_weekend'].shift(1).eq('да')),
            'да', 'нет'
        )
        cal_df['blue_monday'] = cal_df['blue_monday'].astype('category')
        tmp = pd.DataFrame({'msk_key': d}).merge(
            cal_df[['is_holiday','hol_type4','holiday_prox','long_weekend','blue_monday']],
            left_on='msk_key', right_index=True, how='left'
        )
        out['cal_holiday_type4']  = tmp['hol_type4'].fillna('нет').astype('category')
        out['cal_holiday_window5'] = tmp['holiday_prox'].fillna('нет_рядом').astype('category')
        out['cal_long_weekend']    = tmp['long_weekend'].fillna('нет').astype('category')
        weekend = d.dt.dayofweek >= 5
        def day_type(is_h, is_w):
            if is_h == 1: return 'праздник'
            if is_w: return 'выходной'
            return 'рабочий'
        out['cal_day_type3'] = [day_type(h, w) for h, w in zip(tmp['is_holiday'].fillna(0).astype(int), weekend)]
        out['cal_day_type3'] = out['cal_day_type3'].astype('category')
        out['cal_is_blue_monday'] = tmp['blue_monday'].fillna('нет').astype('category')
    else:
        out[['cal_holiday_type4','cal_holiday_window5','cal_long_weekend','cal_day_type3','cal_is_blue_monday']] =             pd.DataFrame([['нет', 'нет_рядом', 'нет', 'рабочий', 'нет']], index=out.index).astype('category')
    def school_break_ext(dt):
        if pd.isna(dt): return 'нет'
        m, day = int(dt.month), int(dt.day)
        if m == 1 and day <= 9: return 'зима'
        if m == 3 and day >= 25: return 'весна'
        if (m == 10 and day >= 26) or (m == 11 and day <= 4): return 'осень'
        if m in (6,7,8): return 'лето'
        return 'нет'
    out['cal_school_break_ext'] = d.map(school_break_ext).astype('category')
    day = d.dt.day
    dist_to_10 = np.minimum(abs(day - 10), abs(day - 10 - d.dt.days_in_month))
    dist_to_25 = np.minimum(abs(day - 25), abs(day - 25 - d.dt.days_in_month))
    min_dist = np.minimum(dist_to_10, dist_to_25)
    def payday_prox_cat(dv):
        if pd.isna(dv): return 'далеко'
        if dv == 0: return 'день выплаты'
        if dv <= 2: return 'рядом (±2 дня)'
        if dv <= 7: return 'неделя до/после'
        return 'далеко'
    out['cal_payday_proximity'] = min_dist.map(payday_prox_cat).astype('category')
    return out


# ============================================================
# ==== БЛОК 3: ПОГОДА, ДЛИНА ДНЯ, МАГНИТНЫЕ БУРИ ====
# ============================================================
WEATHER_COMPACT_CAT = [
    'wth_daylight_duration_cat', 'wth_temp_feels_like_cat', 'wth_precipitation_cat',
    'wth_wind_speed_cat', 'wth_complex_weather_cat', 'wth_bad_weather_last5_cat',
    'wth_extreme_temp_last5_cat', 'wth_national_bad_weather_scale_cat',
    'wth_national_temp_anomaly_scale_cat', 'wth_seasonal_temp_anomaly_cat',
    'wth_temp_change_cat', 'wth_sunshine_ratio_cat', 'wth_precipitation_change_cat',
    'wth_pressure_change_cat', 'wth_wind_speed_change_cat',
    'mag_storm_level_cat', 'mag_storm_change_cat'
]

def add_astro_tension_index(enriched: pd.DataFrame, windows=ASTRO_TENSION_WINDOW) -> pd.DataFrame:
    df = enriched.copy()
    event_weights = {'is_retrograde_any_cat': 1, 'is_ingress_any_cat': 2, 'is_station_any_cat': 3, 'лунное_затмение_cat': 5, 'солнечное_затмение_cat': 5}
    astro_flags = [c for c in event_weights.keys() if c in df.columns]
    if not astro_flags:
        return df
    unique_dates = df[['tz', 'day_local'] + astro_flags].dropna(subset=['day_local']).drop_duplicates()
    unique_dates.sort_values(['tz', 'day_local'], inplace=True)
    for w in windows:
        tension_score = pd.Series(0.0, index=unique_dates.index)
        for col, weight in event_weights.items():
            if col in unique_dates.columns:
                flag_series = unique_dates[col].map({'да': 1, 'нет': 0}).fillna(0)
                sum_in_window = flag_series.groupby(unique_dates['tz']).transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum()).fillna(0)
                tension_score += sum_in_window * weight
        def bin_tension(v):
            v = int(v)
            if v == 0: return 'Спокойно'
            if v <= 4: return 'Легкое напряжение'
            if v <= 9: return 'Среднее напряжение'
            if v <= 14: return 'Высокое напряжение'
            return 'Очень высокое'
        unique_dates[f'astro_tension_last{w}_cat'] = tension_score.map(bin_tension).astype('category')
        df = df.merge(unique_dates[['tz', 'day_local', f'astro_tension_last{w}_cat']], on=['tz', 'day_local'], how='left')
    return df

def compute_daylight_features(lat, lon, day_local, tz):
    try:
        local_midnight_utc = pd.Timestamp(day_local).tz_localize(tz).tz_convert('UTC')
        obs = ephem.Observer(); obs.lat, obs.lon, obs.elevation = str(lat), str(lon), 0
        obs.date = ephem.Date(local_midnight_utc.to_pydatetime())
        sun = ephem.Sun()
        try:
            sr_utc = obs.next_rising(sun, use_center=True)
            ss_utc = obs.next_setting(sun, use_center=True)
            day_length_hours = float((ss_utc - sr_utc) * 24.0)
            return {'day_length_hours': day_length_hours}
        except ephem.AlwaysUpError:
            return {'day_length_hours': 24.0}
        except ephem.NeverUpError:
            return {'day_length_hours': 0.0}
    except Exception:
        return {'day_length_hours': np.nan}

def prefetch_region_weather_multi_station(region, start_date, end_date):
    if not METEOSTAT_AVAILABLE:
        return pd.DataFrame()
    safe_region = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in region)
    cache_file = CACHE_DIR / f"weather_avg_{safe_region}.parquet"
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            if 'day_local' in cached.columns:
                cached['day_local'] = pd.to_datetime(cached['day_local']).dt.date
            if cached['day_local'].min() <= start_date.date() and cached['day_local'].max() >= end_date.date():
                return cached
        except Exception as e:
            print(f"[WARN] Не удалось прочитать кэш погоды для региона {region}: {e}")
    coords_list = REGION_COORDS_MASTER.get(region)
    if not coords_list:
        return pd.DataFrame()
    tz = region_to_tz(region)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    weather_dfs = []
    for lat, lon in coords_list:
        try:
            point = Point(lat, lon)
            dfw = Daily(point, start_dt - pd.Timedelta(days=1), end_dt).fetch()
            if dfw is not None and not dfw.empty:
                weather_dfs.append(dfw)
        except Exception:
            pass
    if not weather_dfs:
        return pd.DataFrame()
    avg_weather = pd.concat(weather_dfs).groupby(level=0).mean()
    avg_weather = avg_weather.reset_index().rename(columns={'time': 'day_local'})
    avg_weather['day_local'] = pd.to_datetime(avg_weather['day_local'], errors='coerce')
    for col in ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'tsun', 'snow', 'pres']:
        if col not in avg_weather.columns:
            avg_weather[col] = np.nan
        avg_weather[col] = avg_weather[col].astype(float)
    mask_na = avg_weather['tavg'].isna() & avg_weather['tmin'].notna() & avg_weather['tmax'].notna()
    avg_weather.loc[mask_na, 'tavg'] = (avg_weather.loc[mask_na, 'tmin'] + avg_weather.loc[mask_na, 'tmax']) / 2.0
    avg_weather['region_std'] = region
    central_lat, central_lon = coords_list[0]
    dfw_daylight = avg_weather[['day_local']].copy()
    dfw_daylight['daylight_hours'] = dfw_daylight['day_local'].apply(
        lambda dt: compute_daylight_features(central_lat, central_lon, dt, tz)['day_length_hours']
    )
    avg_weather = avg_weather.merge(dfw_daylight, on='day_local', how='left')
    avg_weather.rename(columns={'prcp':'precipitation_mm', 'wspd':'wspd_kmh', 'snow':'snow_mm'}, inplace=True)
    try:
        avg_weather.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"[WARN] Не удалось сохранить кэш погоды для региона {region}: {e}")
    return avg_weather

def add_weather_compact_cats(df: pd.DataFrame, date_col='day_local', region_col='region_std') -> pd.DataFrame:
    out = df.copy()
    out.sort_values([region_col, date_col], inplace=True)
    if 'daylight_hours' in out.columns and 'tsun' in out.columns:
        sun_ratio = (out['tsun'].fillna(0) / 60.0) / out['daylight_hours'].replace(0, np.nan)
        out['wth_sunshine_ratio_cat'] = pd.cut(
            sun_ratio, bins=[-0.1, 0.1, 0.4, 0.7, 1.1],
            labels=['пасмурно', 'облачно', 'переменная облачность', 'ясно'], right=True
        ).astype('category')
    if 'daylight_hours' in out.columns:
        out['wth_daylight_duration_cat'] = pd.cut(
            out['daylight_hours'], bins=[-0.1, 8, 12, 16, 24.1],
            labels=['очень короткий', 'короткий', 'длинный', 'очень длинный'], right=True
        ).astype('category')
    if {'tavg','wspd_kmh'}.issubset(out.columns):
        def feels_like(t, w):
            if pd.isna(t) or pd.isna(w): return np.nan
            if (t > 10) or (w < 5): return t
            return 13.12 + 0.6215*t - 11.37*(w**0.16) + 0.3965*t*(w**0.16)
        out['temp_feels_like'] = out.apply(lambda r: feels_like(r['tavg'], r['wspd_kmh']), axis=1)
        out['wth_temp_feels_like_cat'] = pd.cut(
            out['temp_feels_like'],
            bins=[-100, -20, -5, 10, 20, 100],
            labels=['экстр. холод', 'холод', 'прохладно', 'комфорт', 'жара'], right=False
        ).astype('category')
    if {'tavg', region_col, date_col}.issubset(out.columns):
        out['__year'] = out[date_col].dt.year
        out['__month'] = out[date_col].dt.month
        monthly_means = out.groupby([region_col, '__year', '__month'])['tavg'].mean().reset_index()
        monthly_means.sort_values(by=[region_col, '__month', '__year'], inplace=True)
        historical_expanding_mean = monthly_means.groupby([region_col, '__month'])['tavg'].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        monthly_means['seasonal_mean_t'] = historical_expanding_mean
        monthly_means['seasonal_mean_t'] = monthly_means.groupby([region_col, '__month'])['seasonal_mean_t'].bfill()
        out = pd.merge(out, monthly_means[[region_col, '__year', '__month', 'seasonal_mean_t']], on=[region_col, '__year', '__month'], how='left')
        out['__seasonal_temp_anomaly_deg'] = out['tavg'] - out['seasonal_mean_t']
        out['wth_seasonal_temp_anomaly_cat'] = pd.cut(
            out['__seasonal_temp_anomaly_deg'],
            bins=[-100, -8, -3, 3, 8, 100],
            labels=['сильно холоднее', 'холоднее нормы', 'норма', 'теплее нормы', 'сильно теплее'],
            right=False
        ).astype('category')
        out.drop(columns=['__year', '__month', 'seasonal_mean_t', '__seasonal_temp_anomaly_deg'], inplace=True, errors='ignore')
    if 'tavg' in out.columns:
        prev_tavg = out.groupby(region_col)['tavg'].shift(1)
        temp_change = out['tavg'] - prev_tavg
        out['wth_temp_change_cat'] = pd.cut(
            temp_change,
            bins=[-100, -5, -2, 2, 5, 100],
            labels=['сильное похолодание', 'похолодание', 'без изменений', 'потепление', 'сильное потепление'],
            right=False
        ).astype('category')
    if 'precipitation_mm' in out.columns:
        out['wth_precipitation_cat'] = pd.cut(
            out['precipitation_mm'].fillna(0),
            bins=[-1, 0, 1, 5, 10000],
            labels=['без осадков', 'легкие', 'умеренные', 'сильные'],
            right=True
        ).astype('category')
    if 'precipitation_mm' in out.columns:
        precip_current = out['precipitation_mm'].fillna(0)
        precip_prev = out.groupby(region_col)['precipitation_mm'].shift(1).fillna(0)
        conditions = [
            precip_prev.isna(),
            (precip_prev <= 0) & (precip_current > 0),
            (precip_prev > 0) & (precip_current <= 0),
            (precip_current > precip_prev),
            (precip_current < precip_prev),
        ]
        choices = ['Без изменений', 'Начались осадки', 'Осадки прекратились', 'Осадки усилились', 'Осадки ослабли']
        out['wth_precipitation_change_cat'] = pd.Categorical(np.select(conditions, choices, default='Без изменений'))
    if 'wspd_kmh' in out.columns:
        out['wth_wind_speed_cat'] = pd.cut(
            out['wspd_kmh'].fillna(0),
            bins=[-1, 5, 15, 25, 10000],
            labels=['штиль', 'слабый', 'умеренный', 'сильный'],
            right=True
        ).astype('category')
    if 'pres' in out.columns:
        pressure_change = out.groupby(region_col)['pres'].diff()
        out['wth_pressure_change_cat'] = pd.cut(
            pressure_change,
            bins=[-1000, -5, -1, 1, 5, 1000],
            labels=['сильное падение', 'падение', 'стабильно', 'рост', 'сильный рост'],
            right=False
        ).astype('category')
    if 'wspd_kmh' in out.columns:
        wind_change = out.groupby(region_col)['wspd_kmh'].diff()
        out['wth_wind_speed_change_cat'] = pd.cut(
            wind_change,
            bins=[-100, -10, -3, 3, 10, 100],
            labels=['сильно стих', 'стих', 'без изменений', 'усилился', 'сильно усилился'],
            right=False
        ).astype('category')
    def complex_weather_row(r):
        p = r.get('precipitation_mm', 0) if pd.notna(r.get('precipitation_mm')) else 0
        w = r.get('wspd_kmh', 0) if pd.notna(r.get('wspd_kmh')) else 0
        tfl = r.get('temp_feels_like')
        if (p > 5) or (w > 25) or (pd.notna(tfl) and tfl < -20): return 'экстремальная'
        if p > 0: return 'осадки'
        return 'спокойная'
    out['wth_complex_weather_cat'] = out.apply(complex_weather_row, axis=1).astype('category')
    out['__is_bad_weather'] = (out['wth_complex_weather_cat'] != 'спокойная').astype(int)
    out['__is_extreme_temp'] = out.get('wth_seasonal_temp_anomaly_cat', pd.Series(index=out.index)).isin(['сильно холоднее', 'сильно теплее']).astype(int)
    base_cols = [region_col, date_col, '__is_bad_weather', '__is_extreme_temp']
    if 'msk_day' in out.columns:
        base_cols = [region_col, date_col, 'msk_day', '__is_bad_weather', '__is_extreme_temp']
    base = out[base_cols].drop_duplicates().sort_values([region_col, date_col])
    bad_roll5 = base.groupby(region_col)['__is_bad_weather'].shift(1).rolling(5, min_periods=1).sum().fillna(0)
    base['wth_bad_weather_last5_cat'] = pd.cut(
        bad_roll5, bins=[-1, 0, 1, 3, 6],
        labels=['0 дней', '1 день', '2-3 дня', '4-5 дней']
    ).astype('category')
    ext_roll5 = base.groupby(region_col)['__is_extreme_temp'].shift(1).rolling(5, min_periods=1).sum().fillna(0)
    base['wth_extreme_temp_last5_cat'] = pd.cut(
        ext_roll5, bins=[-1, 0, 1, 3, 6],
        labels=['0 дней', '1 день', '2-3 дня', '4+ дней']
    ).astype('category')
    agg_date_col = 'msk_day' if 'msk_day' in out.columns else date_col
    agg = base.groupby(agg_date_col)[['__is_bad_weather', '__is_extreme_temp']].mean().reset_index()
    agg['wth_national_bad_weather_scale_cat'] = pd.cut(
        agg['__is_bad_weather'], bins=[-0.1, 0.1, 0.3, 0.6, 1.1],
        labels=['локально', 'местами', 'многие регионы', 'по всей стране']
    ).astype('category')
    agg['wth_national_temp_anomaly_scale_cat'] = pd.cut(
        agg['__is_extreme_temp'], bins=[-0.1, 0.1, 0.3, 0.6, 1.1],
        labels=['локально', 'местами', 'многие регионы', 'по всей стране']
    ).astype('category')
    out = out.merge(
        base[[region_col, date_col, 'wth_bad_weather_last5_cat', 'wth_extreme_temp_last5_cat']],
        on=[region_col, date_col], how='left'
    ).merge(
        agg[[agg_date_col, 'wth_national_bad_weather_scale_cat', 'wth_national_temp_anomaly_scale_cat']],
        left_on=agg_date_col, right_on=agg_date_col, how='left'
    )
    for c in WEATHER_COMPACT_CAT:
        if c in out.columns and out[c].dtype.name != 'category':
            out[c] = out[c].astype('category')
    return out

def load_magnetic_indices(kp_path: Path, ap_path: Path) -> pd.DataFrame:
    """Загружает Kp и ap индексы, объединяет по дате, агрегирует до уровня дня."""
    def load_json_index(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['datetime']),
            'value': data['Kp'] if 'Kp' in data else data['ap']
        })
        df['date'] = df['datetime'].dt.date
        return df

    kp_df = load_json_index(kp_path)
    ap_df = load_json_index(ap_path)

    kp_daily = kp_df.groupby('date')['value'].mean().rename('Kp_daily')
    ap_daily = ap_df.groupby('date')['value'].mean().rename('ap_daily')

    mag_df = pd.concat([kp_daily, ap_daily], axis=1).reset_index()
    mag_df['date'] = pd.to_datetime(mag_df['date'])
    return mag_df

def add_magnetic_storm_features(df: pd.DataFrame, kp_path: Path, ap_path: Path, date_col_msk: str = 'msk_day') -> pd.DataFrame:
    """Добавляет компактные категориальные признаки на основе магнитных индексов."""
    try:
        mag_df = load_magnetic_indices(kp_path, ap_path)
    except Exception as e:
        print(f"[WARN] Не удалось загрузить данные о магнитных бурях с путей {kp_path}, {ap_path}: {e}")
        df['mag_storm_level_cat'] = 'данные недоступны'
        df['mag_storm_change_cat'] = 'данные недоступны'
        return df

    df = df.merge(mag_df[['date', 'Kp_daily']], left_on=date_col_msk, right_on='date', how='left')
    df['Kp_daily'] = df['Kp_daily'].fillna(df['Kp_daily'].median())

    def classify_storm_level(kp):
        if kp <= 4: return 'спокойно'
        elif kp <= 5: return 'слабая буря'
        elif kp <= 6: return 'умеренная буря'
        elif kp <= 8: return 'сильная буря'
        else: return 'экстремальная буря'
    df['mag_storm_level_cat'] = df['Kp_daily'].apply(classify_storm_level).astype('category')

    df = df.sort_values(date_col_msk).reset_index(drop=True)
    df['Kp_prev'] = df['Kp_daily'].shift(1)
    df['Kp_change'] = df['Kp_daily'] - df['Kp_prev']

    def classify_change(change):
        if pd.isna(change): return 'нет данных'
        if change > 1.5: return 'резкий рост'
        elif change > 0.5: return 'рост'
        elif change < -1.5: return 'резкое падение'
        elif change < -0.5: return 'падение'
        else: return 'стабильно'
    df['mag_storm_change_cat'] = df['Kp_change'].apply(classify_change).astype('category')

    df.drop(columns=['Kp_daily', 'Kp_prev', 'Kp_change', 'date'], errors='ignore', inplace=True)
    return df


# ============================================================
# ==== БЛОК 4: НОВОСТИ ====
# ============================================================
NEWS_COMPACT_CAT = [
    'news_day_group7', 'news_burst_last5_cat', 'news_topics_last5_cat',
    'news_tone_day_cat', 'news_tone_last5_cat', 'news_macro_risk_last5_cat',
    'news_security_risk_last5_cat', 'news_it_pay_last5_cat', 'news_energy_last5_cat',
    'news_recency_major_cat', 'news_span_last14_cat', 'news_holiday_overlay_cat',
    'news_tone_change_cat'
]

NEWS_CONFIG = {
    'include_monetary_in_major': True,
    'compress_security_updates': True,
    'audit_print': True,
    'audit_sample_n': 10
}

NEWS_GROUP_PATTERNS = {
    'Безопасность/ЧС': [
        'военные действ','боевые','бои','удар','контрнаступ','взят','захват','вывод войск','дрг',
        'безопасност','теракт','чс','чп','взрыв','обстрел','ракет','артилл','штурм','прорыв',
        'освобожд','паводк','происшеств','дрон','бпла','диверс'
    ],
    'Санкции/финансы': [
        'санкци','эмбарго','swift','свифт','нкц','клиринг','рейтинг','рейтинга','мус','валют','курс',
        'мосбирж','мособирж','спб-биржа','бирж','банк'
    ],
    'Денежная политика': ['ключевая ставка','денежная политика'],
    'IT/платежи/сбои': [
        'платеж','финтех','сбп','mir pay','кошелек','кошельк','push','пуш','ddos','сбой',
        'онлайн','it','интернет','соцсеть','стрим','dpi','dnssec'
    ],
    'Энергетика/рынки': [
        'энергетик','нефть','газ','сп-1','северн','газпром','опек','алмаз','топлив','экспорт топлив'
    ],
    'Экономика/налоги/бюджет': [
        'экономик','бюджет','налог','макро','форум','пмэф','вэф','ипотек','торговля','импорт','экспорт',
        'логистик','компани','промышленност','металл','апк','сельхоз','опк'
    ],
    'Политика/право/выборы': [
        'политика','право','закон','выбор','междунар','нато','брикс','саммит','шос','совет европы',
        'кадры','территор','оборона','призыв','мобилиз','инаугурац'
    ],
    'Праздники/общество': ['праздник','общество','социальн','мрот','жкх','культур','парад','траур']
}

NEWS_GROUP_WEIGHTS = {
    'Безопасность/ЧС': -2,
    'Санкции/финансы': -2,
    'Денежная политика': -1,
    'IT/платежи/сбои': -1,
    'Энергетика/рынки': -1,
    'Экономика/налоги/бюджет': 0,
    'Политика/право/выборы': 0,
    'Праздники/общество': +1
}

SEC_HINTS = re.compile(r'(захват|обстрел|бои|боест|ракет|артилл|штурм|прорыв|освобожд|дрг|теракт|паводк|дрон|бпла|взрыв)', re.I)

def _normalize_cat_text(s: str) -> str:
    s = str(s or '').lower()
    s = s.replace('междунар.', 'международн')
    s = s.replace('чп', 'чс')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _map_news_group(cat_raw: str, event_text: str) -> str:
    s = f"{str(cat_raw)} {str(event_text)}".lower()
    s = s.replace('междунар.', 'международн').replace('чп', 'чс')
    for grp, pats in NEWS_GROUP_PATTERNS.items():
        if any(pat in s for pat in pats):
            return grp
    return 'Экономика/налоги/бюджет'

def _load_events_from_tsv(events_path: Path) -> pd.DataFrame:
    """Загружает и парсит файл с событиями из TSV."""
    if not events_path.exists():
        print(f"[WARN] Файл событий не найден по пути: {events_path}")
        return pd.DataFrame(columns=['event_date','event','cat_raw'])

    with open(events_path, 'r', encoding='utf-8') as f:
        tsv_text = f.read()

    if not tsv_text.strip():
        return pd.DataFrame(columns=['event_date','event','cat_raw'])

    try:
        # Пытаемся прочитать с заголовком
        ev = pd.read_csv(StringIO(tsv_text.strip()), sep='\t', header=0, dtype=str)
        if 'Дата' not in ev.columns or 'Событие' not in ev.columns:
            raise ValueError("Не найдены обязательные колонки")
    except (Exception, ValueError):
        try:
            # Если не вышло, читаем без заголовка, присваивая имена
            ev = pd.read_csv(StringIO(tsv_text.strip()), sep='\t', header=None, names=['Дата','Событие','Категория'], dtype=str)
        except Exception:
            print(f"[ERROR] Не удалось прочитать файл событий: {events_path}")
            return pd.DataFrame(columns=['event_date','event','cat_raw'])

    if 'Категория' not in ev.columns:
        ev['Категория'] = 'Неизвестно'

    ev.rename(columns={'Событие': 'event', 'Категория': 'cat_raw'}, inplace=True)
    ev['event_date'] = pd.to_datetime(ev['Дата'], format='%d.%m.%Y', errors='coerce')
    ev = ev[['event_date', 'event', 'cat_raw']].dropna(subset=['event_date']).reset_index(drop=True)

    if ev.empty:
        return ev

    ev['cat_norm'] = ev['cat_raw'].apply(_normalize_cat_text)
    ev['group'] = [_map_news_group(c, e) for c, e in zip(ev['cat_norm'], ev['event'])]
    return ev

def _compress_security_updates(ev: pd.DataFrame) -> pd.DataFrame:
    """Сжимает повторы «боевых» апдейтов в один на день (нормализация текста)."""
    if ev.empty or 'group' not in ev.columns:
        return ev
    df = ev.copy()
    mask_sec = df['group'] == 'Безопасность/ЧС'
    def norm_event(txt: str) -> str:
        s = str(txt or '').lower()
        s = re.sub(r'\(.*?\)', ' ', s) # убрать скобки
        s = re.sub(r'\b(захват|взятие?|освобожден\w*|освобожд\w*|обстрел\w*|бои|штурм\w*|прорыв\w*|контр\w*|интенсив\w*|продолжение|попытка|провал|новая|после|стаб\w*)\b', ' ', s)
        s = re.sub(r'[^а-яa-z0-9]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    df['event_norm'] = np.where(mask_sec, df['event'].apply(norm_event), df['event'].str.lower())
    df = pd.concat([
        df[~mask_sec],
        df[mask_sec].drop_duplicates(subset=['event_date','event_norm'])
    ], ignore_index=True).drop(columns=['event_norm'])
    return df

def _audit_events_for_debug(ev: pd.DataFrame, sample_n: int = 10):
    """Проводит аудит качества разметки новостей."""
    try:
        rep = {}
        dupl = (ev.assign(_k=lambda x: x['event_date'].astype(str) + ' | ' + x['event'].str.strip().str.lower())
                  .groupby('_k').size().reset_index(name='n').query('n>1'))
        rep['duplicates_cnt'] = int(dupl['n'].sum() - len(dupl)) if not dupl.empty else 0
        hit_mask = ev.apply(lambda r: any(pat in (str(r['cat_norm']) + ' ' + str(r['event'])).lower() for pat in sum(NEWS_GROUP_PATTERNS.values(), [])), axis=1)
        rep['default_group_share'] = float(1 - hit_mask.mean()) if len(ev) else 0.0
        mask_sec_hint = ev['event'].str.contains(SEC_HINTS, na=False)
        sec_mis = ev[mask_sec_hint & (ev['group'] != 'Безопасность/ЧС')].head(sample_n)
        if rep['duplicates_cnt'] or rep['default_group_share'] > 0.05 or not sec_mis.empty:
            print("\n[audit:news] Дубликатов (сверх 1):", rep['duplicates_cnt'])
            print("[audit:news] Доля 'дефолтного' мэппинга (fallback):", round(rep['default_group_share'], 3))
            if not sec_mis.empty:
                print("[audit:news] Военные события вне 'Безопасность/ЧС' (пример):")
                for r in sec_mis[['event_date','event','cat_raw','group']].to_dict('records'):
                    print(f"  {r['event_date'].date()} | {r['event']} | cat={r['cat_raw']} | mapped={r['group']}")
    except Exception:
        pass

def build_news_compact_features(
    df: pd.DataFrame,
    events_path: Path,
    date_col_msk: str = 'msk_day',
    **kwargs
) -> pd.DataFrame:
    """Строит компактные news-фичи."""
    out = df.copy()

    ev_raw = _load_events_from_tsv(events_path)

    if kwargs.get('audit_print', NEWS_CONFIG['audit_print']):
        _audit_events_for_debug(ev_raw.copy())

    if kwargs.get('compress_security_updates', NEWS_CONFIG['compress_security_updates']):
        ev = _compress_security_updates(ev_raw)
    else:
        ev = ev_raw

    if out[date_col_msk].notna().any():
        full_range = pd.date_range(out[date_col_msk].min(), out[date_col_msk].max(), freq='D')
    else:
        full_range = pd.DatetimeIndex([])

    last_w: int = 5
    long_w: int = 14
    include_monetary_in_major = kwargs.get('include_monetary_in_major', NEWS_CONFIG['include_monetary_in_major'])

    if ev.empty or len(full_range) == 0:
        mat = pd.DataFrame(index=full_range)
    else:
        mat = (ev.groupby(['event_date', 'group']).size().unstack(fill_value=0).reindex(full_range, fill_value=0))
    mat.index.name = date_col_msk

    if mat.shape[1] == 0:
        day_total = pd.Series(0, index=mat.index, dtype=int)
        day_group = pd.Series('нет', index=mat.index, dtype='object')
        tone_day = pd.Series(0.0, index=mat.index, dtype=float)
        pres = pd.DataFrame(index=mat.index)
    else:
        day_total = mat.sum(axis=1)
        day_group = mat.idxmax(axis=1).where(day_total > 0, 'нет')
        w = pd.Series(NEWS_GROUP_WEIGHTS).reindex(mat.columns, fill_value=0)
        tone_day = mat.mul(w, axis=1).sum(axis=1).astype(float)
        pres = (mat > 0).astype(int)

    burst_last5 = day_total.shift(1).rolling(last_w, min_periods=1).sum().fillna(0)
    topics_last5 = (pres.shift(1).rolling(last_w, min_periods=1).max().fillna(0)).sum(axis=1) if pres.shape[1] > 0 else pd.Series(0, index=mat.index, dtype=int)
    tone_last5 = tone_day.shift(1).rolling(last_w, min_periods=1).mean().fillna(0)

    def sum_groups_last5(groups):
        if mat.shape[1] == 0: return pd.Series(0, index=mat.index, dtype=float)
        cols = [c for c in mat.columns if c in groups]
        if not cols: return pd.Series(0, index=mat.index, dtype=float)
        return mat[cols].sum(axis=1).shift(1).rolling(last_w, min_periods=1).sum().fillna(0)

    macro_groups = {'Санкции/финансы', 'Экономика/налоги/бюджет', 'Политика/право/выборы', 'Денежная политика'}
    security_groups = {'Безопасность/ЧС'}
    itpay_groups = {'IT/платежи/сбои'}
    energy_groups = {'Энергетика/рынки'}

    macro_last5 = sum_groups_last5(macro_groups)
    security_last5 = sum_groups_last5(security_groups)
    itpay_last5 = sum_groups_last5(itpay_groups)
    energy_last5 = sum_groups_last5(energy_groups)

    if mat.shape[1] == 0:
        recency = pd.Series(9999, index=mat.index, dtype=int)
    else:
        major_groups = {'Безопасность/ЧС','Санкции/финансы'}
        if include_monetary_in_major:
            major_groups.add('Денежная политика')
        cols = [c for c in mat.columns if c in major_groups]
        major_flag = (mat[cols].sum(axis=1) > 0).astype(int) if cols else pd.Series(0, index=mat.index)
        idx = np.arange(len(major_flag))
        last_idx = pd.Series(np.where(major_flag == 1, idx, np.nan), index=mat.index).ffill()
        recency = (idx - last_idx).fillna(9999).astype(int)

    span14 = (day_total > 0).astype(int).shift(1).rolling(long_w, min_periods=1).sum().fillna(0)

    news_daily = pd.DataFrame(index=mat.index)
    tone_change = tone_day.diff()
    def cut_tone_change(v):
        if pd.isna(v): return 'без изменений'
        if v <= -2: return 'резко негативнее'
        if v < 0: return 'негативнее'
        if v >= 2: return 'резко позитивнее'
        if v > 0: return 'позитивнее'
        return 'без изменений'

    def cut_burst(x):  return '0' if x <= 0 else ('1-2' if x <= 2 else ('3-5' if x <= 5 else '6+'))
    def cut_topics(x): return '0' if x <= 0 else ('1-2' if x <= 2 else ('3-4' if x <= 4 else '5+'))
    def cut_tone(v):   return 'сильно негативный' if v <= -2 else ('негативный' if v < -0.5 else ('нейтральный' if v <= 0.5 else ('позитивный' if v < 2 else 'сильно позитивный')))
    def cut_macro(x):  return '0' if x <= 0 else ('1' if x == 1 else ('2-3' if x <= 3 else '4+'))
    def cut_small(x):  return '0' if x <= 0 else ('1' if x == 1 else '2+')
    def cut_recency(dv):return '0-1' if dv <= 1 else ('2-3' if dv <= 3 else ('4-7' if dv <= 7 else '>7'))
    def cut_span14(x): return '0-2' if x <= 2 else ('3-5' if x <= 5 else ('6-9' if x <= 9 else '10-14'))

    news_daily['news_tone_change_cat']       = tone_change.map(cut_tone_change).astype('category')
    news_daily['news_day_group7']            = day_group.astype('category')
    news_daily['news_burst_last5_cat']       = burst_last5.map(cut_burst).astype('category')
    news_daily['news_topics_last5_cat']      = topics_last5.map(cut_topics).astype('category')
    news_daily['news_tone_day_cat']          = tone_day.map(cut_tone).astype('category')
    news_daily['news_tone_last5_cat']        = tone_last5.map(cut_tone).astype('category')
    news_daily['news_macro_risk_last5_cat']  = macro_last5.map(cut_macro).astype('category')
    news_daily['news_security_risk_last5_cat'] = security_last5.map(cut_macro).astype('category')
    news_daily['news_it_pay_last5_cat']      = itpay_last5.map(cut_small).astype('category')
    news_daily['news_energy_last5_cat']      = energy_last5.map(cut_small).astype('category')
    news_daily['news_recency_major_cat']     = recency.map(cut_recency).astype('category')
    news_daily['news_span_last14_cat']       = span14.map(cut_span14).astype('category')
    news_daily.reset_index(inplace=True)

    if 'cal_holiday_type4' in out.columns and len(full_range) > 0:
        day_holiday = (out.groupby(date_col_msk)['cal_holiday_type4']
                         .apply(lambda s: 1 if (s.astype(str) != 'нет').any() else 0)
                         .reindex(full_range, fill_value=0).reset_index()
                         .rename(columns={date_col_msk: 'msk_day', 'cal_holiday_type4': 'is_hol'}))
    else:
        day_holiday = pd.DataFrame({'msk_day': full_range, 'is_hol': 0})
    day_has_event = (day_total > 0).astype(int).reindex(full_range, fill_value=0).reset_index()
    day_has_event.columns = [date_col_msk, 'has_event_today']

    news_daily = news_daily.merge(day_holiday, on='msk_day', how='left').merge(day_has_event, on='msk_day', how='left')
    def overlay(row):
        if row['is_hol'] == 0: return 'нет'
        return 'праздник+события' if row['has_event_today'] == 1 else 'праздник_без_событий'
    news_daily['news_holiday_overlay_cat'] = news_daily.apply(overlay, axis=1).astype('category')
    news_daily = news_daily.drop(columns=['is_hol','has_event_today'], errors='ignore')

    out = out.merge(news_daily, on=date_col_msk, how='left')
    for c in NEWS_COMPACT_CAT:
        if c in out.columns and out[c].dtype.name != 'category':
            out[c] = out[c].astype('category')
    return out


# ============================================================
# ==== ГРУППЫ ПРИЗНАКОВ И ГЛАВНАЯ ФУНКЦИЯ ОБОГАЩЕНИЯ ====
# ============================================================

# TODO: Проверьте и дополните эти списки актуальными названиями колонок из вашего датасета
CX_CAT = [
    # Пример: 'last_call_reason', 'has_technical_issue_last_month'
]
PROFILE_CAT = [
     # Пример: 'region', 'gender'

]
ASTRO_CAT_BINS = [
    'знак_Солнца','фаза_луны','lunar_day_cat','hard_aspects_cat','pp_hard_aspects_cat',
    'is_retrograde_any_cat','is_station_any_cat','is_ingress_any_cat','солнечное_затмение_cat',
    'лунное_затмение_cat','is_hecate_moon_cat','astro_weighted_aspects_cat',
    'astro_moon_voc_proxy_cat', 'astro_interaction_merc_mars'
]

def build_feature_lists_all(df):
    cx_cols = [c for c in CX_CAT if c in df.columns]
    profile_cols = [c for c in PROFILE_CAT if c in df.columns]
    cal_cols     = [c for c in CALENDAR_COMPACT_CAT if c in df.columns]
    astro_cols   = [c for c in ASTRO_CAT_BINS if c in df.columns]
    weather_cols = [c for c in WEATHER_COMPACT_CAT if c in df.columns]
    news_cols    = [c for c in NEWS_COMPACT_CAT if c in df.columns]
    lag_cols     = [c for c in df.columns if 'astro_tension_last' in str(c)]

    cat_cols = list(dict.fromkeys(cx_cols + profile_cols + cal_cols + astro_cols + weather_cols + news_cols + lag_cols))
    num_cols = [] # Подставляем числовые признаки
    return cat_cols, num_cols

def get_feature_group(feature_name):
    """Возвращает высокоуровневую группу для признака."""
    if feature_name in CX_CAT: return 'Опыт клиента (CX)'
    if feature_name in PROFILE_CAT: return 'Профиль клиента'
    if feature_name in CALENDAR_COMPACT_CAT: return 'Календарь'
    if feature_name in WEATHER_COMPACT_CAT: return 'Погода и геомагнетизм'
    if feature_name in ASTRO_CAT_BINS or feature_name.startswith('astro_tension'): return 'Астрология'
    if feature_name in NEWS_COMPACT_CAT: return 'Новости'
    return 'Прочее'

def enrich_data_full(df, date_col='business_dt', region_col='region'):
    """
    Главная функция для обогащения исходного датасета всеми внешними данными.
    """
    df = df.copy()
    initial_cols = df.columns.tolist()

    print("  - Нормализация регионов и определение таймзон...")
    df['region_std'] = df[region_col].map(REGION_GENT_TO_NOM).fillna(df[region_col]).astype('category')
    df['tz'] = df['region_std'].apply(region_to_tz).astype('category')

    print("  - Расчет локальной даты для каждого респондента...")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    ser = df[date_col]
    if ser.dt.tz is None:
        base_msk = ser.dt.tz_localize('Europe/Moscow', ambiguous='infer', nonexistent='shift_forward')
    else:
        base_msk = ser.dt.tz_convert('Europe/Moscow')
    df['msk_day'] = base_msk.dt.normalize().dt.tz_localize(None)
    df['day_local'] = pd.NaT
    for tz in df['tz'].cat.categories:
        mask = (df['tz'] == tz)
        if not mask.any(): continue
        loc = base_msk[mask].dt.tz_convert(tz)
        df.loc[mask, 'day_local'] = loc.dt.normalize().dt.tz_localize(None)

    print("  - Расчет астрологических факторов...")
    keys = df[['day_local','tz']].dropna().drop_duplicates()
    astro_rows = [
        {'day_local': r.day_local, 'tz': r.tz, **get_astro_cats_only(r.day_local, r.tz)}
        for r in tqdm(keys.itertuples(index=False), total=len(keys), desc="    Астро-расчеты")
    ]
    if astro_rows:
        df = df.merge(pd.DataFrame(astro_rows), on=['day_local','tz'], how='left')

    print("  - Расчет календарных признаков...")
    df = add_calendar_compact_features(df)

    print("  - Загрузка и обработка погодных данных...")
    weather_keys = df[['region_std','day_local']].dropna()
    region_ranges = weather_keys.groupby('region_std')['day_local'].agg(['min','max']).reset_index()
    all_weather = []
    for region, start, end in tqdm(region_ranges.itertuples(index=False), total=len(region_ranges), desc="    Загрузка погоды по регионам"):
        wf = prefetch_region_weather_multi_station(region, start, end)
        if wf is not None and not wf.empty:
            all_weather.append(wf)

    if all_weather:
        weather_features = pd.concat(all_weather, ignore_index=True)
        weather_features['day_local'] = pd.to_datetime(weather_features['day_local'], errors='coerce')
        keep_wx = [c for c in weather_features.columns if c in ['region_std','day_local','daylight_hours','tavg','wspd_kmh','precipitation_mm','pres','tsun']]
        weather_features = weather_features[keep_wx]
        df = df.merge(weather_features, on=['region_std','day_local'], how='left')
        df = add_weather_compact_cats(df, date_col='day_local', region_col='region_std')
    else:
        # Резервный расчет только длины дня, если meteostat недоступен
        print("    [INFO] Meteostat недоступен. Расчет только длины светового дня.")
        if not df.empty:
            df['daylight_hours'] = np.nan
            unique_region_tz = df[['region_std', 'tz']].drop_duplicates()
            for _, row in tqdm(unique_region_tz.iterrows(), total=len(unique_region_tz), desc="    Расчет длины дня"):
                region_name, tz = row['region_std'], row['tz']
                coords_list = REGION_COORDS_MASTER.get(region_name)
                if coords_list:
                    lat, lon = coords_list[0]
                    mask_region = df['region_std'] == region_name
                    unique_dates = df.loc[mask_region, 'day_local'].dropna().unique()
                    daylight_map = {
                        dt: compute_daylight_features(lat, lon, dt, tz).get('day_length_hours')
                        for dt in unique_dates
                    }
                    df.loc[mask_region, 'daylight_hours'] = df.loc[mask_region, 'day_local'].map(daylight_map)
            df['wth_daylight_duration_cat'] = pd.cut(df['daylight_hours'], bins=[-0.1, 8, 12, 16, 24.1], labels=['очень короткий', 'короткий', 'длинный', 'очень длинный']).astype('category')


    print("  - Расчет новостных факторов...")
    df = build_news_compact_features(df, events_path=EVENTS_TSV_PATH, date_col_msk='msk_day', **NEWS_CONFIG)

    print("  - Расчет факторов магнитных бурь...")
    df = add_magnetic_storm_features(df, KP_INDEX_PATH, AP_INDEX_PATH, date_col_msk='msk_day')

    print("  - Финальная обработка и очистка...")
    if 'ww' in df.columns:
        s = df['ww'].astype(str).replace(r',','.', regex=True).replace(r'[^0-9eE\.\-\+]', '', regex=True)
        df['ww_weight'] = pd.to_numeric(s, errors='coerce').fillna(1.0)
    else:
        df['ww_weight'] = 1.0

    df = add_astro_tension_index(df, windows=ASTRO_TENSION_WINDOW)

    cat_cols, _ = build_feature_lists_all(df)
    keep_cols = initial_cols + ['day_local', 'msk_day', 'ww_weight'] + cat_cols
    keep_cols = list(dict.fromkeys(keep_cols))

    final_df = df[[c for c in keep_cols if c in df.columns]].copy()
    for c in cat_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].astype('category')

    return final_df, cat_cols


# ============================================================
# ==== ОБУЧЕНИЕ МОДЕЛИ И КРОСС-ВАЛИДАЦИЯ ====
# ============================================================
def train_one_fold_classifier_bin(data, cat_cols, num_cols, weight_col, target_col_bin, train_idx, test_idx):
    params = {'loss_function': 'Logloss','depth': 5,'iterations': 500,'learning_rate': 0.05,
              'random_seed': RANDOM_SEED,'eval_metric': 'AUC','custom_metric': ['AUC','F1','Precision','Recall'],
              'l2_leaf_reg': 5.0,'od_type': 'Iter','od_wait': 120,'verbose': False,'thread_count': -1,
              'one_hot_max_size': 5,'max_ctr_complexity': 1,'rsm': 0.8}
    if USED_RAM_LIMIT is not None:
        params['used_ram_limit'] = USED_RAM_LIMIT
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    X_train = train[cat_cols + num_cols].copy()
    X_test  = test[cat_cols + num_cols].copy()
    y_train = train[target_col_bin].astype(int).copy()
    y_test  = test[target_col_bin].astype(int).copy()
    w_train = pd.to_numeric(train[weight_col], errors='coerce').fillna(1.0)
    w_test  = pd.to_numeric(test[weight_col], errors='coerce').fillna(1.0)
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str).fillna('NA')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).fillna('NA')
    cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, label=y_train, weight=w_train, cat_features=cat_features_indices)
    eval_pool  = Pool(X_test,  label=y_test,  weight=w_test,  cat_features=cat_features_indices)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        'Accuracy': accuracy_score(y_test, pred, sample_weight=w_test),
        'F1': f1_score(y_test, pred, sample_weight=w_test),
        'ROC_AUC': roc_auc_score(y_test, proba, sample_weight=w_test),
        'PR_AUC': average_precision_score(y_test, proba, sample_weight=w_test),
        'LogLoss': log_loss(y_test, np.column_stack([1-proba, proba]), sample_weight=w_test)
    }
    try:
        cm = confusion_matrix(y_test, pred)
        print("    Матрица ошибок для фолда:\n", cm)
    except Exception:
        pass
    fi = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    return {'metrics': metrics, 'model': model, 'feature_importance': fi,
            'test_idx': test.index.tolist(), 'X_test': X_test, 'y_test': y_test, 'w_test': w_test, 'proba': proba}

def timecv_evaluate_classifier_bin(enriched, target_segment_col, weight_col, n_splits=8, cat_cols=None, num_cols=None):
    enriched = enriched.copy()
    def to_bin(x):
        s = str(x).strip().lower()
        return 1 if s in ('detractor', 'критик') else 0 # Для Промоутеров меняем тут лейблы просто
    enriched['target_bin'] = enriched[target_segment_col].map(to_bin)
    enriched.dropna(subset=['day_local', 'target_bin'], inplace=True)
    enriched['target_bin'] = enriched['target_bin'].astype(int)
    enriched.sort_values('day_local', inplace=True, ignore_index=True)
    enriched['time_block'] = pd.to_datetime(enriched['day_local']).dt.to_period('W').astype(str)

    print(f"   [INFO] Диапазон дат для CV: с {enriched['day_local'].min().date()} по {enriched['day_local'].max().date()}")
    print(f"   [INFO] Количество уникальных временных блоков (недель) для группировки: {enriched['time_block'].nunique()}")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    X = enriched.drop(columns=['target_bin'])
    y_stratify = enriched['target_bin']
    groups = enriched['time_block']
    folds_indices = list(cv.split(X, y_stratify, groups))

    print(f"   [INFO] Сгенерировано {len(folds_indices)} фолдов для CV с помощью StratifiedGroupKFold.")

    if cat_cols is None or num_cols is None:
        cat_cols, num_cols = build_feature_lists_all(enriched)

    folds_results = []
    for i, (train_idx, test_idx) in enumerate(tqdm(folds_indices, desc="Обучение на фолдах (BIN)")):
        test_dates = enriched.iloc[test_idx]['day_local']
        print(f"  Фолд {i+1}/{n_splits}. Тестовый период: {test_dates.min().date()} - {test_dates.max().date()}")
        fold_result = train_one_fold_classifier_bin(enriched, cat_cols, num_cols, weight_col, 'target_bin', train_idx, test_idx)
        fold_result['test_start'], fold_result['test_end'] = test_dates.min(), test_dates.max()
        folds_results.append(fold_result)
        gc.collect()

    cv_rows = [{'fold': i, 'test_start': f['test_start'].date(), 'test_end': f['test_end'].date(), **f['metrics']}
               for i, f in enumerate(folds_results, 1)]
    cv_table = pd.DataFrame(cv_rows).sort_values('test_start').reset_index(drop=True)

    all_fi = [f['feature_importance'] for f in folds_results]
    for fi in all_fi:
        s = fi['importance'].sum()
        fi['share'] = 100 * fi['importance'] / s if s > 0 else 0
    fi_agg = pd.concat([fi[['feature','share']] for fi in all_fi], ignore_index=True).groupby('feature')['share'].mean().sort_values(ascending=False).reset_index()

    last_fold_model = folds_results[-1]['model']
    joblib.dump(last_fold_model, MODEL_SAVE_PATH)
    print(f"   [INFO] Модель последнего фолда сохранена в: {MODEL_SAVE_PATH}")

    return {'cv_table': cv_table, 'fi_agg': fi_agg, 'last_X_test': folds_results[-1]['X_test']}


# ============================================================
# ==== ГЛАВНЫЙ БЛОК ЗАПУСКА ====
# ============================================================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # --- 1. Загрузка данных ---
    print(f"1. Загрузка исходных данных из '{SOURCE_DATA_PATH}'...")
    try:
        df = pd.read_csv(SOURCE_DATA_PATH, sep=',')
        print(f"   Успешно загружено. Размер: {df.shape}")
    except FileNotFoundError:
        print(f"   КРИТИЧЕСКАЯ ОШИБКА: Файл не найден. Проверьте путь в SOURCE_DATA_PATH.")
        print(f"   Ожидаемый путь: {SOURCE_DATA_PATH.resolve()}")
        raise SystemExit(1)

    # --- 2. Обогащение данных ---
    print("\n2. Запуск процесса обогащения данных (Календарь, Астро, Погода, Новости)...")
    enriched_df, cat_cols = enrich_data_full(df, date_col='business_dt', region_col='region')
    del df; gc.collect()

    # --- 3. Отчет по пропускам и сохранение ---
    print("\n3. Отчет по долям пропусков в итоговых признаках:")
    nan_report = enriched_df[cat_cols].isna().mean().mul(100).sort_values(ascending=False).to_frame('NaN_pct')
    nan_report['group'] = nan_report.index.to_series().map(get_feature_group)
    print(nan_report[nan_report['NaN_pct'] > 0].round(2))

    try:
        print(f"\n4. Сохранение обогащенного датасета в '{ENRICHED_DATA_SAVE_PATH}'...")
        enriched_df.to_csv(ENRICHED_DATA_SAVE_PATH, index=False, encoding='utf-8', compression='gzip')
        print("   Успешно сохранено.")
    except Exception as e:
        print(f"\n[WARN] Не удалось сохранить обогащенный датасет: {e}")

    # --- 5. Обучение и оценка модели ---
    print("\n5. Запуск кросс-валидации (Stratified Group Time-based CV)...")
    print("   Цель: Бинарная классификация (Детрактор=1 vs Остальные=0)")
    cv_res = timecv_evaluate_classifier_bin(
        enriched_df,
        target_segment_col='nps_segment',
        weight_col='ww_weight',
        n_splits=N_SPLITS_CV,
        cat_cols=cat_cols,
        num_cols=[]
    )

    # --- 6. Вывод результатов ---
    print("\n" + "="*25 + " РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ " + "="*25)
    print("\nМетрики качества по фолдам:")
    print(cv_res['cv_table'][['fold','test_start','test_end','Accuracy','F1','ROC_AUC','PR_AUC','LogLoss']].round(4))

    print("\nСреднее и стандартное отклонение метрик:")
    means = cv_res['cv_table'][['Accuracy','F1','ROC_AUC','PR_AUC','LogLoss']].mean()
    stds  = cv_res['cv_table'][['Accuracy','F1','ROC_AUC','PR_AUC','LogLoss']].std()
    print(pd.DataFrame({'mean': means, 'std': stds}).round(4))

    print("\nТоп-30 факторов по средней важности (Feature Importance, %):")
    fi_agg_grouped = cv_res['fi_agg'].copy()
    fi_agg_grouped['group'] = fi_agg_grouped['feature'].apply(get_feature_group)
    print(fi_agg_grouped.head(30).round(2))

    group_importance = fi_agg_grouped.groupby('group')['share'].sum().sort_values(ascending=False)
    print("\nСуммарная важность групп признаков (%):")
    print(group_importance.round(2).to_frame(name='Total Share (%)'))

    print("\n" + "="*70)
    print("Скрипт успешно завершил работу.")

