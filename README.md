# On-device predictive-maintenance model for propulsion health monitoring of an unmanned platform

## Содержание

1. [Task Description](#task-description)
2. [Dataset Raw Info](#dataset-raw-info)
3. [Data Validation](#data-validation)
4. [Split Strategy](#split-strategy)
5. [EDA of Dataset](#eda-of-dataset)
6. [Target Analysis](#target-analysis)
7. [Grouped Statistics](#grouped-statistics)
8. [Focus Group Selection](#focus-group-selection)
9. [Feature Trajectory Analysis](#feature-trajectory-analysis)
10. [Iteration 1](#iteration-1)
11. [Iteration 2](#iteration-2)
12. [Summary](#summary)

## Task Description

В начале каждого временного ряда двигатель работает в штатном режиме, но в какой-то момент в нём возникает неисправность. В обучающей выборке тяжесть неисправности нарастает вплоть до отказа системы. В тестовой выборке временной ряд обрывается до момента отказа. Цель - предсказать количество оставшихся циклов работы до отказа для данных из тестовой выборки, то есть число циклов, в течение которых двигатель продолжит функционировать после последнего зафиксированного цикла. Также предоставляется вектор истинных значений оставшегося срока службы (RUL) для тестовых данных.

Данные представлены в виде текстового файла, содержащего 26 столбцов с числовыми значениями, разделёнными пробелами. Каждая строка соответствует состоянию системы в конкретном цикле работы, а каждый столбец - отдельной переменной.

Столбцы содержат следующие данные:

| Столбец в файле | Индекс feature в коде | Переменная |
| ---: | ---: | --- |
| 1 | 0 | Номер агрегата |
| 2 | 1 | Время, в циклах |
| 3 | 2 | Рабочий параметр 1 |
| 4 | 3 | Рабочий параметр 2 |
| 5 | 4 | Рабочий параметр 3 |
| 6 | 5 | Показание датчика 1 |
| 7 | 6 | Показание датчика 2 |
| 8-25 | 7-24 | Показания датчиков 3-20 |
| 26 | 25 | Показание датчика 21 |

Далее номера features указаны по индексам в коде: **№0-№25**.

## Dataset Raw Info

| Параметр | Значение |
| --- | --- |
| Unit of prediction | 1 operational cycle |
| Input | 26 numeric features |
| Regression target | `remaining_cycles` |
| 1 row | 1 engine at 1 operational cycle |
| Dataset Size: n | 20 631 samples |
| Dataset Size: p | 26 features |
| X size | 536 406 |
| Column types: №0-№1 | Numeric discrete |
| Column types: №2-№25 | Numeric continuous |

### RUL of Prediction

В dataset target представлен RUL в виде циклов для конкретного двигателя. Нам нужно привести текущий RUL к target каждого sample-цикла. Сделаем это по формуле:

```text
RUL = cycle_max - cycle_current
```

## Data Validation

| Проверка | Результат |
| --- | --- |
| Missing Values | 0 missing values |
| Duplicate Rows | 0 дубликатов |

### Useless Feature

Колонка №0 описывает собой ID конкретного двигателя. Эта feature помогает сегментировать данные и рассчитывать RUL, но не влияет на сам prediction. Будем опираться на остальные features №1-№25, описывающие настройки двигателей, номер цикла, а также данные с сенсоров.

## Split Strategy

Поскольку есть колонка №0 с обозначением ID двигателей, split произведётся по ней. Отдельная test data есть, поэтому разделю двигатели с ID №1-№80 на train, и №81-№100 на validation.

| Выборка | Engine ID |
| --- | --- |
| Train | 1-80 |
| Validation | 81-100 |
| Test | Отдельная test data |

## EDA of Dataset

### Features Sorted By Variance

| Feature | Variance |
| ---: | ---: |
| 1 | 4744.36083392908 |
| 0 | 854.2131190957757 |
| 13 | 487.62993118542477 |
| 18 | 363.8828513366242 |
| 8 | 81.00695975729347 |
| 7 | 37.589172369190365 |
| 21 | 2.3985506412335056 |
| 11 | 0.7833503331586856 |
| 16 | 0.5439586389999457 |
| 6 | 0.2500411526291694 |
| 15 | 0.07133222081842422 |
| 24 | 0.03266768768508047 |
| 25 | 0.011717683890605804 |
| 17 | 0.005172079728534587 |
| 12 | 0.005038693972269999 |
| 19 | 0.0014065596914734817 |
| 2 | 4.784108223698486e-06 |
| 10 | 1.9291855741282404e-06 |
| 3 | 8.588124592522145e-08 |
| 9 | 2.8398992587956425e-29 |
| 20 | 1.2037062152420224e-35 |
| 4 | 0.0 |
| 5 | 0.0 |
| 14 | 0.0 |
| 22 | 0.0 |
| 23 | 0.0 |

Feature order:

```text
[1, 0, 13, 18, 8, 7, 21, 11, 16, 6, 15, 24, 25, 17, 12, 19, 2, 10, 3, 9, 20, 4, 5, 14, 22, 23]
```

### Variance Columns

- Колонки №4, №5, №14, №22, №23 имеют variance 0 - их можно удалить.
- Колонки №2, №3, №9, №10, №20 имеют variance ≈ 0, например 1e-29, под вопросом на удаление. Возможно, имеют вес, если подтвердится linear type features.
- Колонки №11, №16, №6, №15, №24, №25, №12, №17, №19 имеют ненулевую, но очень маленькую variance. Проверить связь с target.
- Колонка №0 отображает только ID двигателя, поэтому в выборке не будет участвовать.

### Correlation Features

| Feature 1 | Feature 2 | Correlation |
| ---: | ---: | ---: |
| 13 | 18 | 0.963157 |
| 15 | 16 | -0.846884 |
| 8 | 15 | 0.830136 |
| 12 | 17 | 0.826084 |
| 11 | 15 | -0.822805 |
| 8 | 16 | -0.815591 |
| 11 | 16 | 0.812713 |

Данные были подтверждены с помощью Scatter Plots, все перечисленные features имеют корреляцию.

| Features 15 vs 16 | Features 12 vs 17 |
| --- | --- |
| ![Scatter Plot: features 15 и 16](docs/images/on-device-predictive/correlation-15-16.png) | ![Scatter Plot: features 12 и 17](docs/images/on-device-predictive/correlation-12-17.png) |

На features 12 vs 17 значения лежат как будто на дискретной сетке/полосах. Это, скорее всего, значит, что значения одного или обоих сенсоров квантованы или округлены с фиксированным шагом.

## Target Analysis

### Target Metrics

| Метрика | Значение |
| --- | ---: |
| Mean | 75.52 |
| Median | 86.0 |
| Mode | [28] |
| Std | 41.555620558475596 |
| Variance | 1726.8695999999995 |
| Min | 7 |
| Max | 145 |
| Range | 138 |
| 95 percentile | 136.05 |

### Итог

Проанализировал общий RUL по двигателям, не в разрезе rows. По этим цифрам видно, что распределение довольно широкое. В результате RUL имеет высокий spread (std ≈ 41.6) и заметную асимметрию (mean < median). Диапазон широкий (7-145). В dataset представлено большое количество двигателей с разным сроком окончания службы, охватывая достаточно широкий статистический спектр устройств.

## Grouped Statistics

### Split Feature

Сделать split dataset-а решил на основе engine ID, чтобы проверить метрики и корреляции features в разрезе определённого engine.

### Top Correlation Features

| Feature 1 | Feature 2 | Mean corr | Median corr | Std corr | Min corr |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 0.810601 | 0.816341 | 0.039729 | 0.676283 |
| 1 | 16 | -0.789662 | -0.810574 | 0.052677 | -0.85988 |
| 15 | 18 | 0.215769 | 0.637394 | 0.716013 | -0.89038 |
| 15 | 16 | -0.791365 | -0.799357 | 0.068990 | -0.89044 |
| 13 | 18 | 0.651893 | 0.742501 | 0.340804 | -0.09192 |
| 13 | 15 | 0.378230 | 0.755699 | 0.593423 | -0.80516 |
| 8 | 18 | 0.209505 | 0.624856 | 0.690452 | -0.85468 |
| 1 | 13 | 0.385955 | 0.746635 | 0.567226 | -0.76815 |
| 1 | 18 | 0.223504 | 0.670325 | 0.685325 | -0.83650 |
| 8 | 15 | 0.780590 | 0.780973 | 0.057467 | 0.632776 |
| 8 | 13 | 0.365222 | 0.731016 | 0.572438 | -0.78692 |
| 16 | 18 | -0.197899 | -0.653262 | 0.697513 | -0.89264 |
| 1 | 8 | 0.781614 | 0.790356 | 0.037921 | 0.635681 |
| 11 | 15 | -0.761159 | -0.769566 | 0.075113 | -0.87998 |
| 15 | 17 | 0.699648 | 0.753072 | 0.172451 | 0.175386 |
| 1 | 17 | 0.687316 | 0.774206 | 0.189237 | 0.128698 |
| 8 | 16 | -0.760967 | -0.775370 | 0.070161 | -0.87151 |
| 12 | 15 | 0.701173 | 0.757670 | 0.169499 | 0.203020 |
| 1 | 12 | 0.683526 | 0.777609 | 0.192180 | 0.147402 |
| 13 | 16 | -0.357556 | -0.725701 | 0.578847 | -0.90007 |
| 16 | 17 | -0.685532 | -0.749196 | 0.180180 | -0.88389 |
| 11 | 16 | 0.741788 | 0.756403 | 0.085231 | 0.527857 |
| 11 | 18 | -0.189425 | -0.606437 | 0.673281 | -0.88818 |
| 1 | 11 | -0.761514 | -0.783062 | 0.059394 | -0.85518 |
| 12 | 16 | -0.681672 | -0.741420 | 0.178498 | -0.88521 |
| 11 | 12 | -0.667235 | -0.726700 | 0.176573 | -0.86812 |
| 12 | 17 | 0.654654 | 0.724172 | 0.206889 | 0.130383 |
| 8 | 12 | 0.675693 | 0.737017 | 0.164963 | 0.212033 |
| 8 | 17 | 0.676980 | 0.734195 | 0.158950 | 0.149781 |
| 11 | 17 | -0.665515 | -0.724763 | 0.171470 | -0.87789 |

### Top Relation Vector Correlations

| Pair | Median corr | Std | Strong % | Sign stability |
| --- | ---: | ---: | ---: | --- |
| 1 ↔ 15 | 0.816 | 0.040 | 68% | 100% positive |
| 1 ↔ 16 | -0.811 | 0.053 | 57% | 100% negative |
| 15 ↔ 16 | -0.799 | 0.069 | 50% | 100% negative |
| 8 ↔ 15 | 0.781 | 0.057 | 42% | 100% positive |
| 1 ↔ 8 | 0.790 | 0.038 | 38% | 100% positive |
| 11 ↔ 15 | -0.770 | 0.075 | 37% | 100% negative |
| 8 ↔ 16 | -0.775 | 0.070 | 35% | 100% negative |
| 11 ↔ 16 | 0.756 | 0.085 | 30% | 100% positive |
| 13 ↔ 18 | 0.742501 | 0.340804 | 48% | 96% positive |

Эти данные подтверждают корреляцию таких features, как:

- Feature 13 ↔ Feature 18;
- Feature 15 ↔ Feature 16;
- Feature 8 ↔ Feature 15;
- Feature 11 ↔ Feature 15;
- Feature 8 ↔ Feature 16;
- Feature 11 ↔ Feature 16.

Также появились новые корреляции с feature 1, которая отображает определённый цикл работы двигателя:

- Feature 1 ↔ Feature 15;
- Feature 1 ↔ Feature 16;
- Feature 1 ↔ Feature 8.

Данные по корреляциям можно использовать в будущем для облегчения dataset, но они не подтверждают то, что features не важны для prediction.

### Grouped Validation Engine Metrics

Сделал группировку по ID двигателя с выводом основных метрик и mean features. Mean features должны дать дополнительный сигнал к формированию подгрупп, если такие есть.

**N - общее количество циклов на двигатель.** Mean Residual = prediction − target.

| Group | N, cycles | MAE | RMSE | Mean Residual |
| ---: | ---: | ---: | ---: | ---: |
| 81 | 240 | 26.002998 | 30.887417 | -25.430799 |
| 82 | 214 | 10.158482 | 12.659639 | 6.178745 |
| 83 | 293 | 35.513066 | 43.138465 | -20.655822 |
| 84 | 267 | 53.633882 | 61.451327 | -53.184423 |
| 85 | 188 | 12.616463 | 14.500932 | 12.272263 |
| 86 | 278 | 48.590911 | 57.367246 | -48.113961 |
| 87 | 178 | 12.830670 | 14.580212 | 12.604282 |
| 88 | 213 | 8.253293 | 10.262264 | -6.254442 |
| 89 | 217 | 10.793324 | 13.617359 | 8.434481 |
| 90 | 154 | 52.559368 | 55.051679 | 52.338267 |
| 91 | 135 | 31.433058 | 33.608436 | 31.433058 |
| 92 | 341 | 69.208258 | 85.716748 | -65.316606 |
| 93 | 155 | 44.091941 | 46.280871 | 43.977700 |
| 94 | 258 | 44.915145 | 52.253930 | -44.330331 |
| 95 | 283 | 36.861100 | 47.187500 | -32.395311 |
| 96 | 336 | 62.649354 | 78.379379 | -58.176248 |
| 97 | 202 | 16.805585 | 19.506543 | 15.351407 |
| 98 | 156 | 41.356027 | 43.278301 | 41.287625 |
| 99 | 185 | 23.577455 | 25.426826 | 22.857579 |
| 100 | 200 | 6.410840 | 7.764275 | -0.966758 |

Средние feature 13 вынесены отдельно, чтобы таблица метрик не была слишком широкой. В PDF правый край этой колонки обрезан; ниже сохранены видимые цифры, без восстановления недостающих знаков.

| Group | Mean Feature 13, видимая часть |
| ---: | ---: |
| 81 | 9059.772… |
| 82 | 9087.531… |
| 83 | 9055.759… |
| 84 | 9055.437… |
| 85 | 9050.645… |
| 86 | 9065.968… |
| 87 | 9061.060… |
| 88 | 9081.514… |
| 89 | 9055.381… |
| 90 | 9051.968… |
| 91 | 9052.720… |
| 92 | 9071.812… |
| 93 | 9063.142… |
| 94 | 9066.583… |
| 95 | 9078.871… |
| 96 | 9057.228… |
| 97 | 9076.269… |
| 98 | 9066.985… |
| 99 | 9056.732… |
| 100 | 9064.058… |

Mean features не дают явного сигнала для формирования подгрупп.

## Focus Group Selection

| Engine | Причина выбора |
| ---: | --- |
| 100 | Лучший результат |
| 88 | Следующий лучший результат |
| 92 | Наихудший результат недооценки |
| 96 | Подтверждает наихудший результат недооценки |
| 90 | Наихудший результат с противоположной ошибкой: Mean Residual показывает переоценку |

На графиках focus group также есть engine 82.

### Focus Groups Signs

Группы engine 92 и engine 96 имеют 341 и 336 cycles соответственно - наибольшее количество в таблице. Engine 90 имеет 154 cycles, а наилучший результат engine 100 имеет 200 cycles.

Также заметил по статистике, что наилучший результат имеют группы, где полная длина жизни около 200 циклов, то есть имеется разброс ±20 cycles около 200.

- N < 200: сильно overpredict.
- N > 200: сильно underpredict.

Это повод сделать дополнительную сегментацию по длине жизни двигателя.

### Работа внутри Focus Group

Нарисовал predictions:

![Target и prediction по циклам: engines 90 и 92](docs/images/on-device-predictive/focus-predictions-90-92.png)

![Target и prediction по циклам: engine 96](docs/images/on-device-predictive/focus-predictions-96.png)

![Target и prediction по циклам: engines 100 и 82](docs/images/on-device-predictive/focus-predictions-100-82.png)

График показал, что к концу работы engines, проработавших больше или меньше 200 циклов, предсказания стабилизируются.

Судя по графику, нечто в features двигателей с длиной жизни меньше или больше 200 циклов не даёт модели хорошо оценить начальный цикл жизни двигателя.

## Feature Trajectory Analysis

### Исходные траектории

Сделал разбивку 20 оставшихся features по группам engine, получил достаточно noisy features:

| Feature 2 | Feature 3 |
| --- | --- |
| ![Feature 2: исходные траектории focus engines](docs/images/on-device-predictive/feature-02-raw.png) | ![Feature 3: исходные траектории focus engines](docs/images/on-device-predictive/feature-03-raw.png) |

| Feature 6 | Feature 7 |
| --- | --- |
| ![Feature 6: исходные траектории, первый график](docs/images/on-device-predictive/feature-06-raw-a.png) | ![Feature 7: исходные траектории](docs/images/on-device-predictive/feature-07-raw.png) |

Ещё один исходный график feature 6 из конспекта:

![Feature 6: исходные траектории, второй график](docs/images/on-device-predictive/feature-06-raw-b.png)

### Rolling Mean

Для полноценного анализа применим Rolling Mean в 15 cycles.

#### Feature 2

Стартовый уровень у всех engines разный. На протяжении всех циклов нет динамики роста в определённое положение: вверх / вниз.

Гипотеза в том, что похоже на то, что эта feature записывает примерно одинаковые диапазоны данных при поломке и реагирует на неё. Начальный уровень был разным, динамика была в определённых рамках у всех engines.

![Feature 2: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-02-rolling.png)

#### Feature 3

Стартовый уровень у всех engines разный. Engines 92 и 96 имеют самые высокие пики на протяжении всего графика, колебания и разброс значений увеличиваются для этих engines после 200-го цикла. Engine 90 имел большой пик перед тем, как перестал работать. Стабильного роста вверх или вниз нет, график является колебаниями.

![Feature 3: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-03-rolling.png)

#### Feature 6

Стартовый уровень у всех engines разный. Engines 90, 82, 100 заканчиваются примерно на одном значении. Feature по engines 82 и 100 начинает уходить вверх примерно после середины жизни двигателя. Feature по engine 90 идёт вверх и догоняет 82 и 100 почти на последних 20-25% циклов.

По остальным engines, особенно 92 и 96, рост планомерно растянут по всему циклу, не имеет резкого скачка вверх.

Я бы выделил дополнительную feature, нечто вроде `delta(40%)` - уровень изменений за последние 40% финальных циклов. Именно этот диапазон отображает резкий рост у интересующих нас features и показывает их динамику.

![Feature 6: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-06-rolling.png)

#### Feature 7

Стартовый уровень у всех engines разный. Динамика во многом повторяет feature 6. Feature по engine 90 идёт вверх и догоняет 82 и 100 почти на последних 20-25% циклов.

По остальным engines, особенно 92 и 96, рост планомерно растянут по всему циклу, не имеет резкого скачка вверх.

Я бы выделил дополнительную feature, нечто вроде `delta(40%)` - уровень изменений за последние 40% финальных циклов. Именно этот диапазон отображает резкий рост у интересующих нас features и показывает их динамику.

![Feature 7: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-07-rolling.png)

#### Feature 8

Стартовый уровень у всех engines разный. Динамика во многом повторяет feature 6 и feature 7. Feature по engine 90 идёт вверх и догоняет 82 и 100 почти на последних 20-25% циклов.

По остальным engines, особенно 92 и 96, рост планомерно растянут по всему циклу, не имеет резкого скачка вверх.

Я бы выделил дополнительную feature, нечто вроде `delta(40%)` - уровень изменений за последние 40% финальных циклов. Именно этот диапазон отображает резкий рост у интересующих нас features и показывает их динамику.

![Feature 8: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-08-rolling.png)

#### Feature 10

Engines, давшие наилучший результат, 100 и 82, имеют больше провалов пиков вниз, чем engines, показавшие наихудший результат. Для этой feature можно было бы выделить отдельную feature, показывающую расстояние на конкретном sample до минимального значения по текущей group, то есть расстояние до минимального значения, до которого упал пик.

Под конец жизни падений пиков вниз на графике не зафиксировано ни по одному engine, все они выравниваются на верхнем значении.

![Feature 10: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-10-rolling.png)

#### Feature 11

Стартовый уровень у всех engines разный. Динамика во многом повторяет feature 6, feature 7 и feature 8. Только вот теперь на графике все сенсоры показывают не рост, а падение.

Я бы выделил дополнительную feature, нечто вроде `delta(40%)` - уровень изменений за последние 40% финальных циклов. Именно этот диапазон отображает резкое падение у интересующих нас features и показывает их динамику.

![Feature 11: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-11-rolling.png)

#### Остальные траектории

| Feature 12 | Feature 13 |
| --- | --- |
| ![Feature 12: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-12-rolling.png) | ![Feature 13: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-13-rolling.png) |

| Feature 15 | Feature 16 |
| --- | --- |
| ![Feature 15: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-15-rolling.png) | ![Feature 16: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-16-rolling.png) |

| Feature 17 | Feature 18 |
| --- | --- |
| ![Feature 17: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-17-rolling.png) | ![Feature 18: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-18-rolling.png) |

| Feature 19 | Feature 21 |
| --- | --- |
| ![Feature 19: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-19-rolling.png) | ![Feature 21: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-21-rolling.png) |

| Feature 24 | Feature 25 |
| --- | --- |
| ![Feature 24: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-24-rolling.png) | ![Feature 25: Rolling Mean 15 cycles](docs/images/on-device-predictive/feature-25-rolling.png) |

### Общие наблюдения

В общем и целом почти все features имеют несколько схожих моментов:

- Разный Initial Level.
- Engines, показавшие более стабильный результат, имеют более резкий скачок в значениях на последних 40% своих циклов, чем engines, показавшие результат хуже.
- Требуются метрики, отображающие разницу в значениях относительно максимального / минимального значения текущего engine. Это потенциально может показать динамику не только роста, но и сопоставить точки каждого cycle с разницей min / max.
- Также обязательно считаем delta относительно своего стартового значения и финального.

## Iteration 1

### Data Cleaning & Feature Engineering & Scaling

#### Data Cleaning

Удалим 0 variance features на текущем этапе. А именно features №4, №5, №14, №22, №23.

Также удалим useless feature №0, которая обозначает ID двигателя, а не сенсорную информацию.

#### Feature Engineering

На сейчас нет доступа к более детальной расшифровке метрик и данных с сенсоров, из-за чего сложно сделать логические предположения о взаимодействии метрик между собой для генерации новых фич.

С учётом того, что в dataset всего 26 features, из которых 5 имеют нулевую variance и будут удалены до первого обучения, принял решение после первого теста применить PolynomialFeatures для расширения features с `degree=2`, с целью найти нелинейные дополнительные взаимозависимости между features.

Также в пользу этого решения сыграло то, что нет доступа к более детальной расшифровке метрик и данных с сенсоров, из-за чего сложно сделать логические предположения о взаимодействии метрик между собой.

#### Scaling

Применим StandardScaler к features.

### Model Training

#### DummyRegression

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RSS | 70112459.089602 | 27987668.918975 |
| RMSE | 65.913253 | 78.925103 |
| MAE | 54.703735 | 64.089337 |
| RMSE / STD(target) | 1.000000 | 1.018483 |
| RMSE / MinMax(target) | 0.182585 (18.26%) | 0.232133 (23.21%) |

| Train: Prediction vs Target | Train: Residual Histogram |
| --- | --- |
| ![Dummy: Train Prediction vs Target](docs/images/on-device-predictive/dummy-train-scatter.png) | ![Dummy: Train Residual Histogram](docs/images/on-device-predictive/dummy-train-residuals.png) |

![Dummy: Validation Residual Histogram](docs/images/on-device-predictive/dummy-validation-residuals.png)

#### LinearRegression

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RSS | 21601444.038725 | 12686244.818277 |
| RMSE | 36.586135 | 53.137159 |
| MAE | 27.355467 | 42.389842 |
| RMSE / STD(target) | 0.555065 | 0.685705 |
| RMSE / MinMax(target) | 0.101347 (10.13%) | 0.156286 (15.63%) |

| Train: Prediction vs Target | Train: Residual Histogram |
| --- | --- |
| ![LinearRegression: Train Prediction vs Target](docs/images/on-device-predictive/linear-train-scatter.png) | ![LinearRegression: Train Residual Histogram](docs/images/on-device-predictive/linear-train-residuals.png) |

| Validation: Prediction vs Target | Validation: Residual Histogram |
| --- | --- |
| ![LinearRegression: Validation Prediction vs Target](docs/images/on-device-predictive/linear-validation-scatter.png) | ![LinearRegression: Validation Residual Histogram](docs/images/on-device-predictive/linear-validation-residuals.png) |

**ИТОГИ:**

Validation RMSE снизился с 78.93 у Dummy до 53.14 - примерно на 32.7%. В данных есть наличие закономерностей и потенциал для обучения. Однако наблюдается возможный overfitting модели, ведь RMSE на Train dataset показал результат на уровне 36.6 vs 53.14 на Validation.

Можно опробовать кросс-валидацию внутри train с разделением по двигателям. Это покажет, модель систематически ошибается на отдельных двигателях или прежде всего на ранних стадиях их работы.

Судя по Scatter Plot, имеются признаки кривой нелинейной зависимости, точки образуют дугу, а не хаотично расположены около диагонали. Дальнейшие шаги - добавить PolynomialFeatures со степенью 2 в связке с Ridge для занижения коэффициентов коррелирующих features, попробовать охватить нелинейные зависимости линейной регрессией, а затем nonlinear baseline в виде decision tree.

Хвосты Histogram показывают сравнительно редкие, но сильные недооценки RUL. Именно такие большие ошибки заметно увеличивают RMSE, поскольку он учитывает квадрат ошибки.

#### PolynomialFeatures + Ridge

| Настройка | Значение |
| --- | ---: |
| Degree | 2 |
| Best alpha | 100 |

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RSS | 17054687.237993 | 10508479.493749 |
| RMSE | 32.508507 | 48.361719 |
| MAE | 22.632375 | 35.864467 |
| RMSE / STD(target) | 0.493201 | 0.624080 |
| RMSE / MinMax(target) | 0.090051 (9.01%) | 0.142240 (14.22%) |

| Train: Prediction vs Target | Train: Residual Histogram |
| --- | --- |
| ![PolynomialRidge без истории: Train Prediction vs Target](docs/images/on-device-predictive/polynomial-ridge-train-scatter.png) | ![PolynomialRidge без истории: Train Residual Histogram](docs/images/on-device-predictive/polynomial-ridge-train-residuals.png) |

| Validation: Prediction vs Target | Validation: Residual Histogram |
| --- | --- |
| ![PolynomialRidge без истории: Validation Prediction vs Target](docs/images/on-device-predictive/polynomial-ridge-validation-scatter.png) | ![PolynomialRidge без истории: Validation Residual Histogram](docs/images/on-device-predictive/polynomial-ridge-validation-residuals.png) |

**ИТОГИ:**

Validation RMSE снизился с 53.14 до 48.36, примерно на 9%. Судя по Scatter Train и Validation, Polynomial + Ridge смогла выявить закономерности, сделав графики более линейными и снизив RMSE. Но всё равно на Validation Histogram есть большие хвосты, где недооценка RUL доходит до экстремальных значений.

Также, судя по графику Scatter, заметил группирование некоторых данных, вижу 4 отдельных ветки данных. Надо однозначно провести Group Evaluation в разбивке по ID двигателей, возможно, у каждого двигателя есть персонализированные характеристики, не указанные в dataset.

Следующий шаг - пробую применить KNN для prediction по локальным группам.

#### KNN

| Метрика | Validation |
| --- | ---: |
| RSS | 10588986.115210 |
| RMSE | 48.546618 |
| MAE | 34.808159 |
| RMSE / STD(target) | 0.626466 |
| RMSE / MinMax(target) | 0.142784 (14.28%) |

| Validation: Prediction vs Target | Validation: Residual Histogram |
| --- | --- |
| ![KNN: Validation Prediction vs Target](docs/images/on-device-predictive/knn-validation-scatter.png) | ![KNN: Validation Residual Histogram](docs/images/on-device-predictive/knn-validation-residuals.png) |

**ИТОГ:**

Histogram показывает более сжатый график Validation. При малом RUL разброс небольшой, график показывает себя хорошо, лучше даже, чем в линейных моделях, а чем выше RUL, тем больше его недооценка и разброс.

Scatter уже смазывает очертания групп, видимых на PolynomialFeatures + Ridge, из-за специфики KNN к усреднению результатов соседей. Разброс стал более широким и случайным. Интересного мало - пробуем применять Decision Forest.

Скорее всего, в KNN близость по текущим признакам может недостаточно хорошо отражать близость по оставшемуся ресурсу.

Применим Decision Tree.

#### Decision Tree

Лучшие параметры: `depth=5`, `leaf=50`, `split=2`.

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RMSE | 33.55 | 50.08 |
| MAE | 23.76 | 36.31 |

CV RMSE: **34.72**. Пока validation RMSE хуже KNN (48.55) и PolynomialRidge (48.36). Оба лаунчера проверены.

Сохраняется большой разброс данных при высоком RUL.

#### GradientBoostingRegressor

Лучшие параметры:

- `n_estimators=100`;
- `learning_rate=0.05`;
- `max_depth=3`.

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RMSE | 31.71 | 48.23 |
| MAE | 21.95 | 34.88 |

CV RMSE: **33.32**. Validation RMSE немного лучше PolynomialRidge (48.36) и KNN (48.55). Проверены полный подбор и интеграция с обоими лаунчерами.

### Summary Iteration 1

Необходимо больше features с данными о динамике sensors, описанной в Grouped Statistics.

## Iteration 2

### Feature Engineering

Отправил свои наблюдения в GPT Sol 6.1 с данными графиков, без уточнения контекста задачи. Запросил формирование features, описывающих динамику существующих features по моим предположениям, на выходе получил **374 новых features**.

| Семейство | Что описывает | Количество |
| --- | --- | ---: |
| Начальный уровень | Среднее первых 15 наблюдений, отклонение от него, доля устойчивых отклонений за последние 30 циклов | 48 |
| Динамика | Rolling mean, стандартное отклонение, наклон тренда и изменение наклона | 128 |
| Положение относительно экстремумов | Расстояние до минимума/максимума, диапазон и относительное положение внутри него | 128 |
| Давность экстремумов | Количество циклов с последнего минимума и максимума | 64 |
| Редкие события feature 10 | Частота нижнего значения в двух окнах, время с последнего события и флаг его наличия | 4 |
| Совместная динамика features 13/18 | Совпадение или противоположность направлений трендов | 2 |
| **Итого** | | **374** |

### Model Training: PolynomialFeatures + Ridge

PolynomialRidge с `degree=2` для 20 standard features + 374 new features, всего **604 features**.

| Метрика | Train | Validation |
| --- | ---: | ---: |
| RSS | 13988612.497184 | 9179126.931183 |
| RMSE | 29.441672 | 45.199380 |
| MAE | 20.958091 | 32.228229 |
| RMSE / STD(target) | 0.446673 | 0.583272 |
| RMSE / MinMax(target) | 0.081556 (8.16%) | 0.132939 (13.29%) |

| Train: Prediction vs Target | Train: Residual Histogram |
| --- | --- |
| ![Iteration 2: Train Prediction vs Target](docs/images/on-device-predictive/history-ridge-train-scatter.png) | ![Iteration 2: Train Residual Histogram](docs/images/on-device-predictive/history-ridge-train-residuals.png) |

| Validation: Prediction vs Target | Validation: Residual Histogram |
| --- | --- |
| ![Iteration 2: Validation Prediction vs Target](docs/images/on-device-predictive/history-ridge-validation-scatter.png) | ![Iteration 2: Validation Residual Histogram](docs/images/on-device-predictive/history-ridge-validation-residuals.png) |

## Summary

Лучший результат дало **PolynomialFeatures + Ridge**, снижение с 48.36 до 45.19 - это **3.17 цикла, или примерно 6.6%**.

Без истории было 20 исходных, из них вышло 230 полиномиальных признаков, с историей - 604. Поэтому добавление 374 признаков дало улучшение примерно на 6.6%, если остальные настройки эксперимента совпадали.

Это показывает, что история содержит дополнительный сигнал, но не доказывает полезность всех 374 признаков. Многие из них связаны между собой; возможно, компактный набор даст такой же или лучший результат.

Фиксирую текущую гипотезу, что с текущим набором данных нельзя добиться среднего RMSE лучше 40. Возможно, сенсоры лучше описывают текущее состояние двигателя, чем точное количество циклов до отказа. История улучшила RMSE, но оставила заметную ошибку.

Самая полезная проверка для будущих гипотез - разделить ошибку по диапазонам истинного RUL, только для оценки:

- близко к отказу;
- средний оставшийся ресурс;
- большой оставшийся ресурс.

Если около отказа ошибка небольшая, а основной вклад в RMSE дают ранние циклы, это поддержит мою гипотезу: точный дальний прогноз плохо определяется наблюдаемыми сенсорами.
