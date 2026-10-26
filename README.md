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
12. [Group Evaluation: Life Stage](#group-evaluation-life-stage)
13. [Summary Regression](#summary-regression)
14. [Iteration 3: Classification](#iteration-3-classification)
15. [Deep Learning](#deep-learning)
16. [Идея на потом: две модели](#идея-на-потом-две-модели)
17. [Final Summary](#final-summary)

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

## Group Evaluation: Life Stage

### Split Feature

После Iteration 2 сделал дополнительную Group Evaluation по этапам жизни двигателя. Хочу проверить, на каком этапе текущие features и модель лучше описывают оставшийся ресурс: начальном, среднем или ближе к отказу.

Разбивку сделал отдельно для каждого engine по доле уже прожитых циклов:

```text
lifetime = cycle_current + target_RUL
life_fraction = cycle_current / lifetime
```

| Group | Доля прожитых циклов | Этап жизни |
| --- | --- | --- |
| Early | До 33% включительно | Начальный этап, большой оставшийся ресурс |
| Mid | Больше 33%, до 66% включительно | Средний этап |
| Late | Больше 66%, до 100% включительно | Последние 34% циклов, приближение к отказу |

Каждая группа объединяет соответствующие samples всех двигателей своей выборки. Это не split для обучения: train остаётся на engines 1-80, validation - на engines 81-100. Модель обучается на всех train samples, а её predictions после этого оцениваются по этапам.

Полную длину жизни знаем из завершённых траекторий. Использую её только для разбивки при оценке; `lifetime`, `life_fraction` и group не передаются в модель как features. В реальном prediction полная длина жизни ещё неизвестна.

### Grouped Validation Life Stage Metrics

Модель - **PolynomialFeatures degree 2 + Ridge**, те же **604 features**: 230 полиномиальных признаков из 20 исходных + 374 признака истории без расширения.

Mean Residual = prediction − target. Отрицательное значение показывает недооценку оставшегося ресурса, положительное - переоценку.

| Group | Samples | Engines | RUL min | RUL max | MAE | RMSE | Mean Residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early | 1474 | 20 | 91 | 340 | 53.501528 | 67.560595 | -37.186640 |
| Mid | 1484 | 20 | 46 | 228 | 30.667399 | 36.894009 | -6.593524 |
| Late | 1535 | 20 | 0 | 115 | 13.309291 | 16.760006 | 1.292541 |

Вклад каждой группы в общую квадратичную ошибку вынес отдельно:

| Group | RSS | RSS share % |
| --- | ---: | ---: |
| Early | 6727975.617458 | 73.296466 |
| Mid | 2019973.166592 | 22.006158 |
| Late | 431178.147133 | 4.697376 |

Границы RUL пересекаются между группами, потому что длина жизни engines разная. Группы описывают относительный этап жизни каждого двигателя, а не фиксированный диапазон оставшихся циклов.

### Summary Group Evaluation

В проведённых экспериментах модель лучше прогнозирует RUL ближе к отказу. Основной вклад в ошибку дают ранние циклы, где наблюдается систематическая недооценка ресурса. Добавление истории улучшило общий результат, но не решило проблему дальнего прогноза. Причина и предел достижимой точности остаются открытыми.

Количество samples в группах примерно одинаковое, но early даёт 73.3% общей RSS. На позднем этапе MAE составляет 13.31 цикла, RMSE - 16.76. Небольшой Mean Residual 1.29 не означает отсутствие ошибок: переоценки и недооценки могут компенсироваться.

### Гипотеза

Выдвигаю гипотезу, что по текущим features модель может самостоятельно распознавать поздний этап жизни двигателя. На этом этапе это ещё не проверено: в этой Group Evaluation этап задавался по известной полной длине жизни, а не определялся самой моделью. Далее решил проверить эту идею отдельной классификацией.

## Summary Regression

Лучший результат дало **PolynomialFeatures + Ridge**, снижение с 48.36 до 45.19 - это **3.17 цикла, или примерно 6.6%**.

Без истории было 20 исходных, из них вышло 230 полиномиальных признаков, с историей - 604. Поэтому добавление 374 признаков дало улучшение примерно на 6.6%, если остальные настройки эксперимента совпадали.

Это показывает, что история содержит дополнительный сигнал, но не доказывает полезность всех 374 признаков. Многие из них связаны между собой; возможно, компактный набор даст такой же или лучший результат.

Фиксирую текущую гипотезу, что с текущим набором данных нельзя добиться среднего RMSE лучше 40. Возможно, сенсоры лучше описывают текущее состояние двигателя, чем точное количество циклов до отказа. История улучшила RMSE, но оставила заметную ошибку.

Дополнительно проверил ошибку по относительным этапам жизни каждого двигателя: early, mid и late. Результат Group Evaluation поддерживает мою гипотезу: точный дальний прогноз плохо определяется текущими признаками и моделью. При этом невозможность улучшить результат и предел достижимой точности пока не доказаны.


## Iteration 3: Classification

### Идея

После Group Evaluation возникла идея: если точный дальний RUL предсказывается плохо, попробовать regression сменить на classification. Вместо точного количества оставшихся циклов хочу определять, в каком диапазоне ресурса находится engine: большой ресурс, средний, низкий или уже близко к отказу.

Сам создам target y для каждого sample на основе текущего RUL и диапазона, в который он попадает. Это другая постановка задачи: точное количество циклов заменяется категорией ресурса. Цель - проверить, может ли модель сама распознавать поздний этап по текущим features и истории sensors.

### Dataset & Target

Сделал отдельную копию dataset для classification. Исходный regression dataset сохранил.

```text
RUL = cycle_max - cycle_current
remaining_fraction = RUL / cycle_max
```

`remaining_fraction` сохраняется для каждого sample. Для обучения LogisticRegression доля переводится в дискретный класс.

Полная длина жизни используется только для создания targets. В samples не добавляю ни `cycle_max`, ни `remaining_fraction`, ни истинный класс ресурса. История features по-прежнему считается только по текущему и предыдущим циклам, отдельно для каждого engine.

### Первые группы ресурса

| Group | Remaining fraction | Состояние |
| --- | --- | --- |
| high | Больше 75%, до 100% | Большой ресурс |
| medium | Больше 50%, до 75% включительно | Средний ресурс |
| low | Больше 25%, до 50% включительно | Низкий ресурс |
| near_failure | Больше 5%, до 25% включительно | Близко к отказу |
| critical | От 0% до 5% включительно | Критический остаток |

Каждый sample получает ровно один класс. Процент оставшегося ресурса относительный: 5% у короткого и длинного engine - разное количество циклов.

### Model Training: LogisticRegression

Pipeline: **StandardScaler + LogisticRegression**. На вход поступают **394 features**: 20 исходных + 374 признака истории, без PolynomialFeatures.

Настройки: `C=1.0`, `max_iter=5000`, `random_state=42`. Scaling обучается только на train. Для multiclass prediction выбирается класс с максимальной вероятностью; отдельный threshold 0.5 здесь не применяется.

Split оставил прежний: engines 1-80 на train, engines 81-100 на validation.

| Метрика | Train | Validation |
| --- | ---: | ---: |
| Accuracy | 0.984261 | 0.850211 |
| Precision, macro | 0.986883 | 0.860261 |
| Log Loss | 0.060326 | 0.550762 |

Accuracy на новых validation engines - **85.02%**. Результат интересный, но общей метрики недостаточно: нужно посмотреть, какие groups путаются и сколько critical samples модель пропускает.

### Group Evaluation: Resource Classes

Сделал разбивку по истинным группам ресурса. Precision считаю с учётом predictions всех групп, а не только samples своего класса. Recall показывает долю найденных samples истинного класса. Accuracy внутри одной истинной группы совпадает с её Recall.

Log Loss для группы считается по всем вероятностям классов на samples этой истинной группы.

**Validation:**

| Group | Samples | Engines | Correct | Precision | Recall | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 1111 | 20 | 888 | 0.924037 | 0.799280 | 0.958727 |
| medium | 1120 | 20 | 906 | 0.750000 | 0.808929 | 0.570152 |
| low | 1126 | 20 | 1024 | 0.825806 | 0.909414 | 0.306623 |
| near_failure | 902 | 20 | 796 | 0.943128 | 0.882483 | 0.405308 |
| critical | 234 | 20 | 206 | 0.858333 | 0.880342 | 0.256452 |

Confusion Matrix: строки - истинная group, колонки - prediction.

| True / Prediction | high | medium | low | near_failure | critical |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 888 | 217 | 3 | 3 | 0 |
| medium | 73 | 906 | 141 | 0 | 0 |
| low | 0 | 85 | 1024 | 17 | 0 |
| near_failure | 0 | 0 | 72 | 796 | 34 |
| critical | 0 | 0 | 0 | 28 | 206 |

**ИТОГ:**

Большинство ошибок находится между соседними категориями. Из 234 critical samples модель распознала 206, остальные 28 отнесла к near_failure, а не к состояниям большого ресурса. Начальный ресурс по-прежнему различается хуже: заметная путаница между high/medium и medium/low.

Проверил идею объединить near_failure + critical в одно предупреждение: «ресурса осталось ≤25%». Если объединить истинные и предсказанные группы в этой validation-матрице, получается:

- Recall около **93.7%**: обнаружены 1064 из 1136 поздних samples.
- Precision около **98.2%**: 1064 верных предупреждения из 1084, остальные 20 - ложные.

Это стало основанием объединить два поздних класса и обучить отдельную итоговую конфигурацию.

### Объединение поздних групп

Объединил near_failure и critical. В итоговой классификации осталось четыре класса:

| Group | Remaining fraction |
| --- | --- |
| high | Больше 75%, до 100% |
| medium | Больше 50%, до 75% включительно |
| low | Больше 25%, до 50% включительно |
| near_failure | От 0% до 25% включительно |

Теперь near_failure означает предупреждение **«ресурса осталось ≤25%»**. Это доля полной жизни, а не фиксированное число циклов до отказа.

### Обучение на полном Train и проверка на Test

После выбора постановки обучил LogisticRegression на всём `train_FD001.txt`: **100 engines, 20 631 samples**. Train/validation split на этом финальном обучении не делал. Предыдущие validation-эксперименты остаются этапом исследования, а отдельный test используется для итоговой оценки выбранной конфигурации.

Для test взял `test_FD001.txt` и `RUL_FD001.txt`. Test-траектории обрываются до отказа, поэтому последний наблюдаемый cycle нельзя считать полной длиной жизни.

В `RUL_FD001.txt` для каждого test engine задан остаток циклов после его последнего наблюдения. Создание test-targets:

```text
lifetime = last_observed_cycle + RUL_at_last_observed_cycle
RUL_current = lifetime - cycle_current
remaining_fraction = RUL_current / lifetime
```

Эти значения используются только как ground truth для оценки. Модель не получает test RUL или lifetime в features, scaler не обучается на test. Train и test engines обрабатываются отдельно, даже если у них совпадают числовые ID.

Оценил два варианта:

1. Все **13 096 test samples** - prediction по всем доступным циклам.
2. Последний sample каждого engine - **100 наблюдений для 100 test engines**.

### Итоговые общие метрики

| Выборка | Samples | Engines | Accuracy | Precision, macro | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train full | 20 631 | 100 | 0.976104 | 0.976066 | 0.069949 |
| Test: all cycles | 13 096 | 100 | 0.843616 | 0.859033 | 0.544583 |
| Test: last cycle | 100 | 100 | 0.860000 | 0.801190 | 0.496485 |

Результат test по всем cycles - **84.36% Accuracy**, по последнему наблюдению каждого engine - **86.00% Accuracy**.

Это уже новая конфигурация с четырьмя классами, поэтому её Accuracy нельзя напрямую считать приростом или падением относительно предыдущей validation с пятью классами.

### Group Evaluation: все Test Cycles

| Group | Samples | Engines | Correct | Precision | Recall | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 5087 | 100 | 4408 | 0.884075 | 0.866523 | 0.519644 |
| medium | 4469 | 96 | 3612 | 0.776440 | 0.808235 | 0.601899 |
| low | 2696 | 69 | 2239 | 0.863479 | 0.830490 | 0.550006 |
| near_failure | 844 | 32 | 789 | 0.912139 | 0.934834 | 0.374090 |

Для предупреждения «ресурса осталось ≤25%» на всех test cycles получили **Recall 93.48% и Precision 91.21%**. Из 844 поздних samples обнаружены 789, пропущены 55; из 865 предупреждений 76 ложные.

### Group Evaluation: последнее наблюдение каждого Engine

| Group | Samples / Engines | Correct | Precision | Recall | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| high | 4 | 4 | 0.571429 | 1.000000 | 0.059583 |
| medium | 27 | 22 | 0.785714 | 0.814815 | 0.757120 |
| low | 37 | 28 | 0.933333 | 0.756757 | 0.772740 |
| near_failure | 32 | 32 | 0.914286 | 1.000000 | 0.011768 |

Confusion Matrix:

| True / Prediction | high | medium | low | near_failure |
| --- | ---: | ---: | ---: | ---: |
| high | 4 | 0 | 0 | 0 |
| medium | 3 | 22 | 2 | 0 |
| low | 0 | 6 | 28 | 3 |
| near_failure | 0 | 0 | 0 | 32 |

На последних test-наблюдениях обнаружены **все 32 двигателя** с остатком ресурса ≤25%. Дополнительно получили **3 ложных предупреждения** на engines класса low. Precision предупреждения - **91.43%**.

100% Recall здесь относится только к этим 32 test engines, а не означает гарантированное обнаружение всех будущих отказов. Samples одного engine связаны между собой, поэтому метрики по всем cycles не равны тысячам независимых проверок. Когда предупреждение впервые появляется и насколько стабильно держится по всей траектории, отдельно не проверял.

## Deep Learning

После Classical ML решил проверить Deep Learning regression на том же dataset. Хотел понять, сможет ли Neural Network лучше использовать sensors и их историю для prediction точного RUL. Отдельно проверил влияние Feature Engineering, Learning Rate, глубины, ширины и Dropout.

Начал с небольшой MLP на PyTorch. Идея - сначала получить baseline, затем менять признаки и настройки, сравнивая результат по тем же engines.

### Dataset & Training Configuration

Использовал `train_FD001.txt`, отдельный `test_FD001.txt` и `RUL_FD001.txt`. Для Train target остался прежним:

```text
RUL = last_cycle - current_cycle
```

Для Test траектории обрываются до отказа. Поэтому target для каждого sample считаю так:

```text
RUL_current = last_observed_cycle + RUL_at_last_observed_cycle - current_cycle
```

Полная длина жизни и test RUL используются только для targets и последующего анализа. Во вход модели их не добавляю.

Первый запуск сделал на split 85% / 15% по engines. Затем вернулся к split из ML-части:

| Выборка | Engine ID | Samples |
| --- | --- | ---: |
| Train | 1-80 | 16 138 |
| Validation | 81-100 | 4 493 |
| Test | Отдельный файл, 100 test engines | 13 096 |

Это split по двигателям. История одного engine не попадает одновременно в Train и Validation. Совпадающие числовые ID между Train и отдельным Test не означают, что это один и тот же двигатель.

Общие настройки экспериментов:

| Настройка | Значение |
| --- | --- |
| Framework | PyTorch, `nn.Module`, CPU |
| Input | Количество features после preprocessing |
| Hidden Layer Activation | ReLU после каждого hidden layer |
| Output | `nn.Linear(..., 1)`, без дополнительной activation |
| Weight Initialization | Default от `nn.Linear` |
| Loss | MSE |
| Optimizer | AdamW |
| Weight Decay | `1e-2` |
| Batch Size | 64 |
| Максимум Epochs | 40 |
| Early Stopping | `patience=5`, `min_delta=0.001`, мониторинг Validation MSE |
| Random Seed | 42 |

`StandardScaler` обучается только на Train. Те же преобразования применяются к Validation и Test. Target не масштабировал. PolynomialFeatures в этих DL-запусках не использовал.

После обучения возвращаю weights лучшей Validation Epoch. Итоговые Train / Validation / Test RMSE считаю этой моделью в `eval()` и `no_grad()`. Для Dropout это означает, что при итоговой оценке он отключён.

В логах сохраняю MSE по Epochs, RMSE и Mean Residual отдельно для каждой выборки, Learning Curves и до 25 engines с самым высоким RMSE. Если engines меньше 25, вывожу все. Mean Residual считаю как:

```text
Residual = prediction - target
```

Положительный residual означает переоценку RUL, отрицательный - недооценку.

Для Group Evaluation разделил engines по полной длине жизни:

- `100 <= lifetime < 200`;
- `200 <= lifetime < 300`;
- `lifetime >= 300`.

Каждый engine целиком относится к одной группе. Это группировка для анализа по известному target, а не дополнительная feature для модели.

### Первый запуск: 394 Features

Сначала передал в Neural Network признаки из прежнего Feature Engineering: **20 исходных + 374 признака истории = 394 features**.

Архитектура: `394 → 32 → ReLU → 1`, `lr=1e-3`. На этом этапе Train состоял из engines 1-85, Validation - 86-100.

| Метрика | Train | Validation |
| --- | ---: | ---: |
| Samples | 17 340 | 3 291 |
| RMSE | 20.591817 | 44.957644 |
| Mean Residual | 2.005379 | -11.056854 |

Лучшей стала Epoch 17 с Validation MSE **2021.189756**. Early Stopping остановил обучение после Epoch 22.

Результат по Train заметно лучше Validation. При этом меня не устраивало само устройство Feature Engineering: признаков много, сложно понимать вклад каждого семейства, появились сомнения насчёт data leakage. Само подозрение ещё не подтверждает наличие leakage. Решил сначала убрать всю сгенерированную историю и получить чистый baseline.

У этого первого запуска другой split. Поэтому напрямую сравнивать его Validation RMSE с дальнейшими запусками как чистый эффект изменения features нельзя. Test-метрики в сохранённом первом логе не приведены.

### Baseline: 20 Features без истории

Убрал Feature Engineering полностью. После удаления engine ID и прежних constant features осталось **20 исходных features**, включая текущий cycle. Вернул split 80% / 20% по engines.

Архитектура: `20 → 32 → ReLU → 1`, `lr=1e-3`.

| Метрика | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RMSE | 34.008650 | 49.365159 | 42.105555 |
| Mean Residual | 0.945463 | -15.373609 | -4.593604 |

Лучшая Epoch - 17, остановка после Epoch 22. Validation MSE лучшей модели - **2436.918916**.

Это отправная точка для проверки собственного набора признаков истории. Без истории модель уже обучается, но недооценка RUL у длинных engines сохраняется.

### Новый Feature Engineering: 200 Features

Решил самостоятельно определить более понятный набор features. Для каждой из 20 исходных колонок добавил историю только предыдущих циклов, отдельно внутри каждого engine.

| Семейство | Окно / расчёт | Новых features |
| --- | --- | ---: |
| Rolling Mean | 10, 20, 30 предыдущих циклов | 60 |
| Rolling Standard Deviation | Те же окна, `ddof=0` | 60 |
| Trend / Slope | Наклон линейного тренда за 30 предыдущих циклов | 20 |
| Change / Delta | Разница прошлых измерений с расстоянием 30 циклов | 20 |
| Acceleration | Разница slope двух соседних окон по 30 циклов | 20 |
| **Итого** | **20 исходных + 180 новых** | **200 inputs** |

Текущий sample остаётся в исходных features, но не входит в свои rolling-окна. Для cycle `t`:

```text
Rolling Mean / STD 30: cycles t-30 .. t-1
Slope recent: cycles t-30 .. t-1
Delta 30 = x[t-1] - x[t-31]
Acceleration = Slope[t-30 .. t-1] - Slope[t-60 .. t-31]
```

Delta требует 31 предыдущего наблюдения. Acceleration требует 60 предыдущих циклов. Пока полного окна нет, в первом варианте ставил **0 до StandardScaler**.

Проверки на изменение будущих значений, обрезку траектории и перемешивание engines подтвердили, что реализованная история не использует будущие циклы и не смешивает данные разных двигателей.

Сохранил один hidden layer на 32 neurons и `lr=1e-3`.

| Метрика | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RMSE | 28.104568 | 41.405179 | 38.283991 |
| Mean Residual | 1.031703 | -11.435608 | -4.136269 |

Лучшая Epoch - 16, остановка после Epoch 21. Validation MSE - **1714.388825**.

**ИТОГ:**

Относительно 20 исходных features Validation RMSE снизился с **49.37 до 41.41**, примерно на **16.1%**. Test RMSE снизился с **42.11 до 38.28**, примерно на **9.1%**.

Грамотно выбранная история дала заметное улучшение при той же небольшой сети. Новый набор оказался полезной отправной точкой. Сравнение с прежними 394 features остаётся ориентировочным из-за другого split первого запуска.

### Learning Rate: 1e-3 → 1e-2

Создал `dl_experiment_2.py` как копию первого эксперимента с 200 features. Поднял только Learning Rate до `1e-2`.

| Метрика | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RMSE | 27.660636 | 40.559169 | 38.312198 |
| Mean Residual | 5.489526 | -5.926111 | 1.576872 |

Лучшая Epoch теперь 5, остановка после Epoch 10. Validation MSE - **1645.046197**.

Обучение стало быстрее, Validation немного улучшился, Test практически не изменился. После лучшей Epoch Train MSE в целом продолжает снижаться, а Validation колеблется и не улучшает результат. Появляется overfitting. Решил проверить model capacity и управление Learning Rate.

### Model Capacity: глубина и ширина

Вернул `lr=1e-3` и добавил второй hidden layer на 16 neurons:

```text
200 → 32 → ReLU → 16 → ReLU → 1
```

Получил Train RMSE **28.474851**, Validation **41.420307**, Test **38.244414**. Лучшая Epoch - 8, остановка после Epoch 13.

Затем расширил оба hidden layers:

```text
200 → 64 → ReLU → 32 → ReLU → 1
```

Получил Train RMSE **28.685040**, Validation **41.121541**, Test **38.003571**. Лучшая Epoch - 4, остановка после Epoch 9.

**ИТОГ:**

Глубина и ширина не дали заметного устойчивого выигрыша относительно небольшой сети. Небольшие изменения метрик есть, но основной проблемы длинного прогноза они не решили. С увеличением сети лучший Validation достигается раньше, затем Train продолжает улучшаться без улучшения Validation.

### Learning Rate Scheduler: Reduce on Plateau

Вернул архитектуру `200 → 32 → ReLU → 1` и начальный `lr=1e-3`. Добавил `ReduceLROnPlateau`, который отслеживает Validation MSE.

Сначала выставил `patience=3`, `mode="min"`, `factor=0.1`, затем для сохранённого запуска изменил параметры:

```text
patience = 5
mode = "min"
factor = 0.1
cooldown = 5
```

Early Stopping при этом остался с `patience=5`.

Результат полностью повторил baseline с 200 features: Train RMSE **28.104568**, Validation **41.405179**, Test **38.283991**. Лучшая Epoch - 16, остановка после Epoch 21.

Проверил лог: **LR всё время оставался 1e-3**. Early Stopping сработал раньше первого снижения Learning Rate. Поэтому этот запуск не проверил эффект самого scheduler: сделать вывод, что Reduce on Plateau не помогает, по нему нельзя.

После этого убрал scheduler и вернул static Learning Rate.

### Dropout: p=0.2

При `lr=1e-3` добавил Dropout после ReLU:

```text
200 → 32 → ReLU → Dropout(0.2) → 1
```

| Метрика | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RMSE | 28.679392 | 41.892349 | 38.409246 |
| Mean Residual | -0.229780 | -13.005505 | -5.202913 |

Лучшая Epoch - 12, остановка после Epoch 17. Validation MSE - **1754.968904**.

В этой конфигурации Dropout не улучшил результат. Mean Residual на Train стал ближе к нулю, но RMSE на Validation и Test оказался немного хуже baseline. Небольшой общий bias сам по себе не означает точные predictions.

### Стресс-тест: 86 → 64 → 32

Решил сильнее увеличить capacity и проверить, изменится ли результат глобально. Сохранил предыдущую версию файла как отправную точку, убрал Dropout, поставил static `lr=1e-2` и три hidden layers:

```text
200 → 86 → ReLU → 64 → ReLU → 32 → ReLU → 1
```

| Метрика | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RMSE | 27.408647 | 40.420394 | 38.743278 |
| Mean Residual | 7.252125 | -4.267048 | 3.140052 |

Лучшая Epoch - 3 с Validation MSE **1633.808263**. Early Stopping после Epoch 8.

После Epoch 3 Train MSE снизился с **814.30 до 511.46**, но Validation так и не улучшил свой лучший результат. Здесь overfitting видно достаточно явно.

Относительно одного hidden layer на 32 с тем же `lr=1e-2` Validation RMSE изменился с **40.56 до 40.42**. Test RMSE вырос с **38.31 до 38.74**. Большая сеть не дала заметного общего преимущества.

### Разбор ошибок: engines 69, 67, 92, 96

После настройки модели вернулся к ошибкам конкретных engines. Выбрал два длинных двигателя из Train, 69 и 67, и два из Validation, 92 и 96. Для разбора использовал predictions стресс-теста.

| Engine | Выборка | Полная жизнь, cycles | RMSE первых 100 cycles | RMSE после cycle 200 | Доля RSS на первых 100 cycles |
| ---: | --- | ---: | ---: | ---: | ---: |
| 69 | Train | 362 | 123.855 | 14.572 | 78.7% |
| 67 | Train | 313 | 84.802 | 12.440 | 86.8% |
| 92 | Validation | 341 | 110.152 | 19.360 | 76.6% |
| 96 | Validation | 336 | 115.825 | 11.676 | 83.3% |

![Deep Learning: истинный и предсказанный RUL для engines 69, 67, 92 и 96](docs/images/on-device-predictive/dl-long-engine-trajectories.png)

На первом cycle predictions этих engines находятся примерно в диапазоне **217-224 RUL**, хотя истинные значения - **312-361**. Основная недооценка возникает в начале жизни. Позднее predictions приближаются к target, а в конце появляется небольшая переоценка.

В Train всего **два engines с полной длиной жизни 300+ cycles - 67 и 69**. При одинаковом возрасте их текущие sensors и mean за последние 30 cycles могут быть близкими к features двигателей с жизнью около 150-200 cycles. Например, на cycle 100 ближайшие Train-соседи engine 67 по выбранным сенсорным признакам имеют полную жизнь 166, 189 и 195 cycles.

Это поддерживает гипотезу, что текущие признаки недостаточно хорошо различают будущую продолжительность жизни на ранних этапах. При этом невозможность отличить такие engines по данным пока не доказана.

Дополнительно посмотрел кандидаты: изменение mean относительно первых 30 cycles и slope на 90 cycles. На одинаковом cycle 100 по 80 Train engines Spearman с полной длиной жизни составил **0.628** для изменения колонки 16 относительно начала и **0.613** для slope 90 колонки 24.

Это только исследовательские связи. Эти кандидаты в следующих сохранённых обучениях не добавлял; их влияние на RMSE не проверено. Для самих признаков использовались только прошлые cycles, полная длина жизни оставалась ответом для анализа.

### Отсутствие истории: 0 или NaN перед Scaling

При разборе Feature Engineering обнаружил влияние нулевых заглушек на StandardScaler. Например, для Rolling Mean 30 колонки 6 STD среди уже заполненных окон составляет около **0.336**, а при включении начальных нулей scaler использует STD около **229**.

Решил проверить гипотезу: может быть, реальные изменения истории слишком сильно сжимаются при scaling. Вернул один hidden layer на 32 neurons, static `lr=1e-2`, без Dropout и scheduler.

Новый порядок обработки:

```text
Нет полного окна истории → NaN
StandardScaler: fit только по доступным значениям Train
После scaling: оставшиеся NaN → 0
```

Количество inputs осталось **200**. Архитектура и настройки совпадают с предыдущим запуском одного слоя на 32 при `lr=1e-2`.

| Метрика | Нули до scaling | NaN → scaling → нули |
| --- | ---: | ---: |
| Train RMSE | 27.660636 | 26.159711 |
| Validation RMSE | 40.559169 | 42.129002 |
| Test RMSE | 38.312198 | 40.264755 |
| Train Mean Residual | 5.489526 | 3.393849 |
| Validation Mean Residual | -5.926111 | -8.344910 |
| Test Mean Residual | 1.576872 | -0.750902 |

В обоих случаях лучшая Epoch - 5, остановка после Epoch 10. В NaN-варианте Validation MSE лучшей модели - **1774.852847**.

**ИТОГ:**

С моим первоначальным значением **0 до scaling получилось лучше на Validation и Test**. После перехода на NaN Train RMSE улучшился примерно на 5.4%, но Validation ухудшился на 3.9%, Test - на 5.1%.

Предположение о вреде нулевого заполнения для качества в этом сравнении не подтвердилось. Мы изменили масштаб rolling-признаков и способ обозначения отсутствующей истории. Это повлияло на обучение, но более аккуратная обработка пропусков сама по себе не гарантирует лучший prediction.

Вариант с нулями сохраняю как более сильную отправную точку при проверенных настройках. Последний вариант кода `dl_experiment_2.py` соответствует проведённому NaN-эксперименту; таблицы описывают историю запусков, а не только текущий файл.

### Сравнение всех запусков на Split 80% / 20%

В таблице указаны hidden layers. ReLU стоит после каждого из них. Все строки, кроме baseline без истории, используют 200 features. Для истории используется заполнение нулями до scaling, кроме последней строки с NaN.

| Эксперимент | Hidden Layers | LR | Train RMSE | Validation RMSE | Test RMSE | Best / Stop Epoch |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 20 features, без истории | 32 | 1e-3 | 34.008650 | 49.365159 | 42.105555 | 17 / 22 |
| Новый Feature Engineering | 32 | 1e-3 | 28.104568 | 41.405179 | 38.283991 | 16 / 21 |
| Увеличение LR | 32 | 1e-2 | 27.660636 | 40.559169 | 38.312198 | 5 / 10 |
| Добавление глубины | 32 → 16 | 1e-3 | 28.474851 | 41.420307 | 38.244414 | 8 / 13 |
| Увеличение ширины | 64 → 32 | 1e-3 | 28.685040 | 41.121541 | 38.003571 | 4 / 9 |
| Reduce on Plateau, LR не изменился | 32 | 1e-3 | 28.104568 | 41.405179 | 38.283991 | 16 / 21 |
| Dropout, p=0.2 | 32 | 1e-3 | 28.679392 | 41.892349 | 38.409246 | 12 / 17 |
| Стресс-тест | 86 → 64 → 32 | 1e-2 | 27.408647 | 40.420394 | 38.743278 | 3 / 8 |
| NaN до scaling | 32 | 1e-2 | 26.159711 | 42.129002 | 40.264755 | 5 / 10 |

![Deep Learning: Learning Curves для базовой сети, изменения LR, стресс-теста и NaN](docs/images/on-device-predictive/dl-learning-curves-comparison.png)

Лучший Validation RMSE среди этих запусков - **40.42** у стресс-теста. Минимальный Test RMSE в таблице - **38.00** у сети 64 → 32. Эти небольшие различия не означают, что найдена устойчивая лучшая architecture.

Запуски выполнены с одним seed. Не оценивал разброс по нескольким seed или нескольким разбиениям engines. Test просматривал после каждого эксперимента, поэтому для независимого подтверждения итогового решения нужна отдельная проверка на неиспользованных данных.

### Summary Deep Learning

Основное улучшение дал Feature Engineering: удалось снизить Validation RMSE с **49.37 до 41.41** и Test RMSE с **42.11 до 38.28** при сохранении небольшой сети.

Настройка Learning Rate, ширины, глубины и Dropout не дала сопоставимого общего выигрыша. Увеличение capacity ускоряет снижение Train Loss, но после нескольких Epochs появляется overfitting. Scheduler в сохранённом запуске не успел изменить LR, поэтому его полезность отдельно не проверена. Переход с нулей на NaN тоже не улучшил результат на новых engines.

Мой вывод на этом этапе: дальше играть с настройками модели в рамках этих тестовых запусков не вижу большого смысла. И Classical ML, и Deep Learning находят закономерности, но точность дальнего прогноза RUL на текущих inputs остаётся ограниченной. Особенно это заметно на ранних cycles длинных engines.

На практике я бы обсуждал эту проблему с инженерами. Помимо sensors запросил бы дополнительные данные, которые могут объяснять различия между двигателями:

- марку, тип и модификацию, если они отличаются между engines;
- возраст и наработку до начала наблюдения;
- историю обслуживания, ремонтов и замен компонентов;
- фактическую нагрузку, режимы работы и условия эксплуатации;
- сведения о sensors и их калибровке.

Такие данные могут дать модели информацию, которую одни текущие sensors и их история не описывают. Эти поля должны быть доступны на момент prediction. Насколько они улучшат результат, нужно проверять экспериментом; заранее обещать конкретный процент нельзя.

**Фиксирую итог:** проверенные ML- и DL-подходы на текущем dataset не дали желаемой высокой точности прогноза полного RUL. Для следующего этапа считаю главным направлением поиск дополнительных инженерных данных о двигателях и уточнение постановки задачи. Это мой вывод по проведённым экспериментам; математический предел точности на этом dataset не установлен.

## Идея на потом: две модели

Финальная идея - оставить классификацию как первый этап и добавить вторую модель для точного RUL ближе к отказу:

```text
Sensors + доступная история
        ↓
Классификатор ресурса
        ↓
Prediction: осталось ≤25% ресурса
        ↓
Регрессор остатка RUL в циклах
```

Первая модель отвечает, что двигатель вошёл в поздний этап. Вторая должна оценивать, сколько конкретно циклов осталось. Это потенциально позволяет не требовать точного дальнего прогноза на ранних стадиях, где текущая regression ошибается сильнее всего.

**Эта связка пока не реализована и совместно не проверена.** Регрессор должен запускаться по prediction классификатора, а не по истинному классу. При проверке нужно учитывать ложные предупреждения, пропуски позднего этапа и стабильность сигнала по каждому engine.

Прежняя regression-оценка late относится к последним 34% жизни, а классификация near_failure - к оставшимся ≤25%. RMSE 16.76 из прежней таблицы нельзя автоматически переносить на будущий регрессор.

Текущий test уже использован для финальной оценки. Если возвращаться к этой идее, новые решения выбирать на train/validation, а для независимого подтверждения всей связки потребуется отдельная оценка на неиспользованных данных.

## Final Summary

В этом проекте прошёл от анализа исходных features и корреляций до Group Evaluation по engines, разбора траекторий и собственных гипотез о динамике sensors. По этим наблюдениям добавил историю и проверил, что она даёт дополнительный сигнал.

Regression улучшилась с validation RMSE **48.36 до 45.19**, но точный дальний прогноз остался слабым. Разбивка по этапам показала, что ранние cycles дают основную часть ошибки, а ближе к отказу prediction заметно точнее. Причина и предел достижимой точности остаются открытыми.

Следующим шагом сменил постановку на classification ресурса. Она позволила проверить идею распознавания позднего этапа напрямую. Итоговая модель на отдельном test получила **84.36% Accuracy по всем cycles** и **86.00% по последнему наблюдению каждого engine**. Для предупреждения «ресурса осталось ≤25%» на всех test cycles Recall составляет **93.48%**, Precision - **91.21%**.

Фиксирую итог: текущие признаки и LogisticRegression позволяют различать категории ресурса и обнаруживать поздний этап на проверенных test engines. Это результат исследовательского проекта, а не подтверждение готовности к реальной эксплуатации. Скорость и память на целевом устройстве, устойчивость предупреждений и надёжность на других условиях не проверены.

После classification отдельно проверил Deep Learning regression. Добавление собственной истории улучшило Validation RMSE с **49.37 до 41.41**, Test RMSE - с **42.11 до 38.28**. Изменение Learning Rate, глубины, ширины и Dropout не дало сопоставимого устойчивого выигрыша. На длинных engines сохранилась сильная недооценка раннего RUL.

На этом текущий этап экспериментов считаю завершённым. Следующий шаг вижу в работе с инженерами и дополнительных данных о типе двигателя, его истории обслуживания, нагрузке и условиях эксплуатации. Проверенные ML- и DL-модели используют доступный сигнал, но для желаемого качества дальнего прогноза текущих inputs может быть недостаточно. Предел достижимой точности пока не доказан.

Идею двух моделей - классификатор для предупреждения и регрессор для остатка RUL после него - оставляю на потом как отдельное продолжение.
