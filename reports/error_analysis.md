# Where the model fails

## By property type

|                                      |   homes |   mdape |   mape |   median_signed |
|:-------------------------------------|--------:|--------:|-------:|----------------:|
| ('list_aware', 'Co-Op Apt')          |      48 |    5.3  |   5.59 |            0.66 |
| ('list_aware', 'Condo Apt')          |    4843 |    3.32 |   3.94 |            0.08 |
| ('list_aware', 'Condo Townhouse')    |     616 |    3.96 |   4.85 |           -2.11 |
| ('list_aware', 'Detached')           |    3123 |    4.88 |   6.27 |           -0.77 |
| ('list_aware', 'Freehold Townhouse') |     405 |    5.37 |   6.69 |           -0.45 |
| ('list_aware', 'Semi-Detached')      |    1034 |    5.41 |   6.55 |           -1.33 |
| ('off_market', 'Co-Op Apt')          |      48 |   11.39 |  19.21 |           -0.26 |
| ('off_market', 'Condo Apt')          |    4843 |    6.56 |   8.57 |            0.54 |
| ('off_market', 'Condo Townhouse')    |     616 |    6.31 |   7.66 |           -1.89 |
| ('off_market', 'Detached')           |    3123 |    9.94 |  12.52 |           -1.61 |
| ('off_market', 'Freehold Townhouse') |     405 |   10.26 |  11.61 |            0.92 |
| ('off_market', 'Semi-Detached')      |    1034 |    8.95 |  11.15 |           -2.08 |

## By month (each predicted only from earlier sales)

|                           |   homes |   mdape |   mape |   median_signed |
|:--------------------------|--------:|--------:|-------:|----------------:|
| ('list_aware', '2025-12') |    1238 |    3.85 |   4.97 |            1.09 |
| ('list_aware', '2026-01') |    1058 |    3.95 |   4.79 |            0.17 |
| ('list_aware', '2026-02') |    1442 |    3.87 |   5.04 |           -1.03 |
| ('list_aware', '2026-03') |    1816 |    4    |   5.25 |           -1.17 |
| ('list_aware', '2026-04') |    2224 |    3.95 |   5.11 |           -0.68 |
| ('list_aware', '2026-05') |    2291 |    4.07 |   5.26 |           -0.25 |
| ('off_market', '2025-12') |    1238 |    7.91 |  10.55 |            2.97 |
| ('off_market', '2026-01') |    1058 |    7.42 |   9.67 |           -1.16 |
| ('off_market', '2026-02') |    1442 |    7.85 |  10.13 |           -2.67 |
| ('off_market', '2026-03') |    1816 |    7.92 |  10.2  |           -1.18 |
| ('off_market', '2026-04') |    2224 |    8    |  10.03 |           -0.35 |
| ('off_market', '2026-05') |    2291 |    7.72 |  10.37 |            0.34 |

## Hardest neighbourhoods (at least 30 validation homes)

| track      | community     |   homes |   mdape |   mape |   median_signed |
|:-----------|:--------------|--------:|--------:|-------:|----------------:|
| off_market | Maplepark     |      36 |   15.83 |  24.07 |           -5.27 |
| off_market | Hazelfield    |      68 |   15.58 |  17.26 |          -11.17 |
| off_market | Cypressglen   |      54 |   12.64 |  18.21 |            2.18 |
| off_market | Sycamoredale  |      39 |   12.59 |  14.01 |           -7.37 |
| off_market | Willowpark    |      58 |   12.31 |  14.63 |           -2.12 |
| off_market | Sequoiafield  |      36 |   12.23 |  12.56 |           -9.86 |
| off_market | Lindenglen    |      62 |   12.12 |  14.08 |           -0.67 |
| off_market | Birchgrove    |      62 |   11.68 |  11.72 |           -9.17 |
| off_market | Sycamorehaven |      31 |   11.5  |  10.74 |            3.07 |
| off_market | Sumacfield    |      52 |   11.1  |  15.04 |            8.22 |
| list_aware | Oakview       |      30 |    7.69 |   6.92 |           -4.35 |
| list_aware | Sycamorehaven |      31 |    7    |   7.57 |            1.73 |
| list_aware | Sequoiafield  |      36 |    6.48 |   6.98 |           -6.34 |
| list_aware | Aspenwood     |      35 |    6.47 |   7.38 |           -2.38 |
| list_aware | Maplevale     |      93 |    6.4  |   7.07 |           -0.96 |
| list_aware | Maplepark     |      36 |    6.37 |   8.37 |           -0.6  |
| list_aware | Hazelbourne   |      42 |    6.35 |   6.72 |           -2.96 |
| list_aware | Elmbourne     |      36 |    6.09 |   6.85 |           -2.67 |
| list_aware | Cedarhaven    |      40 |    6.09 |   6.79 |           -1.45 |
| list_aware | Magnoliabank  |      43 |    6.05 |   7.57 |           -4.44 |

## Easiest neighbourhoods

| track      | community      |   homes |   mdape |   mape |   median_signed |
|:-----------|:---------------|--------:|--------:|-------:|----------------:|
| list_aware | Cedargrove     |      74 |    2.73 |   4.08 |           -0.89 |
| list_aware | Larchbank      |      32 |    2.94 |   3.94 |           -0.82 |
| list_aware | Mulberryhollow |      36 |    2.95 |   4.06 |           -0.44 |
| list_aware | Mulberryhaven  |      60 |    2.98 |   4.22 |           -0.72 |
| list_aware | Walnutfield    |     127 |    2.98 |   3.73 |           -0.03 |
| off_market | Mulberryhaven  |      60 |    4.54 |   6.34 |           -1.09 |
| off_market | Ashbank        |      69 |    5.06 |   8.15 |            1.6  |
| off_market | Sprucehaven    |      55 |    5.23 |   7.25 |           -3.47 |
| off_market | Elmgardens     |     101 |    5.23 |   7.02 |           -1.77 |
| off_market | Chestnutridge  |      32 |    5.26 |   8.66 |           -0.4  |
