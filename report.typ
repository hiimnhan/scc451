#import "@preview/charged-ieee:0.1.4": ieee

#let appendix(body) = {
  set heading(numbering: "A.1", supplement: [Appendix])
  counter(heading).update(0)
  show heading: it => {
    set text(11pt, weight: 400)
    set align(center)
    show: block.with(above: 15pt, below: 13.75pt, sticky: true)
    show: smallcaps
    [#it.supplement #counter(heading).display():]
    h(0.3em)
    it.body
  }
  body
}

#set table(
  align: (left, left, left, left),
  stroke: 0.5pt + black,
  fill: (none, none, none, none),
  inset: 6pt,
)

#show: ieee.with(
  title: [SCC451 Coursework Report],
  authors: (
    (
      name: "Nhan Nguyen",
      department: [School of Computing and Communications],
      organization: [SCC451 Machine Learning],
    ),
  ),
  index-terms: (
    "Scientific writing",
    "Typesetting",
    "Document creation",
    "Syntax",
  ),
  figure-supplement: [Fig.],
)

= Task 1: Basel Climate Dataset
The Basel Climate Dataset (ClimateDataBasel.csv) is a subset of publicly available weather data from Basel, Switzerland, obtained from @meteoblue. It contains 1,763 records with 18 features collected during the summer and winter seasons from 2010 to 2019.

As the dataset lacks column headers, assigning appropriate names to each feature is necessary to ensure readability, consistency, and efficient data handling during analysis. The complete list of column names and their meanings is provided in @column-name .

== Preprocessing

Preprocessing is an important step ensure our data is high quality and reliable for analysis. Our process for this step is as follow.
1. Data cleaning: handling missing data, identifying and treating outliers, resolving data inconsistencies and removing duplicates.
2. Feature scaling.
3. Feature selection / extraction.
\
=== Data Cleaning

First, import data file as DataFrame using `pandas` package. We named it `basel_df`.

*Handling missing data*
By using `isna()` function of Pandas. We can see that the dataset has no missing value. Code for missing value detection is in @missing-val (@app-code).


*Identifying and treating outliers / noise* Outliers are data points that deviate significantly from the overall pattern of the dataset. They can take unusually high or low values compared to the majority of observations, potentially indicating measurement errors, or rare but valid phenomena @hanDataMining2011. In the following step, we will detect potential anomalies across all features and decide on appropriate treatments, such as removal, transformation, or imputation, depending on their impact on the dataset.

There are some techniques to detect anomalies in data. Here we consider four methods commonly used in practice, i.e. Z-score (Standard Deviation Method), IQR Method, Isolation Forest and Modified Z-score (Median Absolute Deviation). The advantages and disadvantages of these methods are listed in @table-outlier-detection. Based on the dataset's mixture of symmetric and heavy-tailed distributions (@fig-distribution), we decide to employ the Modified Z-score for outliers detection task due to its robustness, simplicity and resistant to the influence of outliers (see implemented code and result in @fig-mod-zscore).

The deteted anomalies are categorised into two types: (1) physically impossible values (e.g., negative precipitation, humidity above 100%), (2) reasonable extreme values (e.g., heavy rainfall or strong gusts). Each type is treated differently that the dataset remains both statistically consistent and physically realistic. With the first type, the outliers are replaced with missing values and later imputed using k-Nearest Neighbors (kNN) imputation for its decent performance @jadhavCompareImputation (see code implemented in @fig-outlier-treatment).
#figure(
  table(
    columns: (auto, auto, auto),
    table.header([Method], [Advantage], [Disadvantages]),
    [Z-score (Standard Deviation Method)],
    [
      Simple and fast to compute.\
      Works well for approximately normal distributions. \
      Easy to interpret (values |z| > 3 often flagged).
    ],
    [
      Sensitive to outliers and non-Gaussian data. \
      Not reliable for skewed or heavy-tailed features.
    ],

    [IQR Method],
    [
      Simple and interpretable. \
      Robust to outliers; no distributional assumptions. \
      Suitable for skewed climate data (e.g., precipitation).
    ],
    [
      Univariate only; ignores multivariate interactions. \
      May flag valid extremes in naturally variable data.
    ],

    [
      Isolation Forest,
      "Randomly partition data; anomalies have shorter average path length."],
    [
      Model-free and scalable to large datasets. \
      Handles non-linear, high-dimensional data. \
      Works well for mixed seasonal patterns and irregular weather events.
    ],
    [
      Randomness may cause variability between runs. \
      Requires hyperparameter tuning for small datasets. \
      Less interpretable than statistical methods.
    ],

    [Modified Z-score (Median Absolute Deviation)],
    [
      Robust to skewness and extreme values. \
      Non-parametric (no assumption of normality). \
      Intepretable threshold (|M| > 3.5 commonly used).
    ],
    [
      Univariate: ignores feature correlations. \
      May miss contextual (multivariate) anomalies
    ],
  ),
  caption: [Anomaly detection methods],
)<table-outlier-detection>

*Feature Scaling* Because the dataset includes features in different measurement unit (e.g., temperature in °C, humidity in %, pressure in hPa, precipitation in mm, and wind in km/h), directly compare raw values would bias distance-based clustering algorithm. Therefore, feature scaling is applied to bring all variables to comparable baseline and ensure they contribute equally in learning process.

There are three methods considered, including Z-score standardization, Min-max Normalization and Robust Scaling. @table-feature-scaling shows the benefits and tradeoffs of these methods. Among these, Robust Scaling is the most suitable because of its robustness with skewed distribution and outliers (see code implemented in @fig-feature_scaling). @fig-befor-after-scaling shows the differences between before and after scaling. Before scaling, the data distributions are varied among features but after applying Robust Scaling method, we can see mostly distributions now look similar.

#figure(
  image("feature_scaling.png"),
  caption: [Before and after scaling],
)<fig-befor-after-scaling>

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([Method], [Advantage], [Disadvantages]),
    [Z-score Standardization],
    [
      Centres data to mean 0 and variance 1\
      Works well for approximately normal distributions. \
      Common choice for distance-based algorithm (e.g., K-Means)
    ],
    [
      Sensitive to outliers and non-Gaussian data. \
      Not ideal for skewed or heavy-tailed features.
    ],

    [Min-max Normalization],
    [
      Rescales data to a bounded range [0, 1] \
      Preserves relative relationship among values \
      Useful for distance-based algorithms (e.g., k-Nearest Neighbors)
    ],
    [
      Extreme sensitive to outliers \
    ],

    [Robust Scaling],
    [
      Robust to outliers and skewed distributions. \
      Maintains feature comparability across mixed-scale data. \
      Suitable for heavy-tailed variables.
    ],
    [
      not optimal for symmetric, normal distribtion. \
      computationally expensive.
    ],
  ),
  caption: [Feature Scaling Methods],
)<table-feature-scaling>


#bibliography("refs.bib")

#show: appendix
= Code <app-code>
#figure(
  image("basel_missing_value.png"),
  caption: [Code and result of detecting missing value.],
)<missing-val>

#figure(
  image("mod_zscore.png"),
  caption: [Modified Z-score code and result],
)<fig-mod-zscore>

#figure(
  image("outlier_treatment.png"),
  caption: [Treating different types of outlier],
)<fig-outlier-treatment>

#figure(
  image("feature_scaling_code.png"),
  caption: [Feature Scaling],
)<fig-feature_scaling>
= Table and Figures <app-table>

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([*Column Name*], [*Description*], [*Unit*]),
    [temp_min_c], [Minimum daily temperature], [°C],
    [temp_max_c], [Maximum daily temperature], [°C],
    [temp_mean_c], [Mean daily temperature], [°C],
    [rh_min_pct], [Minimum daily relative humidity], [%],
    [rh_max_pct], [Maximum daily relative humidity], [%],
    [rh_mean_pct], [Mean daily relative humidity], [%],
    [slp_min_hpa], [Minimum daily sea level pressure], [hPa],
    [slp_max_hpa], [Maximum daily sea level pressure], [hPa],
    [slp_mean_hpa], [Mean daily sea level pressure], [hPa],
    [precip_mm], [Total daily precipitation], [mm],
    [snow_cm], [Total daily snowfall], [cm],
    [sunshine_min], [Total sunshine duration per day], [min],
    [gust_min_kmh], [Minimum daily wind gust speed], [km/h],
    [gust_max_kmh], [Maximum daily wind gust speed], [km/h],
    [gust_mean_kmh], [Mean daily wind gust speed], [km/h],
    [wind_min_kmh], [Minimum daily wind speed], [km/h],
    [wind_max_kmh], [Maximum daily wind speed], [km/h],
    [wind_mean_kmh], [Mean daily wind speed], [km/h],
  ),
  caption: [Table columns name and their meaning],
)<column-name>

#figure(
  image("dist_all_features.png"),
  caption: "Distribution of all features",
)<fig-distribution>



