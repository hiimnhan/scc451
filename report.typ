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

*Feature Scaling* Because the dataset includes features in different measurement unit (e.g., temperature in °C, humidity in %, pressure in hPa, precipitation in mm, and wind in km/h), directly compare raw values would bias distance-based clustering algorithm. Therefore, feature scaling is applied to bring all variables to comparable baseline and ensure they contribute equally in learning process.

There are three methods considered, including Z-score standardization, Min-max Normalization and Robust Scaling. @table-feature-scaling shows the benefits and tradeoffs of these methods. Among these, Robust Scaling is the most suitable because of its robustness with skewed distribution and outliers (see code implemented in @fig-feature_scaling). All the rest methods are severely impacted by outliers and skewness because they depend on the mean (Z-score standardization) or the min and max of data (Min-max Normalization) @deAmorimTheChoiceOfScaling2023. Robust Scaling is a method that use median and interquartile range (IQR) for standardization. For each feature $x$,
$
  x' = frac(x - "median"(x), "IQR"(x))
$
where
$
  "IQR"(x) = Q_3(x) - Q_1(x)
$
The median is less sentitive to outliers than the mean and the IQR measure the spread of the central 50% of data, ignoring extreme values.

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


*Identifying and treating outliers / noise* Outliers are data points that deviate significantly from the overall pattern of the dataset. They can take unusually high or low values compared to the majority of observations, potentially indicating measurement errors, or rare but valid phenomena @hanDataMining2011. In the following step, we will detect potential anomalies across all features and decide on appropriate treatments, such as removal, transformation, or imputation, depending on their impact on the dataset.
According to @dasOutlierDetectionTechniques2022, there are four primary categories of outlier dectection model: (1) Neigborhood-based model, (2) Subspace-based model, (3) Ensemble-based model and (4) Mixed-typed model. Here we analyze four methods corresponding to four categories, i.e. Local Outlier Factor (LOF) (Neigborhood-based), Subspace Outlier Degree (SOD) (Subspace-based), High Contrast Subspace (HiCF) (Ensemble-based) and Link-Based Outlier and Anomaly Detection (LOADED) (Mix-type). Our dataset has only 1750 rows so it is reasonable to choose the LOF method for detecting anomalies for its efficiency with small dataset (@table-outlier-detection). The LOF algorithm @Upadhyaya2012NearestNB is a neigborhood-based detection method which measure how the local density of a data point differs from that of its neighbors. The main idea is to calculate a score which tell us how much sparser a data point compared to its neigbors. A point in a dense region has a local density similar to its neighbors whereas in a sparse region (outlier), it has a local density lower than its neighbors.
$
  "LOF Score" approx frac("Average Density of Neighbors", "Density of a Point")
$

After detecting anomalies with LOF algorithm, a total of 0.74% of samples are marked as outliers. Instead of remove them, which may represent actual rare extreme events, we cap them under its 1st and 99th percentile which is called Winsorization technique. It will limit the effect of extreme values while preserving consistency of the distribution.

#figure(
  image("feature_scaling.png"),
  caption: [the differences between before and after scaling. Before scaling, the data distributions are varied among features but after applying Robust Scaling method, we can see mostly distributions now look similar.],
)<fig-befor-after-scaling>

#figure(
  table(
    columns: (auto, auto, auto),
    table.header([Method / Technique], [Advantages (Strengths)], [Disadvantages (Limitations)]),

    [Local Outlier Factor (LOF) — Neighborhood-based],
    [
      Measures how isolated a point is relative to its neighborhood density.\
      Handles variable-density datasets effectively.\
      Can reveal subtle, locally defined anomalies.
      Work well with small dataset.
    ],
    [
      Time complexity O(n²).\
      Not well-suited for large-scale datasets.\
      Requires tuning of neighborhood size and data scaling.
    ],

    [Subspace Outlier Degree (SOD) — Subspace-based],
    [
      Detects anomalies confined to certain feature subspaces.\
      Effective for correlated, high-dimensional attributes.\
      Useful when global distance metrics are unreliable.
    ],
    [
      Needs selection of suitable subspaces and thresholds.\
      Computationally demanding for large datasets.\
      May underperform if irrelevant dimensions are included.
    ],

    [High Contrast Subspace (HiCF) — Ensemble-based],
    [
      Identifies outliers based on contrasting behavior across subspaces.\
      Robust for complex, high-dimensional data.\
      Gains stability through ensemble integration.
    ],
    [
      Computationally expensive due to repeated subspace analysis.\
      Requires well-defined contrast criteria.\
      Unsuitable for real-time or streaming applications.
    ],

    [Link-Based Outlier and Anomaly Detection (LOADED) — Mix-type / Hybrid],
    [
      Achieves high detection accuracy and strong true-positive rates.\
      Supports both categorical and numerical attributes.\
      Combines diverse detection mechanisms adaptively.
    ],
    [
      Complex model structure with several parameters to tune.\
      Computational cost not fully reported but expected to be high.\
      Interpretation can be difficult due to hybrid design.
    ],
  ),
  caption: [Anomaly detection methods],
)<table-outlier-detection>


*Feature selection and extraction* Feature selection and extraction are two important steps in data preprocessing. The goal of these two is to identify and keep only valuable variables that contribute most to algorithm. By applying appropriate methods, we can build a more robust, simpler and less time-consuming as well as outstanding performance model. Feature selection focuses on selecting most relevant features that necessarily involve in predictive model while drop the rest features. Feature extraction means transforming original features that in high-dimensional into new features with lower-dimensional representation without losing any informative value.

In this project, we use Pearson's product-moment correlation coefficient (PMCC) @puthEffectiveUsePearsons2014 for finding the correlation between features. The Pearson correlation method is the most commonly used in practice @nettletonSelectionOfVariables2014. It measures how strongly two variables change together and in what direction. Given two variables $X$ and $Y$,
$
  r = frac("cov"(X, Y), sigma_X sigma_Y)
$
where $"cov"(X, Y)$ is the covariance between $X$ and $Y$ and $sigma_X$, $sigma_Y$ are the standard deviations of $X$ and $Y$, respectively. The value of $r$ is in range $[-1, 1]$. The higher in value of $r$ the tighter in relationship between $X$ and $Y$. As shown in @fig-corr, temperature features (i.e. `temp_min_c`, `temp_mean_c`, `temp_max_c`) are highly correlated with each other. The pressure features (i.e. `slp_min_hpa`, `slp_max_hpa`, `slp_mean_hpa`) are also correlated. Similarly, wind and gust features are correlated. The rest features are weakly correlated. From this heatmap, we can easily identify which features providing similar information and which are unique. It is useful for feature selection and avoiding redundancy. From this information, we can easily remove all high correlated columns (see code implemented and columns removed in @fig-remove-high-corr).

We use Principal Component Analysis (PCA) for dimensionality reduction. With `n_components = 6`, we capture approximately 94.6% information (@fig-pca). It helps reduce a cost of computation of the model.

#figure(
  image("corr.png"),
  caption: [Correlation between features],
)<fig-corr>

#figure(
  image("pca_explained.png"),
  caption: [PCA Explained Variance],
)<fig-pca>

== Clustering

According to @yinRapidReviewClustering2024 @waniComprehensiveAnalysisClustering2024, clustering algorithms can be categorized into seven buckets: (1) Partition-based clustering, (2) Hierarchical clustering, (3) Density-based clustering, (4) Grid-based clustering, (5) Model-based clustering, (6) Graph-based clustering and (7) Deep Learning-based clustering. These models serve in different situations and business based on their characteristics.

To choose suitable algorithms, we look at the structure of the dataset. As shown in @fig-distribution, the dataset contains bimodal variables (temperature), heavily skewed variables (precipitation, snow, wind, gust). We can conclude that the climate data is not spherical, not Gaussian and does not have uniform cluster density.

Here we choose two algorithms for clustering task: K-Means from partition-based approach and HDBSCAN from density-based clustering category. K-Means is included as our baseline due to its simpleness in implementation and commonly used. HDBSCAN @campelloDensityBasedClusteringBased2013 is an algorithm that group points based on the density of the surrounding region. Extending from DBSCAN, it builds a hierarchy of density-based clusters and choose the most stable clusters out of them.

= Task 2: Image Processing with Deep Neural Networks (DNNs)
The purpose of this task is to effectively, thoroughly apply pre-trained DNNs for image clustering and classification tasks. In this task, we choose two datasets, i.e. Kaggle: Cats vs Dogs Dataset @catsVsDogs and Food101 @food101. Kaggle: Cats vs Dogs Dataset is a dataset providing a collection of photos about cats and dogs. Statistically, there are 12491 pictures of cat and 12470 images of dog. Food101 includes images of food. With over 100000 files, it is a good set of data for image clustering and classification training.

== Dataset 1: Food101
=== Data preprocessing

*Image Cleaning* As shown in @fig-food101-corrupt_size, all images are in good quality with various sizes. Therefore we need to resize these images for the sake of consistent input for aftward downstream analysis and model. Here we resize all images to $256 times 256$ dimension and save in `image_resized` folder.

*Noise Removal* Apply Gaussian filter to reduce noise in images.

*Normalization* Normalization is a process that rescale pixel values to a specific range (usually 0-255 to [0-1] range).

#figure(
  image("food101-image-preprocessed-samples.png"),
  caption: [Image of some preprocessed samples],
)<fig-food101-image-preprocessed-samples>

=== Clustering


#bibliography("refs.bib")

#pagebreak()

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
  image("lof.png"),
  caption: [Local Outlier Factor (LOF)],
)<fig-mod-zscore>

#figure(
  image("outlier_treatment.png"),
  caption: [Capping extreme values],
)<fig-outlier-treatment>

#figure(
  image("feature_scaling_code.png"),
  caption: [Feature Scaling],
)<fig-feature_scaling>

#figure(
  image("remove_high_corr.png"),
  caption: [Remove high correlated columns],
)<fig-remove-high-corr>

#figure(
  image("food101-corrupt_size.png"),
  caption: "Code and result for detecting corrupted images and sizes in Food101 dataset.",
)<fig-food101-corrupt_size>
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

