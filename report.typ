#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/algorithmic:1.0.6"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

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
  image("images/feature_scaling.png"),
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

We use Principal Component Analysis (PCA) for dimensionality reduction. With `n_components = 4`, we capture approximately 90% information (@fig-pca). It helps reduce a cost of computation of the model.

#figure(
  image("images/corr.png"),
  caption: [Correlation between features],
)<fig-corr>

#figure(
  image("images/pca_explained.png"),
  caption: [PCA Explained Variance],
)<fig-pca>

== Clustering

According to @yinRapidReviewClustering2024 @waniComprehensiveAnalysisClustering2024, clustering algorithms can be categorized into seven buckets: (1) Partition-based clustering, (2) Hierarchical clustering, (3) Density-based clustering, (4) Grid-based clustering, (5) Model-based clustering, (6) Graph-based clustering and (7) Deep Learning-based clustering. These models serve in different situations and business based on their characteristics.

To choose suitable algorithms, we look at the structure of the dataset. As shown in @fig-distribution, the dataset contains bimodal variables (temperature), heavily skewed variables (precipitation, snow, wind, gust). We can conclude that the climate data is not spherical, not Gaussian and does not have uniform cluster density.

Here we choose three algorithms for clustering task: K-Means from partition-based approach, HDBSCAN from density-based clustering category and GMM from model-based approach.

*K-Means*   K-Means @lloyd1982least @macqueen1967multivariate is an unsupervised, partition-based clustering algorithm that divides data into $K$ groups based on similarity. It is one of the simplest, fastest, and most widely used in real world. The main idea of K-Means is to find K cluster centers (centroid) such that each data point belongs to the cluster with closest centroid by minizing the Within-Cluster Sum of Squares (WCSS) which is calculated by summing the squared Euclidean distance between each data point ($x_i$) and the centroid ($mu_i$) of the assign cluster ($C_k$).
$
  "WCSS" = sum_i^n min_(mu_j in C_k) norm(x_i - mu_j)_2
$

In this task, we run K-Means clustering with multiple options of $k$ ranging from 2 to 10 and use Silhouette Score to select the optimal number of clusters. As shown in @fig-kmeans-elbow-curve, the analysis reveals that $k = 4$ provides the best balance between cluster cohesion and separation, achieving a Silhouette Score of 0.4890. This suggests that the climate data naturally divides into four seasonal patterns, potentially representing spring, summer, autumn, and winter weather conditions.

#figure(
  image("images/kmeans_analysis.png"),
  caption: [K-Means elbow curve],
)<fig-kmeans-elbow-curve>

#figure(
  image("images/kmeans_clustering_visualization.png"),
  caption: [K-Means Clustering Visualization],
)<fig-kmeans-visualization>


*HDBSCAN* HDBSCAN @campelloDensityBasedClusteringBased2013 is an algorithm that groups points based on the density of the surrounding region. Extending from DBSCAN, it builds a hierarchy of density-based clusters and selects the most stable clusters from them. Unlike K-Means, HDBSCAN does not require specifying the number of clusters in advance and can automatically identify outliers as noise points. The algorithm works by computing a minimum spanning tree of the mutual reachability distance, then extracting a hierarchy of clusters based on density thresholds, and finally selecting the most persistent clusters using a stability measure. In this task, we test different `min_cluster_size` parameters (10, 15, 20, 30, 50) and select the configuration with the highest Silhouette Score. The best configuration uses `min_cluster_size=10`, identifying 2 main clusters and 275 noise points (15.60% of the data).

#figure(
  image("images/hdbscan_clustering_visualization.png"),
  caption: [HDBSCAN Clustering Visualization],
)<fig-hdbscan-visualization>

*GMM* Gaussian Mixture Model @reynoldsGMM2009 is a probabilistic, model-based clustering algorithm that assumes data is generated from a mixture of multiple Gaussian distributions. Unlike K-Means which assigns each point to exactly one cluster (hard assignment), GMM provides soft assignments by computing the probability that each point belongs to each cluster. This is particularly useful when cluster boundaries are uncertain or when points may belong to multiple clusters. The algorithm uses the Expectation-Maximization (EM) algorithm to iteratively estimate the parameters (means, covariances, and mixture weights) of the Gaussian components. For cluster $k$, the probability that a data point $bold(x)$ belongs to that cluster is:
$
  P(k | bold(x)) = frac(pi_k cal(N)(bold(x) | bold(mu)_k, bold(Sigma)_k), sum_(j=1)^K pi_j cal(N)(bold(x) | bold(mu)_j, bold(Sigma)_j))
$
where $pi_k$ is the mixing coefficient, $bold(mu)_k$ is the mean vector, and $bold(Sigma)_k$ is the covariance matrix. We use `covariance_type='full'` to allow elliptical cluster shapes, which is more appropriate for climate data than spherical clusters. Model selection is performed using Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC), with lower values indicating better models. As shown in @fig-gmm-analysis, the best configuration uses 2 components.

#figure(
  image("images/gmm_analysis.png"),
  caption: [GMM model selection using BIC, AIC, and Silhouette Score],
)<fig-gmm-analysis>

#figure(
  image("images/gmm_clustering_visualization.png"),
  caption: [GMM Clustering Visualization with Gaussian ellipses showing the probabilistic cluster boundaries],
)<fig-gmm-visualization>

=== Evaluation and Comparison

To evaluate the performance of the three clustering algorithms, we use three widely-adopted metrics: Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index. The Silhouette Score measures how similar a point is to its own cluster compared to other clusters, with values ranging from -1 to 1 (higher is better). The Davies-Bouldin Index evaluates the average similarity between each cluster and its most similar cluster, where lower values indicate better separation. The Calinski-Harabasz Index (also known as Variance Ratio Criterion) measures the ratio of between-cluster dispersion to within-cluster dispersion, with higher values indicating better-defined clusters.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header([*Metric*], [*K-Means*], [*HDBSCAN*], [*GMM*]),
    [Number of Clusters], [4], [2], [3],
    [Silhouette Score], [0.4890], [0.5704], [0.1909],
    [Davies-Bouldin Index], [0.8551], [0.4255], [1.6062],
    [Calinski-Harabasz Score], [982.21], [85.45], [490.63],
  ),
  caption: [Comparison of clustering algorithms performance on Basel climate dataset],
)<fig-comparison-table>

As shown in @fig-comparison-table, HDBSCAN achieved the best overall performance with the highest Silhouette Score (0.5704) and the lowest Davies-Bouldin Index (0.4255), indicating superior cluster quality and separation. HDBSCAN identified 2 main clusters with 275 noise points (15.60% of the data), suggesting the presence of two dominant climate patterns (summer and winter) along with transitional or extreme weather events . The noise points represent anomalous weather conditions that HDBSCAN appropriately separates rather than forcing into existing clusters, demonstrating the algorithm's robustness to outliers. K-Means, configured with 4 clusters based on silhouette analysis, achieved a moderate Silhouette Score (0.4890) and the highest Calinski-Harabasz Score (982.21), indicating well-defined cluster centers. However, its Davies-Bouldin score (0.8551) was higher than HDBSCAN, suggesting more overlap between clusters. This is expected given K-Means' assumption of spherical clusters and its sensitivity to the bimodal and skewed distributions present in climate data. GMM, with 3 components, performed poorly compared to the other algorithms, achieving the lowest Silhouette Score (0.1909) and the highest Davies-Bouldin Index (1.6062), indicating poorly separated clusters with significant overlap.

The choice of algorithm depends on the specific analytical goals. HDBSCAN is the most suitable for this climate dataset, effectively identifying the two main seasonal patterns while robustly handling outliers and extreme weather events. The results suggest that the Basel climate data exhibits two primary seasonal clusters with a substantial number of transitional or anomalous days, validating the use of density-based clustering approaches for meteorological data analysis.

= Task 2: Image Processing with Deep Neural Networks (DNNs)
The purpose of this task is to effectively, thoroughly apply pre-trained DNNs for image clustering and classification tasks. In this task, we choose two datasets, i.e. Kaggle: Cats vs Dogs Dataset @catsVsDogs and Food101 @food101. Kaggle: Cats vs Dogs Dataset is a dataset providing a collection of photos about cats and dogs. Statistically, there are 12491 pictures of cat and 12470 images of dog. Food101 includes images of food. With over 100000 files, it is a good set of data for image clustering and classification training.

== Dataset 1: Food101
=== Data preprocessing

*Image Cleaning* As shown in @fig-food101-corrupt_size, all images are in good quality with various sizes. Therefore we need to resize these images for the sake of consistent input for aftward downstream analysis and model. Here we resize all images to $256 times 256$ dimension and save in `image_resized` folder.

*Noise Removal* Apply Gaussian filter to reduce noise in images.

*Normalization* Normalization is a process that rescale pixel values to a specific range (usually 0-255 to [0-1] range).


=== Clustering


#bibliography("refs.bib")

#pagebreak()

#show: appendix
= Code <app-code>
#figure(
  image("images/basel_missing_value.png"),
  caption: [Code and result of detecting missing value.],
)<missing-val>

#figure(
  image("images/mod_zscore.png"),
  caption: [Modified Z-score code and result],
)<fig-mod-zscore>

#figure(
  image("images/lof.png"),
  caption: [Local Outlier Factor (LOF)],
)<fig-mod-zscore>

#figure(
  image("images/outlier_treatment.png"),
  caption: [Capping extreme values],
)<fig-outlier-treatment>

#figure(
  image("images/feature_scaling_code.png"),
  caption: [Feature Scaling],
)<fig-feature_scaling>

#figure(
  image("images/remove_high_corr.png"),
  caption: [Remove high correlated columns],
)<fig-remove-high-corr>

#figure(
  image("images/food101-corrupt_size.png"),
  caption: "Code and result for detecting corrupted images and sizes in Food101 dataset.",
)<fig-food101-corrupt_size>
= Tables, Figures and Algorithms <app-table>


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
  image("images/dist_all_features.png"),
  caption: "Distribution of all features",
)<fig-distribution>


