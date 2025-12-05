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

= Task 1: Basel Climate Dataset<task1>
The Basel Climate Dataset (ClimateDataBasel.csv) is a subset of publicly available weather data from Basel, Switzerland, obtained by Meteoblue #footnote[https://www.meteoblue.com/en/weather/week/galgate_united-kingdom_2648924]. It contains 1,763 records with 18 features collected during the summer and winter seasons from 2010 to 2019.

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
    table.header([*Method*], [*Advantage*], [*Disadvantages*]),
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
  table(
    columns: (auto, auto, auto),
    table.header([*Method / Technique*], [*Advantages (Strengths)*], [*Disadvantages (Limitations)*]),

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

#figure(
  image("images/corr.png"),
  caption: [Correlation between features],
)<fig-corr>

#figure(
  image("images/pca_explained.png"),
  caption: [PCA Explained Variance],
)<fig-pca>

#figure(
  image("images/feature_scaling.png"),
  caption: [the differences between before and after scaling. Before scaling, the data distributions are varied among features but after applying Robust Scaling method, we can see mostly distributions now look similar.],
)<fig-befor-after-scaling>

We use Principal Component Analysis (PCA) for dimensionality reduction. With `n_components = 4`, we capture approximately 90% information (@fig-pca). It helps reduce a cost of computation of the model.


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

To evaluate the performance of the three clustering algorithms, we use three widely-adopted metrics: Silhouette Score, Davies-Bouldin Index. The Silhouette Score measures how similar a point is to its own cluster compared to other clusters, with values ranging from -1 to 1 (higher is better). The Davies-Bouldin Index evaluates the average similarity between each cluster and its most similar cluster, where lower values indicate better separation.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header([*Metric*], [*K-Means*], [*HDBSCAN*], [*GMM*]),
    [Number of Clusters], [4], [2], [3],
    [Silhouette Score], [0.4890], [0.5704], [0.1909],
    [Davies-Bouldin Index], [0.8551], [0.4255], [1.6062],
  ),
  caption: [Comparison of clustering algorithms performance on Basel climate dataset],
)<fig-comparison-table>

As shown in @fig-comparison-table, HDBSCAN achieved the best overall performance with the highest Silhouette Score (0.5704) and the lowest Davies-Bouldin Index (0.4255), indicating outperforming cluster quality and separation. HDBSCAN identified 2 main clusters with 275 noise points (15.6% of the data), suggesting the presence of two climate patterns (summer and winter) along with transitional or extreme weather events. K-Means, configured with 4 clusters based on silhouette analysis, achieved a moderate Silhouette Score (0.4890). However, its Davies-Bouldin score (0.8551) was higher than HDBSCAN, suggesting more overlap between clusters. This is expected given K-Means' assumption of spherical clusters and its sensitivity to the bimodal and skewed distributions present in climate data. GMM, with 3 components, performed poorly compared to the other algorithms, achieving the lowest Silhouette Score (0.1909) and the highest Davies-Bouldin Index (1.6062), indicating poorly separated clusters with significant overlap.

HDBSCAN is the most suitable for this climate dataset, effectively identifying the two main seasonal patterns while robustly handling outliers and extreme weather events. The results suggest that the Basel climate data exhibits two primary seasonal clusters with a substantial number of transitional or anomalous days, which is verified by the fact that the dataset was collected in winters and summers.

= Task 2: Image Processing with Deep Neural Networks (DNNs)
The purpose of this task is to effectively, thoroughly apply pre-trained DNNs for image clustering and classification tasks. In this task, we choose two datasets, i.e. Kaggle: Cats vs Dogs Dataset @catsVsDogs and Food101 @food101. Kaggle: Cats vs Dogs Dataset is a dataset providing a collection of photos about cats and dogs. Statistically, there are 12491 pictures of cat and 12470 images of dog. Food101 includes images of food. With over 100000 files, it is a good set of data for image clustering and classification training. We also choose DINOV2 @oquab2023dinov2 (dinov2_vits14 model via Pytorch Hub) and DenseNet @Huang2016DenselyCC (DenseNet-121 model pre-trained on ImageNet) as our DNN models. DINOV2, developed by Meta AI Research (or FAIR), is a self-supervised vision model trained using self-distillation and self-supervised learning which means it can learn from unlabeled images. DenseNet is a CNN architecture, well-known for its densen connectivity pattern, where each layers receives the feature maps of all previous layers as input.

#figure(
  image("images/densenet_architecture.png"),
  caption: [Illutration of DenseNet architecture. Image taken from @sujawat2022.],
)<fig-densenet-architecture>

== Data Preprocessing
In this project, we resizes images to $256 times 256$ using interpolation mode Bilinear, followed by a center crop of $224$. After that, they will be normalized using $"mean" = [0.485, 0.456, 0.406]$ and $"std" = [0.229, 0.224, 0.225]$. These values are industry-standard and implemented in torchvision's preset for classification. #footnote[https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69]

== Feature Extraction<feature-extraction>
Image feature extraction is a process of converting an image into a set of meaningful numerical features that capture important visual information such as shapes, textures, colors. In this task, we use deep learning-based model for extracting features. Deep learning-based feature extraction. Deep neural networks (i.e. CNN and Transformers) is suitable for this step because of their ability to learn hierarchical, sementically rich representations directly from pixels. A DNN informative outputs are essential and useful for downstream processes like clustering or classification.

As you can see in @fig-densenet-architecture, the last layer of DenseNet-121 is the classification layer. Thus, to use the model for feature extraction, we need to drop the last layer. With DINOV2, we can use it as normal. After the process of extracting feature is done, we save all results into `.npy` files for later use.

The same as @task1, we use PCA technique for dimensionality reduction. We retain 95% of the variance, reducing the Food101 feature vectors from 1024 to 14 dimensions with DenseNet-121 and from 384 to 202 dimensions with DINOv2. Regarding Cats vs Dogs dataset, we reduce feature vectors from 1024 to 14 dimensions with DenseNet-121 and from 384 to 197 dimensions with DINOV2 (shown in @fig-task2-dim-reduction).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, center, center),
    table.header(
      [*Dataset - Model*], [*Original Dimensions*], [*Reduced Dimensions*],
      [Food101 - DenseNet-121], [1024], [14],
      [Food101 - DINOV2], [384], [202],
      [Cats vs Dogs - DenseNet-121], [1024], [14],
      [Cats vs Dogs - DINOV2], [384], [197],
    ),
  ),
  caption: [Before and after reduction dimensions of two datasets with two models],
)<fig-task2-dim-reduction>

== Clustering
In clustering task, we use K-Means clustering method. $k=101$ and $k=2$ are used for Food101 dataset and Cats vs. Dgos dataset, respectively. @fig-task2-cluster-viz shows all clusters in 2D with 400 samples. @fig-task2-cluster-score shows all related scores of K-Means cluster method.
#figure(
  image("images/clustering_all_2x2_grid.png"),
  caption: [K-Means Cluster of all combinations of datasets and models. We use 400 samples for visualization.],
)<fig-task2-cluster-viz>

#figure(
  table(
    columns: (90pt, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header(
      [*Dataset - Model*], [*k*], [*Silhouette score*], [*Davies-Bouldin Index*], [*Calinski-Harabasz Score*],
      [Food101 - DenseNet-121], [101], [0.0986], [1.6199], [5728.69],
      [Food101 - DINOV2], [101], [0.0904], [2.6753], [888.37],
      [Cats vs. Dogs - DenseNet-121], [2], [0.3468], [1.0796], [18609.98],
      [Cats vs. Dogs - DINOV2], [2], [0.0973], [3.0443], [2671.85],
    ),
  ),
  caption: [Silhouette score, Davies-Bouldin index and Calinski-Harabasz score of all K-Means clusters.],
)<fig-task2-cluster-score>

@fig-task2-cluster-viz shows how different feature extractors have different clustering representations with the same dataset. The cluster of the DenseNet-121 model with Food101 dataset (shown at top-left) forms a cloud with no obvious structure. There is no tight groups suggesting features are not clearly seperating categories. It can be explained as DenseNet-121 model is trained for ImageNet classification so it may not work well to this dataset. With the same dataset, however, DINOV2 model does slightly better (shown in top-right). Although data points are still overlapping but the structure is more compact with several local groupings existing and the distribution is less noisy. The bottom-left visualization of Cats vs. Dogs dataset with DenseNet-121 model still shows noisy display. The clusters is heavily overlapping. However, the DINOV2 model performs significantly good with this dataset. The two clusters are clearly separated with clean left and right grouping. There is no overlap between clusters and there is no noise too.

By these visualizations, DINOV2 performs better than DenseNet-121 in all cases.

== Classification
In this task, we add a linear fine-tunable layer to the DNN used in @feature-extraction. This layer is added to perform the classification task. The goal of this section is to scrutinize the performance of the classification layer including all classification metrics i.e. precision/recall, F1 score and accuracy. Moreover, we also measure its computational complexity on different machines (see @fig-spec-machine).

The linear layer is a multilayer perceptron (MLP) with input layer equals to the dimension of extracted features connecting to hidden layer of size 500 with ReLU activation function. Utimately, it returns the probabilites of all classes via Softmax function (see detailed implementation in @fig-linear-classifer). We train the model with a batch of size 128 over 200 epochs.

#figure(
  table(
    columns: (90pt, auto, auto, auto),
    align: (center, center, center, center),
    table.header(
      [*Dataset - Model*], [*Accuracy*], [*Precision*], [*F1*],
      [Food101 - DenseNet-121], [0.017], [0.0004], [0.0008],
      [Food101 - DINOV2], [0.0197], [0.0004], [0.0008],
      [Cats vs. Dogs - DenseNet-121], [0.6593], [0.0.6593], [0.6592],
      [Cats vs. Dogs - DINOV2], [0.998], [0.998], [0.998],
    ),
  ),
  caption: [Accuracy, Precision and F1 score of all datasets and models with Linear Classifier.],
)<fig-task2-classifier-linear-score>

As shown in @fig-task2-classifier-linear-score, linear classifier performs poorly with Food101 dataset whilst significantly better with Cats vs. Dogs dataset. It can be explained that the first dataset includes multiple classes which a linear layer cannot clearly separate. The second dataset has only two classes which is a ideal case for linear classifier.

Thus, we decide to apply Support Vector Machines (SVM) algorithm for the Food101 dataset. SVM is suitable because it can handle multiclass classification. SVM creates a hyperplane in multidimensional space to split different classes then tries to maximize the margin of it that best clusters the dataset. We implement the algorithm with _rbf_ kernel and `C=1.0`.

@fig-task2-classifier-svm-score show the scores of classification using SVM algorithm. While model DenseNet-121 has poor result in all scores (0.1162, 0.1003, 0.0973 for accuracy, precision, F1 score, respectively), DINOV2 gets all high scores, mostly over 0.8 (0.8640 for accuracy, 0.8655 for precision, 0.8643 for F1). It indicates that with the same classifer, it is possible that we get different results. The scores depend on the feature extraction methods we use. In this project, we have observed outperforming performance of DINOV2 model in feature extraction task compared to DenseNet-121 model.

#figure(
  table(
    columns: (90pt, auto, auto, auto),
    align: (center, center, center, center),
    table.header(
      [*Model*], [*Accuracy*], [*Precision*], [*F1*],
      [DenseNet-121], [0.1162], [0.1003], [0.0973],
      [DINOV2], [0.8640], [0.8655], [0.8643],
    ),
  ),
  caption: [Accuracy, Precision and F1 score of Food101 dataset and two models with SVM algorithm.],
)<fig-task2-classifier-svm-score>

#pagebreak()

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
  image("images/image_preprocessing.png"),
  caption: "Model import and images preprocessing using model transform.",
)<fig-image-preprocessing>

#figure(
  image("images/linear_classifier.png"),
  caption: "Implementation of Linear Classifier",
)<fig-linear-classifer>

#figure(
  image("images/f_densenet_linear.png"),
  caption: "Classification scores of dataset Food101 and model DenseNet-121 with Linear Layer Classifier",
)<f_densenet_linear>

#figure(
  image("images/f_dino_linear.png"),
  caption: "Classification scores of dataset Food101 and model DINOV2 with Linear Layer Classifier",
)<f_dino_linear>

#figure(
  image("images/cd_densenet_linear.png"),
  caption: "Classification scores of dataset Cats vs. Dogs and model DenseNet-121 with Linear Layer Classifier",
)<cd_densenet_linear>

#figure(
  image("images/cd_dino_linear.png"),
  caption: "Classification scores of dataset Cats vs Dogs and model DINOV2 with Linear Layer Classifier",
)<cd_dino_linear>

#figure(
  image("images/f_densenet_svm.png"),
  caption: "Classification scores of dataset Food101 and model DenseNet-121 with SVM",
)<f_densenet_svm>

#figure(
  image("images/f_dino_svm.png"),
  caption: "Classification scores of dataset Food101 and model DINOV2 with SVM",
)<f_dino_svm>
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
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Name*], [*RAM*], [*GPU*], [*Torch backend*]),
    [Apple Macbook Pro M4 Max], [32GB], [32-core built-in GPU], [mps],
    [LU InfoLab workstatation], [30GB], [NVIDIA A2000 12GB], [cuda],
  ),
  caption: [Specifications of machines used in classification subtask in task 2.],
)<fig-spec-machine>

#figure(
  image("images/dist_all_features.png"),
  caption: "Distribution of all features",
)<fig-distribution>


