# Installation

[conda](https://anaconda.org/anaconda/conda)

```bash
conda env create -f env.yaml
conda activate scc451
```

# Download dataset

## [Cats vs. Dogs](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)

```bash
# Download the dataset
curl -L -o data/task2/cats_dogs.zip \
  https://www.kaggle.com/api/v1/datasets/download/karakaggle/kaggle-cat-vs-dog-dataset

# Extract the dataset
unzip data/task2/cats_dogs.zip -d data/task2/cats_dogs
```

## [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101)

```bash
# Download the dataset
curl -L -o data/task2/food-101.zip \
  https://www.kaggle.com/api/v1/datasets/download/dansbecker/food-101

# Extract the dataset
unzip data/task2/food-101.zip -d data/task2/food-101
```
