# assignment1.py

import os
import time
import gzip
import csv
from collections import defaultdict
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import tensorflow as tf

from libreco.algorithms import Transformer, NGCF
from libreco.data import DatasetPure, random_split
from libreco.evaluation import evaluate

# Set random seeds for reproducibility
random_seed = 2
np.random.seed(random_seed)
random.seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)

# Suppress warnings
warnings.filterwarnings("ignore")

def readCSV(path):
    """
    Generator function to read CSV files compressed with gzip.
    """
    with gzip.open(path, 'rt') as f:
        f.readline()  # Skip header
        for l in f:
            yield l.strip().split(',')

############################################
# Rating Prediction Task using Latent Factor Model with Biases
############################################

print("Rating Prediction Task using Latent Factor Model with Biases")

# Read training data and build user-item interactions
print("Reading training data...")

user2id = {}
item2id = {}
uid = 0
iid = 0

ratings = []
for user, item, r in readCSV("train_Interactions.csv.gz"):
    r = int(r) + 1  # Adjust rating scale to start from 1
    if user not in user2id:
        user2id[user] = uid
        uid += 1
    if item not in item2id:
        item2id[item] = iid
        iid += 1
    ratings.append((user2id[user], item2id[item], r))

num_users = len(user2id)
num_items = len(item2id)

print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")

# Initialize parameters
print("Initializing parameters...")

# Global average rating
alpha = np.mean([r for _, _, r in ratings])

# User biases
beta_u = np.zeros(num_users)

# Item biases
beta_i = np.zeros(num_items)

# Latent factors
K = 20  # Number of latent factors

init_scale = 1e-6
gamma_u = np.random.uniform(-0.5 * init_scale, 0.5 * init_scale, (num_users, K))
gamma_i = np.random.uniform(-0.5 * init_scale, 0.5 * init_scale, (num_items, K))

# Training parameters
learning_rate = 0.01
regularization = 0.21
num_epochs = 20

# Training loop using Stochastic Gradient Descent
print("Training Latent Factor Model...")

for epoch in range(num_epochs):
    random.shuffle(ratings)
    total_loss = 0
    for u_idx, i_idx, r_ui in ratings:
        # Predict the rating
        pred = alpha + beta_u[u_idx] + beta_i[i_idx] + np.dot(gamma_u[u_idx], gamma_i[i_idx])
        error = r_ui - pred

        # Update parameters
        beta_u[u_idx] += learning_rate * (error - regularization * beta_u[u_idx])
        beta_i[i_idx] += learning_rate * (error - regularization * beta_i[i_idx])
        gamma_u[u_idx] += learning_rate * (error * gamma_i[i_idx] - regularization * gamma_u[u_idx])
        gamma_i[i_idx] += learning_rate * (error * gamma_u[u_idx] - regularization * gamma_i[i_idx])

        # Accumulate loss
        total_loss += error ** 2 + regularization * (
            beta_u[u_idx] ** 2 + beta_i[i_idx] ** 2 +
            np.linalg.norm(gamma_u[u_idx]) ** 2 + np.linalg.norm(gamma_i[i_idx]) ** 2
        )

    # Update alpha after all parameter updates in the epoch
    sum_residuals = 0
    for u_idx, i_idx, r_ui in ratings:
        sum_residuals += (r_ui - beta_u[u_idx] - beta_i[i_idx] - np.dot(gamma_u[u_idx], gamma_i[i_idx]))
    alpha = sum_residuals / len(ratings)

    rmse = math.sqrt(total_loss / len(ratings))
    print(f"Epoch {epoch+1}/{num_epochs}, RMSE: {rmse:.6f}, Alpha: {alpha:.6f}")

print("Training completed.")

# Prepare to make predictions
print("Preparing to make predictions...")

id2user = {v: k for k, v in user2id.items()}
id2item = {v: k for k, v in item2id.items()}

# Predict ratings for test pairs
print("Predicting ratings for test data...")
with open("predictions_Rating_LFM.csv", 'w') as predictions:
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            # Header
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        if u in user2id and b in item2id:
            u_idx = user2id[u]
            i_idx = item2id[b]
            pred = alpha + beta_u[u_idx] + beta_i[i_idx] + np.dot(gamma_u[u_idx], gamma_i[i_idx])
        elif u in user2id:
            u_idx = user2id[u]
            pred = alpha + beta_u[u_idx]
        elif b in item2id:
            i_idx = item2id[b]
            pred = alpha + beta_i[i_idx]
        else:
            pred = alpha

        pred = min(6, max(1, pred)) - 1  # Adjust back to original rating scale (0 to 5)
        predictions.write(f"{u},{b},{pred}\n")

############################################
# Rating Prediction Task using Transformer Model
############################################

print("\nRating Prediction Task using Transformer Model")

# Load training data into a pandas DataFrame
print("Loading training data into DataFrame...")
train_data = pd.DataFrame(
    list(readCSV("train_Interactions.csv.gz")),
    columns=["userID", "bookID", "rating"]
)
train_data["rating"] = train_data["rating"].astype(float) + 1  # Adjust rating scale

# Rename columns to match LibRecommender's expectations
train_data.rename(columns={"userID": "user", "bookID": "item", "rating": "label"}, inplace=True)

print("Building trainset and data_info for training...")
train_set, data_info = DatasetPure.build_trainset(train_data)  # Use all data for training
print(data_info)

# Define evaluation metrics
metrics_rating = ["rmse", "mae", "r2"]

# Read pairs for Rating Prediction
print("Loading Rating Prediction pairs...")
pairs_rating = pd.read_csv("pairs_Rating.csv")
pairs_rating.rename(columns={"userID": "user", "bookID": "item"}, inplace=True)

# Define the Transformer model parameters - Obtained via Hyperparameter Optimization using Optuna
transformer_params = {
    "task": "rating",
    "data_info": data_info,
    "embed_size": 512,
    "n_epochs": 3,
    "lr": 0.0003392308111583879,
    "lr_decay": True,
    "epsilon": 1e-05,
    "reg": 8.303908209260279e-05,
    "batch_size": 256,
    "sampler": 'random',
    "use_bn": False,
    "dropout_rate": 0.10142529816569815,
    "hidden_units": (512, 256, 128),
    "recent_num": 10,
    "random_num": None,
    "num_heads": 16,
    "num_tfm_layers": 6,
    "multi_sparse_combiner": 'sqrtn',
    "seed": 153,
    "lower_upper_bound": (1, 7),
    "tf_sess_config": None,
}

print("\nTraining Transformer model...")

# Reset TensorFlow graph
tf.compat.v1.reset_default_graph()

# Initialize the Transformer model with specified parameters
model = Transformer(**transformer_params)

# Fit the model
model.fit(
    train_set,
    neg_sampling=False,
    verbose=2,
    metrics=metrics_rating,
)

# Make predictions for Rating Prediction Task
print("Making rating predictions with Transformer...")
predictions_rating = model.predict(
    user=pairs_rating["user"].tolist(),
    item=pairs_rating["item"].tolist(),
    cold_start='popular'
)

# Save predictions to CSV
output_path_rating = "predictions_Rating_Transformer.csv"
print(f"Saving Transformer Rating predictions to {output_path_rating}...")
pairs_rating_copy = pairs_rating.copy()
pairs_rating_copy["prediction"] = pd.Series(predictions_rating).apply(lambda pred: min(6, max(1, pred)) - 1)
pairs_rating_copy.rename(columns={"user": "userID", "item": "bookID"}).to_csv(output_path_rating, index=False)

############################################
# Combine LFM and Transformer Predictions
############################################

print("\nCombining LFM and Transformer predictions...")

# Define file paths
file_path_lfm = "predictions_Rating_LFM.csv"
file_path_transformer = "predictions_Rating_Transformer.csv"

# Load predictions
df_lfm = pd.read_csv(file_path_lfm)
df_transformer = pd.read_csv(file_path_transformer)

# Merge on userID and bookID
merged_df = pd.merge(df_lfm, df_transformer, on=['userID', 'bookID'], suffixes=('_lfm', '_transformer'))

# Calculate weighted average prediction
merged_df['prediction'] = (0.2 * merged_df['prediction_lfm'] + 0.8 * merged_df['prediction_transformer'])

# Select relevant columns and save to a new CSV file
output_df = merged_df[['userID', 'bookID', 'prediction']]
output_df.to_csv("predictions_Rating.csv", index=False)

print("Combined predictions saved to 'predictions_Rating.csv'.")

############################################
# Read Prediction Task via an Ensemble of Various Methods
############################################

import os
import time
import gzip
import csv
from collections import defaultdict
import random
import math
import pandas as pd
from pathlib import Path
import warnings
import tensorflow as tf
import numpy as np
import torch

# Import all the models
from libreco.algorithms import (
    BPR,
    LightGCN,
    RNN4Rec,
    NGCF,
    GraphSage,
)
from libreco.data import DatasetPure, random_split
from libreco.evaluation import evaluate


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    # Resetting PyTorch seed if needed
    try:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
    print("\n", "=" * 30, name, "=" * 30)


start_time = time.perf_counter()

# Suppress warnings from LibRecommender and other libraries
warnings.filterwarnings("ignore")


# Function to read CSV files (assuming gzip compressed)
def readCSV(path):
    with gzip.open(path, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            yield line.strip().split(',')


# Read training data into a pandas DataFrame with correct column names
print("Loading training data into DataFrame...")
train_data = pd.DataFrame(
    list(readCSV("train_Interactions.csv.gz")),
    columns=["userID", "bookID", "rating"]
)
train_data["rating"] = 1.0

# Rename columns to match LibRecommender's expectations
train_data.rename(columns={"userID": "user", "bookID": "item", "rating": "label"}, inplace=True)

print("Building trainset and data_info for training...")
train_set, data_info = DatasetPure.build_trainset(train_data)  # Use all data for training
print(data_info)

# Define evaluation metrics for ranking
metrics_ranking = ["precision", "recall", "ndcg", "map"]

# Read pairs for Read Prediction Task
print("Loading Read Prediction pairs...")
pairs_read = pd.read_csv("pairs_Read.csv")
pairs_read.rename(columns={"userID": "user", "bookID": "item"}, inplace=True)

# Ensure 'LibRecommender' directory exists
Path("LibRecommender").mkdir(parents=True, exist_ok=True)

############################################
# List of Models to Train and Evaluate
############################################
models_to_train = [
    # GraphSage
    {
        "name": "GraphSage",
        "class": GraphSage,
        "params": {
            "task": "ranking",
            "data_info": data_info,
            "loss_type": "bpr",
            "embed_size": 128,
            "num_layers": 2,
            "sampler": "popular",
            "device": "cuda",
            "seed": 42,
        },
        "metrics": metrics_ranking,
    },
    # LightGCN
    {
        "name": "LightGCN",
        "class": LightGCN,
        "params": {
            "task": "ranking",
            "data_info": data_info,
            "loss_type": "bpr",
            "embed_size": 128,
            "n_layers": 3,
            "sampler": "unconsumed",
            "device": "cuda",
            "seed": 42,
        },
        "metrics": metrics_ranking,
    },
    # NGCF
    {
        "name": "NGCF",
        "class": NGCF,
        "params": {
            "task": "ranking",
            "data_info": data_info,
            "loss_type": "bpr",
            "embed_size": 128,
            "n_epochs": 50,
            "lr": 0.001,
            "lr_decay": False,
            "reg": 1e-4,
            "batch_size": 1024,
            "num_neg": 4,
            "node_dropout": 0.1,
            "message_dropout": 0.1,
            "hidden_units": (64, 64, 64),
            "sampler": "unconsumed",
            "seed": 42,
            "device": "cuda",
        },
        "metrics": metrics_ranking,
    },
    # RNN4Rec
    {
        "name": "RNN4Rec",
        "class": RNN4Rec,
        "params": {
            "task": "ranking",
            "data_info": data_info,
            "loss_type": "bpr",
            "embed_size": 128,
            "hidden_units": 16,
            "sampler": "unconsumed",
            "recent_num": 10,
            "seed": 42,
        },
        "metrics": metrics_ranking,
    },
    # BPR
    {
        "name": "BPR",
        "class": BPR,
        "params": {
            "task": "ranking",
            "data_info": data_info,
            "embed_size": 128,
            "use_tf": True,
            "sampler": "unconsumed",
            "seed": 42,
        },
        "metrics": metrics_ranking,
    }
]

############################################
# Iterate Through Each Model
############################################
for model_info in models_to_train:
    model_name = model_info["name"]
    model_class = model_info["class"]
    model_params = model_info["params"]
    model_metrics = model_info.get("metrics", metrics_ranking)

    print(f"\nProcessing model: {model_name}")

    # Create directory for the model
    model_dir = Path("LibRecommender") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Reset state
        reset_state(model_name)

        # Initialize the model with specified parameters
        model = model_class(**model_params)

        # Define neg_sampling for ranking task
        neg_sampling = True

        # Fit the model
        model.fit(
            train_set,
            neg_sampling=neg_sampling,
            verbose=2,
            # eval_data=eval_set,
            metrics=model_metrics,
        )

        # Evaluate the model on the evaluation set and print metrics
        # eval_result = evaluate(
        #     model=model,
        #     data=eval_set,
        #     neg_sampling=neg_sampling,
        #     metrics=model_metrics
        # )
        # print(f"Evaluation results for {model_name}: {eval_result}")

        # Make predictions for Ranking Task
        print("    Making ranking predictions...")
        predictions = model.predict(
            user=pairs_read["user"].tolist(),
            item=pairs_read["item"].tolist(),
            cold_start='popular'
        )

        # Save predictions to CSV
        output_path = model_dir / f"predictions_Ranking.csv"
        print(f"    Saving ranking predictions to {output_path}...")
        pairs_read_copy = pairs_read.copy()
        pairs_read_copy["prediction"] = predictions
        pairs_read_copy.rename(columns={"user": "userID", "item": "bookID"}).to_csv(output_path, index=False)

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        continue

import os
import time
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

import cornac
from cornac.data import Dataset
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, AUC

# Import all the models
from cornac.models.bivaecf import BiVAECF
from cornac.models.lightgcn import LightGCN
from cornac.models.recvae import RecVAE
from cornac.models.ease import EASE
from cornac.models.ngcf import NGCF
from cornac.models.gcmc import GCMC
from cornac.models.vaecf import VAECF
from cornac.models.ncf import GMF, MLP, NeuMF
from cornac.models.ibpr import IBPR
from cornac.models.online_ibpr import OnlineIBPR
from cornac.models.coe import COE
from cornac.models.skm import SKMeans
from cornac.models.hpf import HPF
from cornac.models.bpr import WBPR
from cornac.models.mf import MF
from cornac.models.mmmf import MMMF
from cornac.models.nmf import NMF
from cornac.models.pmf import PMF
from cornac.models.svd import SVD
from cornac.models.wmf import WMF

import torch

def readCSV(path):
    with gzip.open(path, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            yield line.strip().split(',')

# Suppress warnings
warnings.filterwarnings("ignore")

# Read training data into DataFrame
print("Loading training data into DataFrame...")
train_data = pd.DataFrame(
    list(readCSV("train_Interactions.csv.gz")),
    columns=["userID", "bookID", "rating"]
)

# For read prediction, any interaction implies the user read the book
train_data["rating"] = 1.0  # Assign rating 1.0 for all interactions

# Prepare data as list of tuples (user, item, rating)
data = list(zip(train_data['userID'], train_data['bookID'], train_data['rating']))

# Create Cornac Dataset
print("Creating Cornac Dataset...")
dataset = Dataset.from_uir(data, seed=42)

# Read pairs for Read Prediction Task
print("Loading Read Prediction pairs...")
pairs_read = pd.read_csv("pairs_Read.csv")

# Map user IDs and item IDs to indices
user_id_map = dataset.uid_map
item_id_map = dataset.iid_map

pairs_read['user_idx'] = pairs_read['userID'].map(user_id_map)
pairs_read['item_idx'] = pairs_read['bookID'].map(item_id_map)

# Drop pairs with unknown users or items
pairs_read = pairs_read.dropna(subset=['user_idx', 'item_idx'])
pairs_read['user_idx'] = pairs_read['user_idx'].astype(int)
pairs_read['item_idx'] = pairs_read['item_idx'].astype(int)

# Ensure 'Cornac' directory exists
Path("Cornac").mkdir(parents=True, exist_ok=True)

############################################
# List of Models to Train and Evaluate
############################################
models_to_train = [
    {
        'name': 'VAECF',
        'class': VAECF,
        'params': {
            'name': 'VAECF',
            'k': 20,
            'autoencoder_structure': [20],
            'act_fn': 'tanh',
            'likelihood': 'mult',
            'n_epochs': 100,
            'batch_size': 100,
            'learning_rate': 0.002,
            'beta': 1.0,
            'trainable': True,
            'verbose': True,
            'seed': 42,
            'use_gpu': torch.cuda.is_available(),
        }
    },
    # BiVAECF
    {
        'name': 'BiVAECF',
        'class': BiVAECF,
        'params': {
            'k': 20,
            'encoder_structure': [20],
            'act_fn': 'tanh',
            'likelihood': 'pois',
            'n_epochs': 100,
            'batch_size': 100,
            'learning_rate': 0.002,
            'beta_kl': 1.0,
            'cap_priors': {'item': False, 'user': False},
            'trainable': True,
            'verbose': True,
            'seed': 42,
            'use_gpu': torch.cuda.is_available(),
        }
    },
]
############################################
# Iterate Through Each Model
############################################
for model_info in models_to_train:
    model_name = model_info['name']
    model_class = model_info['class']
    model_params = model_info['params']

    print(f"\nProcessing model: {model_name}")

    # Create directory for the model
    model_dir = Path("Cornac") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the model with specified parameters
        model = model_class(**model_params)

        # Fit the model
        model.fit(dataset)

        # Make predictions for the user-item pairs in pairs_read
        # Get user indices and item indices
        user_indices = pairs_read['user_idx'].values
        item_indices = pairs_read['item_idx'].values

        # Predict scores
        predictions = []
        for u_idx, i_idx in zip(user_indices, item_indices):
            score = model.score(user_idx=u_idx, item_idx=i_idx)
            predictions.append(score)

        # Save predictions to CSV
        output_path = model_dir / f"predictions_Ranking.csv"
        print(f"    Saving ranking predictions to {output_path}...")
        pairs_read_copy = pairs_read.copy()
        pairs_read_copy["prediction"] = predictions
        pairs_read_copy[['userID', 'bookID', 'prediction']].to_csv(output_path, index=False)

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        continue

############################################
# Final Ensemble Method: Combining Various Read Prediction Models
############################################

import os
import pandas as pd
import numpy as np
from functools import reduce
import ast
import time

ranking_dict = dict()
base_dirs = ['LibRecommender', 'Cornac']

for base_dir in base_dirs:
    if not os.path.isdir(base_dir):
        print(f"Directory '{base_dir}' does not exist.")
        continue
    for dir_name in os.listdir(base_dir):
        # Construct the file path using os.path.join
        file_path = os.path.join(base_dir, dir_name, 'predictions_Ranking.csv')
        exists = os.path.exists(file_path)
        print(f"Checking {file_path}: {exists}")
        if exists:
            try:
                ranking_dict[dir_name] = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

pred_dfs = []

for model_name, model_df in ranking_dict.items():
    model_df = model_df.copy()

    if isinstance(model_df.prediction.iloc[0], str):
        model_df.prediction = model_df.prediction.apply(lambda x: ast.literal_eval(x)[0])

    # Rename the normalized_prediction column to include the model name
    pred_col_name = f'pred_{model_name}'
    model_df.rename(columns={'prediction': pred_col_name}, inplace=True)

    # Keep only the necessary columns for merging
    pred_dfs.append(model_df[['userID', 'bookID', pred_col_name]])

# Merge all normalized prediction DataFrames on ['userID', 'bookID']
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['userID', 'bookID'], how='outer'), pred_dfs)

# Identify all normalized prediction columns
# pred_cols = [col for col in merged_df.columns if col.startswith('pred_')]

pred_cols = [
 # 'pred_ALS',  # 0.6595
 # 'pred_AutoInt',  # 0.7446
 'pred_BPR',  # 0.7519
 # 'pred_Caser',  # 0.7326
 # 'pred_DeepFM',  # 0.6996
 # 'pred_DeepWalk',  # 0.5111
 # 'pred_DIN',  # 0.7162
 # 'pred_FM',  # 0.7327
 'pred_GraphSage',  # 0.7674
 # 'pred_Item2Vec',  # 0.7077
 # 'pred_ItemCF',  # 0.5376
 'pred_LightGCN',  # 0.7631
 # 'pred_NCF',  # 0.7119
 'pred_NGCF',  # 0.7821
 'pred_RNN4Rec',  # 0.7518
 # 'pred_RsItemCF',  # 0.5237
 # 'pred_RsUserCF',  # 0.6006
 # 'pred_SIM',  # 0.7296
 # 'pred_SVD',  # 0.4983
 # 'pred_SVDpp',  # 0.7106
 # 'pred_Swing',  # 0.7206 (with just 7,200 1's)
 # 'pred_Transformer',  # 0.737
 # 'pred_TwoTower',  # 0.7175
 # 'pred_UserCF',  # 0.6595
 # 'pred_WaveNet',  # 0.7281
 # 'pred_WideDeep',  # 0.6872
 # 'pred_YouTubeRanking',  # 0.715
 # 'pred_YouTubeRetrieval',  # 0.7341
 'pred_BiVAECF',  # 0.7584
 # 'pred_GMF',  # 0.735
 # 'pred_HPF',  # 0.6737
 # 'pred_MF',  # 0.4983
 # 'pred_MLP',  # 0.7207
 # 'pred_NeuMF',  # 0.7128
 # 'pred_NMF',  # 0.4983
 # 'pred_PMF',  # 0.4983
 # 'pred_RecVAE',  # 0.6937
 'pred_VAECF',  # 0.7891
 # 'pred_WBPR',  # 0.5409
 # 'pred_WMF'  # 0.5581
]

def rank_list(values):
    # Convert values to a list to avoid issues with non-iterable types like pandas Series
    values = list(values)

    # Separate NaN values and non-NaN values
    non_nan_values = [val for val in values if not np.isnan(val)]
    sorted_values = sorted(set(non_nan_values), reverse=True)  # Unique sorted non-NaN values

    # Create a rank dictionary for non-NaN values
    rank_dict = {val: rank + 1 for rank, val in enumerate(sorted_values)}

    # Assign NaN the lowest rank (one rank below the minimum non-NaN rank)
    nan_rank = max(rank_dict.values(), default=0) + 1

    # Return a list of ranks, with NaN values assigned to `nan_rank`
    return [rank_dict.get(val, nan_rank) for val in values]

for col in pred_cols:
    merged_df[col] = rank_list(merged_df[col])

merged_df['pred_VAECF'] = merged_df['pred_VAECF'] * 7

# Sum the normalized predictions across all models
merged_df['total_score'] = merged_df[pred_cols].sum(axis=1)

# Create a binary 'prediction' column by assigning 1 to the top half ranks per user
merged_df['prediction'] = (
    merged_df.groupby('userID')['total_score']
    .rank(method='first', ascending=True)
    .le(merged_df.groupby('userID')['total_score'].transform('count') / 2)
    .astype(int)
)

# Select the necessary columns for the final output
final_df = merged_df[['userID', 'bookID', 'prediction']]

# Export the final predictions to 'predictions_Read.csv' without the index
final_df.to_csv('predictions_Read.csv', index=False)

print(f"\nAll computations completed successfully in {time.perf_counter() - start_time:.2f} seconds!")
