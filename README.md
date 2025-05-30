## Final Results (ranks mentioned are on the 258 Gradescope leaderboard, out of 737 students)

- **Rating Prediction (Public)**: **1.399998740354906** (Rank 1)
- **Rating Prediction (Private)**: **1.45064734344175** (Rank 3)
- **Read Prediction (Public)**: **0.8621** (Rank 1)
- **Read Prediction (Private)**: **0.8597** (Rank 1)

## Explanation of Final Solution

For this assignment, I utilized the [LibRecommender (LibReco)](https://librecommender.readthedocs.io/en/latest/) and [Cornac](https://cornac.readthedocs.io/en/latest/index.html) libraries to implement recommendation models for both tasks.

### Rating Prediction Task

- **Approach**: Combined a Latent Factor Model with Biases and a Transformer-based model.
  - **Latent Factor Model with Biases**: Implemented a custom matrix factorization model in `numpy` that includes global bias, user biases, and item biases, along with latent factors for users and items. Formula: $$\hat{r}_{ui} = \alpha + \beta_u + \beta_i + \mathbf{\gamma}_u^\top \mathbf{\gamma}_i$$.
  - **Transformer Model**: Used the Transformer model from LibRecommender to capture complex sequential patterns in the data.
- **Ensemble Method**: To enhance performance, I combined the predictions from both models using a weighted average, assigning more weight $$(80 \%)$$ to the Transformer model.
- **Iterations**: Ran multiple iterations with hyperparameter optimization using `optuna` to find the best configurations. Averaged the best results from several runs to improve the MSE on the public leaderboard.

#### **Hyperparameters Used**

| Model                        | Hyperparameter           | Value                      |
|------------------------------|--------------------------|----------------------------|
| **Latent Factor Model (LFM)**| Number of Latent Factors | 20                         |
|                              | Learning Rate            | 0.01                       |
|                              | Regularization           | 0.21                       |
|                              | Number of Epochs         | 20                         |
|                              | Initialization Scale     | 1e-6                       |
| **Transformer**              | Task                     | Rating                     |
|                              | Embed Size               | 512                        |
|                              | Number of Epochs         | 3                          |
|                              | Learning Rate            | 0.0003392308111583879      |
|                              | Learning Rate Decay      | True                       |
|                              | Epsilon                  | 1e-05                      |
|                              | Regularization           | 8.303908209260279e-05      |
|                              | Batch Size               | 256                        |
|                              | Sampler                  | 'random'                   |
|                              | Use Batch Normalization  | False                      |
|                              | Dropout Rate             | 0.10142529816569815        |
|                              | Hidden Units             | (512, 256, 128)            |
|                              | Number of Heads          | 16                         |
|                              | Number of Transformer Layers | 6                      |
|                              | Multi Sparse Combiner    | 'sqrtn'                    |
|                              | Seed                     | 153                        |
|                              | Lower Upper Bound        | (1, 7)                     |

### Read Prediction Task

- **Approach**: Implemented an ensemble of various ranking algorithms from LibReco and Cornac.
  - Trained models including `NGCF`, `GraphSage`, `LightGCN`, `BPR`, `RNN4Rec`, `VAECF`, and `BiVAECF`.
- **Ensemble Method**: Combined the predictions from all models by ranking their scores and aggregating them to make the final binary prediction. _**For each user, classified half of the highest ranked interactions as positive and the rest as negative (because the test dataset was generated with 1 negative sample per each unique positive interaction).**_
- **Iterations**: Performed multiple runs and selected the best-performing models out of all the ranking models available in said libraries, based on their accuracy on the validation set and the public leaderboard.

#### **Hyperparameters Used**

| Model        | Hyperparameter           | Value                      |
|--------------|--------------------------|----------------------------|
| **GraphSage**| Task                     | Ranking                    |
|              | Loss Type                | 'bpr'                      |
|              | Embed Size               | 128                        |
|              | Number of Layers         | 2                          |
|              | Sampler                  | 'popular'                  |
| **LightGCN** | Task                     | Ranking                    |
|              | Loss Type                | 'bpr'                      |
|              | Embed Size               | 128                        |
|              | Number of Layers         | 3                          |
|              | Sampler                  | 'unconsumed'               |
| **NGCF**     | Task                     | Ranking                    |
|              | Loss Type                | 'bpr'                      |
|              | Embed Size               | 128                        |
|              | Number of Epochs         | 50                         |
|              | Learning Rate            | 0.001                      |
|              | Learning Rate Decay      | False                      |
|              | Regularization           | 1e-4                       |
|              | Batch Size               | 1024                       |
|              | Number of Negatives      | 4                          |
|              | Node Dropout             | 0.1                        |
|              | Message Dropout          | 0.1                        |
|              | Hidden Units             | (64, 64, 64)               |
|              | Sampler                  | 'unconsumed'               |
| **RNN4Rec**  | Task                     | Ranking                    |
|              | Loss Type                | 'bpr'                      |
|              | Embed Size               | 128                        |
|              | Hidden Units             | 16                         |
|              | Sampler                  | 'unconsumed'               |
|              | Recent Number            | 10                         |
| **BPR**      | Task                     | Ranking                    |
|              | Embed Size               | 128                        |
|              | Use TensorFlow           | True                       |
|              | Sampler                  | 'unconsumed'               |
| **VAECF**    | Latent Factors (k)       | 20                         |
|              | Autoencoder Structure    | [20]                       |
|              | Activation Function      | 'tanh'                     |
|              | Likelihood               | 'mult'                     |
|              | Number of Epochs         | 100                        |
|              | Batch Size               | 100                        |
|              | Learning Rate            | 0.002                      |
|              | Beta                     | 1.0                        |
|              | Trainable                | True                       |
|              | Verbose                  | True                       |
| **BiVAECF**  | Latent Factors (k)       | 20                         |
|              | Encoder Structure        | [20]                       |
|              | Activation Function      | 'tanh'                     |
|              | Likelihood               | 'pois'                     |
|              | Number of Epochs         | 100                        |
|              | Batch Size               | 100                        |
|              | Learning Rate            | 0.002                      |
|              | Beta KL                  | 1.0                        |
|              | Cap Priors               | {'item': False, 'user': False} |
|              | Trainable                | True                       |
|              | Verbose                  | True                       |

### Overfitting Note

While optimizing for the public leaderboard, I may have overfitted to that subset of data, which resulted in a slight drop in performance on the private test dataset for the rating prediction task, moving from rank 1 to rank 3.

SVD / LFM got me down to $$\approx 1.4135$$ MSE on the public leaderboard but it produced high variance in the results. I also didn't use PyTorch in my implementation.

Ablation studies:
1. For the **read prediction** task:
    - Implementing the 50/50 split between positive and negative classifications for each user's interactions gave me the largest boost on AUC & accuracy on the public leaderboard. In particular, it boosted my accuracy from 0.7938 to 0.8621 with one logic change. 
    - For selecting the models to factor into my final ensemble approach, I looked at the validation accuracy (AUC), precision, recall, and the accuracy on the public leaderboard for all ranking algorithms available in LibReco and in Cornac. I also attempted to implement a ranking method for cutting-edge collaborative filtering models that are known to be the best at this task (see [PapersWithCode](https://paperswithcode.com/task/collaborative-filtering)); in particular, I tried out [BSPM](https://github.com/jeongwhanchoi/BSPM), [DiffRec](https://github.com/YiyanXu/DiffRec), and [CF-KAN](https://github.com/jindeok/CF-KAN). The papers for these models provide a comparative analysis of these models' performance with some of the popular older algorithms that I have used like `NGCF` using common metrics like Precision@20, Recall@20 with the same dataset; ergo, this approach seemed lucrative to me. However, I only succeeded in reaching $$\approx 75\%$$ accuracy with `BSPM` and did not get decent results with the other models. It's possible that these models overfit the training data, and I could have spent some time hyperopting best parameters for them to capture implicit user-book relationships and achieved better results with them.
2. For the **rating prediction** task, I realized that the [Transformers4Rec](https://dl.acm.org/doi/10.1145/3460231.3474255) model was overfitting very quickly (i.e., the training loss was decreasing, indicating effective training, but my validation RMSE was increasing with subsequent epochs). Thus, I reduced the number of epochs in my `optuna` trials and provided more leeway with lower learning rates. I also observed that my validation RMSE and the leaderboard MSE decreased significantly with larger sized embeddings and training with more hidden units, but I had limited compute available (DataHub & Kaggle's T4 x2 GPUs) to scale my solution further.
