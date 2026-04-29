# Evaluation Results Summary

## Probe Results

| Target               | Model   |   Accuracy |   Balanced Acc |      MAE |     RMSE |       R2 |
|:---------------------|:--------|-----------:|---------------:|---------:|---------:|---------:|
| mode_id              | linear  |     0.244  |         0.266  | nan      | nan      | nan      |
| mode_id              | rbf     |     0.2415 |         0.2674 | nan      | nan      | nan      |
| spectral_id          | linear  |     0.657  |         0.6529 | nan      | nan      | nan      |
| spectral_id          | rbf     |     0.6425 |         0.6409 | nan      | nan      | nan      |
| coupling_id          | linear  |     0.4106 |         0.4231 | nan      | nan      | nan      |
| coupling_id          | rbf     |     0.3986 |         0.4077 | nan      | nan      | nan      |
| is_transition_window | linear  |     0.5966 |         0.5263 | nan      | nan      | nan      |
| is_transition_window | rbf     |     0.6353 |         0.5344 | nan      | nan      | nan      |
| mean_load            | linear  |   nan      |       nan      |   0.1211 |   0.1408 |  -0.1356 |
| mean_load            | rbf     |   nan      |       nan      |   0.1123 |   0.1319 |   0.0035 |

## Retrieval Performance

| Target               |    R@1 |    R@5 |   R@10 |   Top1 Match |   Top5 Match |
|:---------------------|-------:|-------:|-------:|-------------:|-------------:|
| mode_id              | 0.7802 | 0.9614 | 0.9783 |       0.7802 |       0.5092 |
| spectral_id          | 0.8623 | 0.9831 | 0.9903 |       0.8623 |       0.7005 |
| coupling_id          | 0.8502 | 0.9928 | 1      |       0.8502 |       0.6164 |
| is_transition_window | 0.6981 | 0.9686 | 0.9952 |       0.6981 |       0.6077 |

## Clustering Metrics

| Target               |   Clusters |     ARI |    NMI |   Purity |   Silhouette |
|:---------------------|-----------:|--------:|-------:|---------:|-------------:|
| mode_id              |         12 |  0.1344 | 0.3394 |   0.3454 |       0.1354 |
| spectral_id          |          4 |  0.03   | 0.0792 |   0.3502 |       0.1509 |
| coupling_id          |          3 |  0.0197 | 0.0213 |   0.4469 |       0.1286 |
| is_transition_window |          2 | -0.008  | 0.0006 |   0.6401 |       0.1555 |

## Transition vs Clean

| Metric              |   Clean Acc |   Transition Acc |   Clean |   Transition |
|:--------------------|------------:|-----------------:|--------:|-------------:|
| Mode Classification |      0.2819 |           0.2226 |     nan |     nan      |
| Retrieval@5         |    nan      |         nan      |       1 |       0.9396 |

## Transition Difficulty

| Distance   |   Samples |   Accuracy |   Balanced Acc |
|:-----------|----------:|-----------:|---------------:|
| near       |        54 |     0.2963 |         0.3048 |
| mid        |        89 |     0.2584 |         0.2795 |
| far        |       122 |     0.1639 |         0.1882 |

