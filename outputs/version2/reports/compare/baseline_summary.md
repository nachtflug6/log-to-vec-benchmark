# Baseline Summary

## Feature Shapes

| Feature Type   | Train Shape   | Test Shape   |
|:---------------|:--------------|:-------------|
| fft            | [1932, 56]    | [414, 56]    |
| summary        | [1932, 64]    | [414, 64]    |
| raw_flatten    | [1932, 192]   | [414, 192]   |

## Classification Tasks

| Feature Type   | Target      |   Linear Acc |   Linear Bal Acc |   RBF Acc |   RBF Bal Acc |
|:---------------|:------------|-------------:|-----------------:|----------:|--------------:|
| fft            | coupling_id |        0.386 |            0.37  |     0.37  |         0.355 |
| fft            | mode_id     |        0.242 |            0.256 |     0.258 |         0.276 |
| fft            | spectral_id |        0.688 |            0.69  |     0.698 |         0.699 |
| fft            | transition  |        0.652 |            0.558 |     0.652 |         0.539 |
| summary        | coupling_id |        0.406 |            0.413 |     0.386 |         0.387 |
| summary        | mode_id     |        0.244 |            0.277 |     0.208 |         0.235 |
| summary        | spectral_id |        0.698 |            0.701 |     0.7   |         0.701 |
| summary        | transition  |        0.647 |            0.539 |     0.655 |         0.547 |
| raw_flatten    | coupling_id |        0.382 |            0.394 |     0.353 |         0.363 |
| raw_flatten    | mode_id     |        0.085 |            0.13  |     0.184 |         0.22  |
| raw_flatten    | spectral_id |        0.271 |            0.274 |     0.693 |         0.694 |
| raw_flatten    | transition  |        0.548 |            0.483 |     0.652 |         0.53  |

## Regression Tasks

| Feature Type   | Target    | Model   |   MAE |   RMSE |     R2 |
|:---------------|:----------|:--------|------:|-------:|-------:|
| fft            | mean_load | linear  | 0.12  |  0.138 | -0.089 |
| fft            | mean_load | rbf     | 0.119 |  0.138 | -0.089 |
| summary        | mean_load | linear  | 0.117 |  0.135 | -0.044 |
| summary        | mean_load | rbf     | 0.117 |  0.137 | -0.068 |
| raw_flatten    | mean_load | linear  | 0.114 |  0.133 | -0.008 |
| raw_flatten    | mean_load | rbf     | 0.119 |  0.138 | -0.084 |
