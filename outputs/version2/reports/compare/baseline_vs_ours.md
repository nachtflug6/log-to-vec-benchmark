# Baseline vs Ours

## Best Overall Baseline vs Ours

| Target      | Best Baseline Feature   | Best Baseline Model   |   Best Baseline Acc |   Best Baseline Bal Acc | Best Ours Model   |   Best Ours Acc |   Best Ours Bal Acc |
|:------------|:------------------------|:----------------------|--------------------:|------------------------:|:------------------|----------------:|--------------------:|
| mode_id     | summary                 | linear                |               0.244 |                   0.277 | rbf               |           0.242 |               0.267 |
| spectral_id | summary                 | rbf                   |               0.7   |                   0.701 | linear            |           0.657 |               0.653 |
| coupling_id | summary                 | linear                |               0.406 |                   0.413 | linear            |           0.411 |               0.423 |
| transition  | fft                     | linear                |               0.652 |                   0.558 | rbf               |           0.635 |               0.534 |
| mean_load   | raw_flatten             | linear                |               0.114 |                  -0.008 | rbf               |           0.112 |               0.004 |

## Classification Comparison

| Target      | Baseline Feature   |   Baseline Linear Acc |   Baseline Linear Bal Acc |   Baseline RBF Acc |   Baseline RBF Bal Acc |   Ours Linear Acc |   Ours Linear Bal Acc |   Ours RBF Acc |   Ours RBF Bal Acc |
|:------------|:-------------------|----------------------:|--------------------------:|-------------------:|-----------------------:|------------------:|----------------------:|---------------:|-------------------:|
| mode_id     | fft                |                 0.242 |                     0.256 |              0.258 |                  0.276 |             0.244 |                 0.266 |          0.242 |              0.267 |
| mode_id     | summary            |                 0.244 |                     0.277 |              0.208 |                  0.235 |             0.244 |                 0.266 |          0.242 |              0.267 |
| mode_id     | raw_flatten        |                 0.085 |                     0.13  |              0.184 |                  0.22  |             0.244 |                 0.266 |          0.242 |              0.267 |
| spectral_id | fft                |                 0.688 |                     0.69  |              0.698 |                  0.699 |             0.657 |                 0.653 |          0.643 |              0.641 |
| spectral_id | summary            |                 0.698 |                     0.701 |              0.7   |                  0.701 |             0.657 |                 0.653 |          0.643 |              0.641 |
| spectral_id | raw_flatten        |                 0.271 |                     0.274 |              0.693 |                  0.694 |             0.657 |                 0.653 |          0.643 |              0.641 |
| coupling_id | fft                |                 0.386 |                     0.37  |              0.37  |                  0.355 |             0.411 |                 0.423 |          0.399 |              0.408 |
| coupling_id | summary            |                 0.406 |                     0.413 |              0.386 |                  0.387 |             0.411 |                 0.423 |          0.399 |              0.408 |
| coupling_id | raw_flatten        |                 0.382 |                     0.394 |              0.353 |                  0.363 |             0.411 |                 0.423 |          0.399 |              0.408 |
| transition  | fft                |                 0.652 |                     0.558 |              0.652 |                  0.539 |             0.597 |                 0.526 |          0.635 |              0.534 |
| transition  | summary            |                 0.647 |                     0.539 |              0.655 |                  0.547 |             0.597 |                 0.526 |          0.635 |              0.534 |
| transition  | raw_flatten        |                 0.548 |                     0.483 |              0.652 |                  0.53  |             0.597 |                 0.526 |          0.635 |              0.534 |

## Regression Comparison

| Baseline Feature   |   Baseline Linear MAE |   Baseline Linear RMSE |   Baseline Linear R2 |   Baseline RBF MAE |   Baseline RBF RMSE |   Baseline RBF R2 |   Ours Linear MAE |   Ours Linear RMSE |   Ours Linear R2 |   Ours RBF MAE |   Ours RBF RMSE |   Ours RBF R2 |
|:-------------------|----------------------:|-----------------------:|---------------------:|-------------------:|--------------------:|------------------:|------------------:|-------------------:|-----------------:|---------------:|----------------:|--------------:|
| fft                |                 0.12  |                  0.138 |               -0.089 |              0.119 |               0.138 |            -0.089 |             0.121 |              0.141 |           -0.136 |          0.112 |           0.132 |         0.004 |
| summary            |                 0.117 |                  0.135 |               -0.044 |              0.117 |               0.137 |            -0.068 |             0.121 |              0.141 |           -0.136 |          0.112 |           0.132 |         0.004 |
| raw_flatten        |                 0.114 |                  0.133 |               -0.008 |              0.119 |               0.138 |            -0.084 |             0.121 |              0.141 |           -0.136 |          0.112 |           0.132 |         0.004 |
