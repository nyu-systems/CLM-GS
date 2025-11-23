# CLM-GS Example Scripts: Rubble 2K

# Rubble 2K Experiment Results

## Performance Metrics by Model Size and Offload Strategy

| Experiment                 | Test PSNR   | Train PSNR   | Num 3DGS   | Max GPU Memory (GB)   | Pinned CPU Memory (GB)   | Training Time (s)   |
|:---------------------------|:------------|:-------------|:-----------|:----------------------|:-------------------------|:--------------------|
| rubble2k_10m_clm_offload   | 26.8        | 28.26        | 15757325   | 6.64                  | 11.31                    | 11458.1             |
| rubble2k_10m_naive_offload | 26.61       | 28.09        | 15186773   | 10.26                 | 19.3                     | 20952.91            |
| rubble2k_10m_no_offload    | 26.5        | 28.31        | 15560921   | 21.5                  | 0.45                     | 6710.64             |
| rubble2k_28m_clm_offload   | 20.41       | 20.76        | 29815430   | 11.54                 | 11.43                    | OOM                 |
| rubble2k_28m_naive_offload | 20.21       | 21.01        | 35001330   | 22.52                 | 38.23                    | OOM                 |
| rubble2k_28m_no_offload    | 20.28       | 20.58        | 16589350   | 22.39                 | 0.45                     | OOM                 |
| rubble4k_28M_no_offload    | OOM         | OOM          | OOM        | OOM                   | OOM                      | OOM                 |


