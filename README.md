# Cardiac-electrophysiology

Task 1: https://arxiv.org/pdf/1805.00794.pdf

Task 3 & 4:  
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10081783&tag=1
https://medium.com/@mikel.landajuela.larma/machine-learning-for-cardiac-electrocardiography-a20661669937
https://landajuela.github.io/cardiac_ml/

Options for Task 3:

| Models | CNN | LSTM | Transformers |
| -------- | -------- | -------- | -------- |
| Pros   | Effective for capturing local patterns   | Captures long-term dependencies    | Handles variable-length sequences   |
|     | Parameter sharing reduces complexity  | Can handle sequential and temporal data    | Parallelizable, efficient computation   |
|     | Computationally efficient	| Robust to noise and missing data	| Self-attention captures global context |
|     | Good for feature extraction	| Supports online learning	| Allows for parallel training |
| Cons   | Limited capture of global dependencies	|  Computationally expensive	| Memory-intensive for large sequences |
|     | May require stacking layers for depth	| Susceptible to vanishing/exploding grad.	| Lack of positional information |
|     | Assumes stationarity in the data	| Longer training time	| Limited interpretability |
|     | May struggle with variable-length input	| Difficulty handling irregular intervals | |
