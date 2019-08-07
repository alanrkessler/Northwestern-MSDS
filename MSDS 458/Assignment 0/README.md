# Assignment 0

Demonstrates how backpropagation is derived and shows how it can be used to solve the XOR problem.

```
Training Model

Input:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]

Target:
[[0 1]
 [1 0]
 [1 0]
 [0 1]]

 Total SSE by Iteration

   0. 2.0623
 100. 2.0004
 200. 2.0003
 300. 2.0004
 400. 1.9996
 500. 1.9974
 600. 1.9838
 700. 1.8977
 800. 1.6345
 900. 1.4576
1000. 1.3782
1100. 1.2831
1200. 1.0052
1300. 0.4354
1400. 0.1951
1500. 0.1151
1600. 0.0794
1700. 0.0599

------------

Input array:
[0 0]

Target array
[0 1]

Weights (first row corresponds to first output):
[array([[-4.26780679, -4.27825257],
       [-6.50177003, -6.59680989]]), array([[ 6.10839383, -6.47054433],
       [-6.20770112,  6.57469073]])]

Biases:
[array([6.26681704, 2.51723276]), array([-2.76119179,  2.80705311])]

Output Layer Final Output:
[0.06587619 0.93672771]

------------

Input array:
[0 1]

Target array
[1 0]

Weights (first row corresponds to first output):
[array([[-4.26780679, -4.27825257],
       [-6.50177003, -6.59680989]]), array([[ 6.10839383, -6.47054433],
       [-6.20770112,  6.57469073]])]

Biases:
[array([6.26681704, 2.51723276]), array([-2.76119179,  2.80705311])]

Output Layer Final Output:
[0.92442674 0.07284226]

------------

Input array:
[1 0]

Target array
[1 0]

Weights (first row corresponds to first output):
[array([[-4.26780679, -4.27825257],
       [-6.50177003, -6.59680989]]), array([[ 6.10839383, -6.47054433],
       [-6.20770112,  6.57469073]])]

Biases:
[array([6.26681704, 2.51723276]), array([-2.76119179,  2.80705311])]

Output Layer Final Output:
[0.92416058 0.07310363]

------------

Input array:
[1 1]

Target array
[0 1]

Weights (first row corresponds to first output):
[array([[-4.26780679, -4.27825257],
       [-6.50177003, -6.59680989]]), array([[ 6.10839383, -6.47054433],
       [-6.20770112,  6.57469073]])]

Biases:
[array([6.26681704, 2.51723276]), array([-2.76119179,  2.80705311])]

Output Layer Final Output:
[0.10027702 0.90298085]
```