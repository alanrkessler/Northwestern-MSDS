Week 3 Practice
================
Alan Kessler

This problem is from Chapter 5: Discrete Distributions, Section 5.4: Poisson Distribution [(Black, 2016, p. 153)](#ref).

> 5.20
> On Monday mornings, the First National Bank only has one teller window open for deposits and withdrawals. Experience has shown that the average number of arriving customers in a four-minute interval on Monday mornings is 2.8, and each teller can serve more than that number efficiently. These random arrivals at this bank on Monday mornings are Poisson distributed.
>
> 1.  What is the probability that on a Monday morning exactly six customers will arrive in a four-minute interval?
>
> 2.  What is the probability that no one will arrive at the bank to make a deposit or withdrawal during a four-minute interval?
>
> 3.  Suppose the teller can serve no more than four customers in any four-minute interval at this window on a Monday morning. What is the probability that, during any given four-minute interval, the teller will be unable to meet the demand? What is the probability that the teller will be able to meet the demand? When demand cannot be met during any given interval, a second window is opened. What percentage of the time will a second window have to be opened?
>
> 4.  What is the probability that exactly three people will arrive at the bank during a two-minute period on Monday mornings to make a deposit or a withdrawal? What is the probability that five or more customers will arrive during an eight-minute period?
>
The following package is used in this problem:

``` r
library(scales)

# Adjust my output to avoid scientific notation
options(scipen=999)
```

Part A
------

"What is the probability that on a Monday morning exactly six customers will arrive in a four-minute interval?" suggests that I need to calculate:<a href="https://www.codecogs.com/eqnedit.php?latex=P(n=6)=\frac{\lambda^{6}e^{-\lambda}}{6!}" target="_blank"><img src="https://latex.codecogs.com/png.latex?P(n=6)=\frac{\lambda^{6}e^{-\lambda}}{6!}" title="P(n=6)=\frac{\lambda^{6}e^{-\lambda}}{6!}" /></a>.

``` r
(2.8**6*exp(-2.8))/factorial(6)
```

The resulting probability of exactly six customers in a four-minute interval is **0.0407**. I can check using R functions as well.

``` r
dpois(6, 2.8)
```

I get a matching result of **0.0407**.

Part B
------

"What is the probability that no one will arrive at the bank to make a deposit or withdrawal during a four-minute interval?" suggests that I need to calculate what simplifies to:*P*(*n* = 0)=*e*<sup>−*λ*</sup>.

``` r
exp(-2.8)
```

This results in a probability of no customers will arrive during a four-minute interval of **0.0608**.

Part C
------

#### Question 1

"Suppose the teller can serve no more than four customers in any four-minute interval at this window on a Monday morning. What is the probability that, during any given four-minute interval, the teller will be unable to meet the demand?" suggests that I need to calculate *P*(*n* &gt; 4).

``` r
ppois(4, 2.8, lower.tail = FALSE)
```

This results in a probability that the teller will be unable to meet the demand of **0.1523**.

#### Question 2

"What is the probability that the teller will be able to meet the demand?" suggests that I need to calculate *P*(*n* ≤ 4).

``` r
ppois(4, 2.8, lower.tail = TRUE)
```

This results in a probability that the teller will be unable to meet the demand of **0.8477**.

#### Question 3

"When demand cannot be met during any given interval, a second window is opened. What percentage of the time will a second window have to be opened?" refers to the question 1.

The probability would also refer to the expected percentage of time a second window will need to be opened, **15.2%**.

Part D
------

#### Question 1

"What is the probability that exactly three people will arrive at the bank during a two-minute period on Monday mornings to make a deposit or a withdrawal?" suggests that I need to adjust lambda to a two minute interval.This involves simply dividing by two to get *λ* = 1.4.

``` r
dpois(3, 1.4)
```

This results in a probability that exactly three people will arrive at the bank during a two-minute period of **0.1128**.

#### Question 2

"What is the probability that five or more customers will arrive during an eight-minute period?" suggest that instead of halving the value lambda, I should double it to get *λ* = 5.6.

``` r
ppois(5, 5.6, lower.tail = FALSE)
```

This results in a probability that five or more customers will arrive during an eight-minute period of **0.4881**.

References
----------

Black, K. (2016). *Business statistics : for contemporary decision making*. Hoboken: Wiley.
