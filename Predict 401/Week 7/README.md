Week 7 Practice
================
Alan Kessler

A lot of the problems this week can be done very quickly in R so I am submitting two problems for practice this week.

The following packages are used this week:

``` r
library(knitr)
library(kableExtra)
```

Problem 1
=========

This problem is from Chapter 10: Statistical Inferences about Two Populations, Section 10.3: Statistical Inferences for Two Related Populations [(Black, 2016, p. 346)](#ref).

> 10.26 The vice president of marketing brought to the attention of sales managers that most of the company's manufacturer representatives contacted clients and maintained client relationships in a disorganized, haphazard way. The sales managers brought the reps in for a three-day seminar and training session on how to use an organizer to schedule visits and recall pertinent information about each client more effectively. Sales reps were taught how to schedule visits most efficiently to maximize their efforts. Sales managers were given data on the number of site visits by sales reps on a randomly selected day both before and after the seminar. Use the following data to test whether significantly more site visits were made after the seminar (α = .05). Assume the differences in the number of site visits are normally distributed.

The Data
--------

The data originally from the problem is manually entered into vectors to create a data frame.

``` r
Rep <- 1:9
Before <- c(2, 4, 1, 3, 4, 2, 2, 3, 1)
After <- c(4, 5, 3, 3, 3, 5, 6, 4, 5)
df <- data.frame(Rep, Before, After)

kable(df, format="html") %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Rep
</th>
<th style="text-align:right;">
Before
</th>
<th style="text-align:right;">
After
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
4
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
5
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
5
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
6
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
4
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
5
</td>
</tr>
</tbody>
</table>

Test
----

The test is if the difference in means is significantly greater than 0.

``` r
t.test(x = df$After, y = df$Before, var.equal = TRUE,
       paired = TRUE, alternative = "greater")
```

    ##
    ##  Paired t-test
    ##
    ## data:  df$After and df$Before
    ## t = 3.1081, df = 8, p-value = 0.007243
    ## alternative hypothesis: true difference in means is greater than 0
    ## 95 percent confidence interval:
    ##  0.7141545       Inf
    ## sample estimates:
    ## mean of the differences
    ##                1.777778

Based on the result of the test, I can reject the null hypothesis that the means are equal or differ by less than zero.

Problem 2
=========

This problem is from Chapter 10: Statistical Inferences about Two Populations, Section 10.5: Testing Hypotheses About Two Population Variances [(Black, 2016, p. 361)](#ref).

> 10.42 How long are resale houses on the market? One survey by the Houston Association of Realtors reported that in Houston, resale houses are on the market an average of 112 days. Of course, the length of time varies by market. Suppose random samples of 13 houses in Houston and 11 houses in Chicago that are for resale are traced. The data shown here represent the number of days each house was on the market before being sold. Use the given data and a 1% level of significance to determine whether the population variances for the number of days until resale are different in Houston than in Chicago. Assume the numbers of days resale houses are on the market are normally distributed.

The Data
--------

The data originally from the problem is manually entered into vectors to create a data frame.

``` r
Houston <- c(132, 126, 138, 94, 131, 161, 127, 133, 99, 119, 126, 88, 134)
Chicago <- c(118, 56, 85, 69, 113,  67, 81, 54, 94, 137, 93, NaN, NaN)
df <- data.frame(Houston, Chicago)

kable(df, format="html") %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Houston
</th>
<th style="text-align:right;">
Chicago
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
132
</td>
<td style="text-align:right;">
118
</td>
</tr>
<tr>
<td style="text-align:right;">
126
</td>
<td style="text-align:right;">
56
</td>
</tr>
<tr>
<td style="text-align:right;">
138
</td>
<td style="text-align:right;">
85
</td>
</tr>
<tr>
<td style="text-align:right;">
94
</td>
<td style="text-align:right;">
69
</td>
</tr>
<tr>
<td style="text-align:right;">
131
</td>
<td style="text-align:right;">
113
</td>
</tr>
<tr>
<td style="text-align:right;">
161
</td>
<td style="text-align:right;">
67
</td>
</tr>
<tr>
<td style="text-align:right;">
127
</td>
<td style="text-align:right;">
81
</td>
</tr>
<tr>
<td style="text-align:right;">
133
</td>
<td style="text-align:right;">
54
</td>
</tr>
<tr>
<td style="text-align:right;">
99
</td>
<td style="text-align:right;">
94
</td>
</tr>
<tr>
<td style="text-align:right;">
119
</td>
<td style="text-align:right;">
137
</td>
</tr>
<tr>
<td style="text-align:right;">
126
</td>
<td style="text-align:right;">
93
</td>
</tr>
<tr>
<td style="text-align:right;">
88
</td>
<td style="text-align:right;">
NaN
</td>
</tr>
<tr>
<td style="text-align:right;">
134
</td>
<td style="text-align:right;">
NaN
</td>
</tr>
</tbody>
</table>

Test
----

The test is if the difference in means is significantly greater than 0.

``` r
var.test(x = Houston, y = Chicago, alternative = "two.sided")
```

    ##
    ##  F test to compare two variances
    ##
    ## data:  Houston and Chicago
    ## F = 0.55984, num df = 12, denom df = 10, p-value = 0.3387
    ## alternative hypothesis: true ratio of variances is not equal to 1
    ## 95 percent confidence interval:
    ##  0.1546127 1.8886640
    ## sample estimates:
    ## ratio of variances
    ##          0.5598442

Based on the result of the test, I cannot reject the null hypothesis that the ratio of variances is equal to 1.

References
==========

Black, K. (2016). *Business statistics : for contemporary decision making*. Hoboken: Wiley.
