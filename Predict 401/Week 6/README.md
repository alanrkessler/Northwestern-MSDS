Week 6 Practice
================
Alan Kessler

A lot of the problems this week can be done very quickly in R so I am submitting two problems for practice this week.

The following packages are used this week:

``` r
library(dplyr)
library(knitr)
library(kableExtra)
```

Problem 1
=========

This problem is from Chapter 16: Analysis of Categorical Data, Section 16.1: Chi-Square Goodness-of-Fit Test [(Black, 2016, p. 608)](#ref).

> 16.8 The Springfield Emergency Medical Service keeps records of emergency telephone calls. A study of 150 five-minute time intervals resulted in the distribution of number of calls that follows. For example, during 18 of the 5-minute intervals, no calls occured. Use the chi-square goodness-of-fit test and *α* = 0.01 to determine whether this distribution is Poisson.

The Data
--------

The data originally from the problem is manually entered into vectors to create a data frame.

``` r
Frequency <- c(18, 28, 47, 21, 16, 11, 9)
df <- data.frame(Frequency)
rownames(df) <- c("0", "1", "2", "3", "4", "5", "6 or more")

kable(df, format="html") %>%
  kable_styling(full_width = FALSE) %>%
  group_rows("Calls", 1, 7)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Frequency
</th>
</tr>
</thead>
<tbody>
<tr grouplength="7">
<td colspan="2" style="border-bottom: 1px solid;">
<strong>Calls</strong>
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
1
</td>
<td style="text-align:right;">
28
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
2
</td>
<td style="text-align:right;">
47
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
3
</td>
<td style="text-align:right;">
21
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
4
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
5
</td>
<td style="text-align:right;">
11
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
6 or more
</td>
<td style="text-align:right;">
9
</td>
</tr>
</tbody>
</table>

Test for Goodness-of-Fit
------------------------

``` r
# Fill in Poisson probabilities
df$ncalls <- (0:6)
lambda <- sum(df$ncalls * df$Frequency) / sum(df$Frequency)
df$PoisProbs <- dpois(df$ncalls, lambda)

# Replace the last cell to reflect the inequality
df$PoisProbs[7] <- 1 - sum(df$PoisProbs[0:6])

chisq.test(x = df$Frequency, p = df$PoisProbs)
```

    ##
    ##  Chi-squared test for given probabilities
    ##
    ## data:  df$Frequency
    ## X-squared = 10.483, df = 6, p-value = 0.1057

The p-value is greater than 0.01 which suggest that we cannot reject the null hypothesis that the number of calls is Poisson distributed at this value for *α*.

Problem 2
=========

This problem is from Chapter 16: Analysis of Categorical Data, Section 16.2: Contingency Analysis: Chi-Square Test of Independence [(Black, 2016, p. 614)](#ref).

> 16.12 Use the following contingency table and chi-square test of independence to determine whether social class is independent of number of children in a family. Let *α* = 0.05.

The Data
--------

The data originally from the problem is manually entered into a table.

``` r
ClassStudy <- as.table(rbind(c(7, 18, 6),
                             c(9, 38, 23),
                             c(34, 97, 58),
                             c(47, 31, 30)))

# Label columns and rows
dimnames(ClassStudy) <- list(NumberChildren = c("0", "1", "2 or 3", "More than 3"),
                             SocialClass = c("Lower","Middle", "Upper"))


kable(ClassStudy, format="html") %>%
  kable_styling(full_width = FALSE) %>%
  add_header_above(c(" ", "Social Class" = 3)) %>%
  group_rows("Children", 1, 4)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="border-bottom:hidden" colspan="1">
</th>
<th style="text-align:center; border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;" colspan="3">
Social Class

</th>
</tr>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
Lower
</th>
<th style="text-align:right;">
Middle
</th>
<th style="text-align:right;">
Upper
</th>
</tr>
</thead>
<tbody>
<tr grouplength="4">
<td colspan="4" style="border-bottom: 1px solid;">
<strong>Children</strong>
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
0
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
6
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
1
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
38
</td>
<td style="text-align:right;">
23
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
2 or 3
</td>
<td style="text-align:right;">
34
</td>
<td style="text-align:right;">
97
</td>
<td style="text-align:right;">
58
</td>
</tr>
<tr>
<td style="text-align:left; padding-left: 2em;" indentLevel="1">
More than 3
</td>
<td style="text-align:right;">
47
</td>
<td style="text-align:right;">
31
</td>
<td style="text-align:right;">
30
</td>
</tr>
</tbody>
</table>

Test for Independence
---------------------

``` r
chisq.test(ClassStudy)
```

    ##
    ##  Pearson's Chi-squared test
    ##
    ## data:  ClassStudy
    ## X-squared = 34.963, df = 6, p-value = 4.382e-06

The p-value is less than 0.05 which suggests that we can reject the null hypothesis that social class is independent of the number of children in a family at this value for *α*.

References
==========

Black, K. (2016). *Business statistics : for contemporary decision making*. Hoboken: Wiley.
