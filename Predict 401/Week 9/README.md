Week 9 Practice
================
Alan Kessler

A lot of the problems this week are related so for chapter 12, I will base multiple problems on the data in problem 12.6. Then I will complete a few of the problems

The following packages are used this week:

``` r
library(knitr)
library(kableExtra)
```

Chapter 12
==========

Data
----

The data used in the next two problems is from Chapter 12: Simple Regression Analysis and Correlation, Section 12.3: Determining the Equation of the Regression Line, problem 12.6 [(Black, 2016, p. 435)](#ref).

The data originally from the problem is manually entered into vectors to create a data frame.

``` r
x <- c(12, 21, 28, 8, 20)
y <- c(17, 15, 22, 19, 24)
df <- data.frame(x, y)

kable(df, format="html") %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
x
</th>
<th style="text-align:right;">
y
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
15
</td>
</tr>
<tr>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
22
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
19
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
24
</td>
</tr>
</tbody>
</table>

Problem 1
---------

This problem is from Chapter 12: Simple Regression Analysis and Correlation, Section 12.6: Coefficient of Determination [(Black, 2016, p. 448)](#ref).

> 12.32 Compute *r*<sup>2</sup> for Problem 12.24 (Problem 12.6). Discuss the value of *r*<sup>2</sup> obtained.

``` r
m <- summary(lm(y ~ x))
round(m$r.squared, 4)
```

    ## [1] 0.1233

The value calculated above shows that x explains about 12% of the variability of *y*. Like the section discusses, the acceptability of this result depends on the context of the use of the model.

Problem 2
---------

This problem is from Chapter 12: Simple Regression Analysis and Correlation, Section 12.7: Hypothesis Tests for the Slope of the Regression Model and Testing the Overall Model [(Black, 2016, p. 453)](#ref).

> 12.38 Test the slope of the regression line determined in Problem 12.6. Use *α* = 0.05.

``` r
m$coefficients[2,]
```

    ##   Estimate Std. Error    t value   Pr(>|t|)
    ##  0.1623794  0.2499729  0.6495882  0.5622608

The result is that x is not a signficant variable in the model at the specified confidence level.

Chapter 13
==========

This problem is from Chapter 13: Multiple Regression Analysis, Section 13.1: The Multiple Regression Model [(Black, 2016, p. 480)](#ref).

> 13.2 Use a computer to develop the equation of the regression model for the following data. Comment on the regression coefficientss. Determine the predicted value of y for *x*<sub>1</sub> = 33, *x*<sub>2</sub> = 29, and *x*<sub>3</sub> = 13.

Data
----

``` r
y <- c(114, 94, 87, 98, 101)
x1 <- c(21, 43, 56, 19, 29)
x2 <- c(6, 25, 42, 27, 20)
x3 <- c(5, 8, 25, 9, 12)
df <- data.frame(y, x1, x2, x3)

kable(df, format="html") %>%
  kable_styling(full_width = FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
y
</th>
<th style="text-align:right;">
x1
</th>
<th style="text-align:right;">
x2
</th>
<th style="text-align:right;">
x3
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
114
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
5
</td>
</tr>
<tr>
<td style="text-align:right;">
94
</td>
<td style="text-align:right;">
43
</td>
<td style="text-align:right;">
25
</td>
<td style="text-align:right;">
8
</td>
</tr>
<tr>
<td style="text-align:right;">
87
</td>
<td style="text-align:right;">
56
</td>
<td style="text-align:right;">
42
</td>
<td style="text-align:right;">
25
</td>
</tr>
<tr>
<td style="text-align:right;">
98
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
9
</td>
</tr>
<tr>
<td style="text-align:right;">
101
</td>
<td style="text-align:right;">
29
</td>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
12
</td>
</tr>
</tbody>
</table>

``` r
m <- lm(y ~ x1 + x2 + x3, data = df)
summary(m)
```

    ##
    ## Call:
    ## lm(formula = y ~ x1 + x2 + x3, data = df)
    ##
    ## Residuals:
    ##       1       2       3       4       5
    ##  1.2215 -0.2758  0.6851  0.4866 -2.1174
    ##
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)  
    ## (Intercept) 119.1171     3.1797  37.461    0.017 *
    ## x1           -0.1837     0.1392  -1.319    0.413  
    ## x2           -0.8424     0.2009  -4.193    0.149  
    ## x3            0.5146     0.3579   1.438    0.387  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ##
    ## Residual standard error: 2.6 on 1 degrees of freedom
    ## Multiple R-squared:  0.9831, Adjusted R-squared:  0.9322
    ## F-statistic: 19.34 on 3 and 1 DF,  p-value: 0.1653

Based on the results, with this small of a sample size, the model and variables are not statistically significant. The *r*<sup>2</sup> shows that the model explains almost all of the variance in *y*. Variables *x*<sub>1</sub> and *x*<sub>2</sub> have a negative relationship with *y* while *x*<sub>3</sub> has a postive relationship.

``` r
new <- data.frame(x1 = 33, x2 = 29, x3 = 13)
predict(m, newdata = new)
```

    ##       1
    ## 95.3156

The above code calculates the preditced value for the inputs shown in the problem.

References
==========

Black, K. (2016). *Business statistics : for contemporary decision making*. Hoboken: Wiley.
