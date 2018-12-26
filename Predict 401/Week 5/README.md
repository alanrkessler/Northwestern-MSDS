Week 5 Practice
================
Alan Kessler

This problem is from Chapter 8: Statistical Inference: Estimation for Single Populations, Section 8.4 Estimating the Population Variance [(Black, 2016, p. 254)](#ref).

> 8.38
> A manufacturing plant produces steel rods. During one production run of 20,000 such rods, the specifications called for rods that were 46 centimeters in length and 3.8 centimeters in width. Fifteen of these rods comprising a random sample were measured for length; the resulting measurements are shown here. Use these data to estimate the population variance of length for the rods. Assume rod length is normally distributed in the population. Construct a 99% confidence interval. Discuss the ramifications of the results.

The following packages are used in this problem:

``` r
library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
library(ggplot2)

# Adjust my output to avoid scientific notation
options(scipen=999)
```

The Data
--------

The data originally from the problem is manually entered into vectors to create a data frame.

``` r
# Load problem data
DataVector <- c(44, 47, 43, 46, 46, 45, 43, 44, 47, 46, 48, 48, 43, 44, 45)

# Modify to present table
PresVector <- paste(DataVector, "cm", sep=" ")
ProbDataPrint <- t(data_frame(PresVector[1:5],
                              PresVector[6:10],
                              PresVector[11:15]))
rownames(ProbDataPrint) <- NULL

kable(ProbDataPrint, format="html") %>%
  kable_styling(full_width=FALSE)
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
<tbody>
<tr>
<td style="text-align:left;">
44 cm
</td>
<td style="text-align:left;">
47 cm
</td>
<td style="text-align:left;">
43 cm
</td>
<td style="text-align:left;">
46 cm
</td>
<td style="text-align:left;">
46 cm
</td>
</tr>
<tr>
<td style="text-align:left;">
45 cm
</td>
<td style="text-align:left;">
43 cm
</td>
<td style="text-align:left;">
44 cm
</td>
<td style="text-align:left;">
47 cm
</td>
<td style="text-align:left;">
46 cm
</td>
</tr>
<tr>
<td style="text-align:left;">
48 cm
</td>
<td style="text-align:left;">
48 cm
</td>
<td style="text-align:left;">
43 cm
</td>
<td style="text-align:left;">
44 cm
</td>
<td style="text-align:left;">
45 cm
</td>
</tr>
</tbody>
</table>

Estimate of the Population Variance
----------------------------------------------

``` r
sampleVariance <- var(DataVector)
```

Returns an unbiased estimate of **3.0667**.

99% Condifence Interval of Population Variance
----------------------------------------------

I use the following formaula to construct the 99% confidnce interval:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{(n-1)s^2}{X^2_{\alpha/2}}\leq\sigma^2\leq\frac{(n-1)s^2}{X^2_{1-\alpha/2}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\frac{(n-1)s^2}{X^2_{\alpha/2}}\leq\sigma^2\leq\frac{(n-1)s^2}{X^2_{1-\alpha/2}}" title="\frac{(n-1)s^2}{X^2_{\alpha/2}}\leq\sigma^2\leq\frac{(n-1)s^2}{X^2_{1-\alpha/2}}" /></a>

``` r
n <- length(DataVector)
lower <- (n-1)*sampleVariance / qchisq(0.995, n-1)
upper <- (n-1)*sampleVariance / qchisq(1-0.995, n-1)
```

This returns a confidence interval of **(1.3708, 10.5366)**.

Ramifications
-------------

These results can help determine if the process creating these rods meet specified tolerances. For example, we could take a conservative point of view and use the upper end of the confidence interval to calculate the probability that a given rod will exceed a difference threshold.

For a tolerance threshold of 3cm, the calculation would look like this:

``` r
pOutsideTolerance <- 2*pnorm(43, 46, sqrt(upper))
```

So conservatively, we could estimate that steel rods are produced out of threshold with approximately a probability of **0.3554**. For an industrial application, such a value would likely not be acceptable.

References
----------

Black, K. (2016). *Business statistics : for contemporary decision making*. Hoboken: Wiley.
