# Task: Forward and Backward Stepwise Selection

In this task, I implemented forward and backward stepwise selection methods to create and compare predictive models.
The task is based on **ITSL 6.6 Applied 8 (a-d)** and can be found in **ITSL (Python edition)** on page 286 (PDF page
294). I rephrased the task a bit for this readme, but the main idea is the same.

## Details of the Task:

### (a) Generate Predictors and Noise

I generated predictors and noise to simulate data:

- Created a random number generator.
- Generated:
    - A predictor vector **X** of length `n = 100`.
    - A noise vector **ε** (epsilon) of length `n = 100`.

### (b) Generate the Response Variable

The response vector **Y** of length `n = 100` was constructed using the following model:

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \varepsilon
$$

The coefficients were selected by me so that they had more or less similar affect on the target variable.
### (c) Forward Stepwise Selection

Forward stepwise selection was used to identify the best model from the predictors:

$$
X, X^2, X^3, \dots, X^{10}
$$

The selection was based on **Mallows' $C_p$** criterion.

### (d) Backward Stepwise Selection

Similarly to (c), Backward Stepwise Selection was used to identify the best model from the predictors.


### (f) Forward Stepwise Selection with only one power monomial(no lasso)
Generate a response vector **Y** according to the following model:

$$
Y = \beta_0 + \beta_7 X^7 + \varepsilon
$$

Then, perform **Forward Stepwise Selection** on the data and discuss and compare the results obtained.

---

### Report on the work done & some observations

- The results of FSS and BSS alone differ quite a lot. As those are greedy algorithms, they usually fall into some
local minimums and cannot proceed anymore. As there were only 4 true feature coefficients for the target variable out
of 10 features in total, the FSS performed better on average; however, even it had some room for improvement.

- I decided to extend the task a little bit and try to apply FSS & BSS together - one after another.
However, in most of the cases the result was not improving at all - the coefficients stayed the same.
Most probably it is because I just was not lucky enough or because the data is synthetically
generated. However, probably I should have done not only 1 iteration of FSS+BSS, but several, until they
stop improving. 

- When first implementing the algorithm, I tried to use the r2_score metric instead of the Mallow's
Cp score. So, the FSS algo was working very well, while the BSS mostly refused to remove features from its selection.
In such a way I found out the use of Mallow's metric - it not only awards the model when it performs well, but punishes
it for having too many features and being overcomplicated.

- Discussion on task F: when having only one true term in the true mathematical model, the FSS performs
perfectly. I believe this is because the FSS sees from the very beginning which feature brings the most
value on the very first iteration and just uses it. A bit more interesting picture can be seen with
BSS: when putting the `min_amount_of_features`=1, the model achieves absolutely the same result, as the FSS.
However, if we limit `min_amount_of_features`=10 (so just linear regression without any BSS), the result gets worse.
This is why I conclude, that FSS and BSS are very powerful tools for including or excluding some 
features from the model.

- When choosing tiny values of parameters BETA, the FSS & BSS is no more able to clearly determine which 
features matter because of the noise. So, for example, `BETA_7`=0.01 already makes FSS think that the term with
degree 9 is more important than the true one. Unexpectedly, it actually produces slightly better result
than the model with the true feature. However, one should note that even so the FSS and BSS perform much better
than using just linear regression with all the features included.

- ChatGPT is not the best companion while trying to understand complex concepts & algos in ML, and one
should not trust it very easily. And the textbooks are actually not so scary:)

### Requirements

- Python 3.12 or higher.
- Dependencies (e.g., numpy, scikit-learn, scipy) can be installed using `pip install -r requirements.txt`.

### References

**Task Source:** ITSL 6.6 Applied 8 (a-d), ITSL (Python edition), page 286 (PDF page 294).  
