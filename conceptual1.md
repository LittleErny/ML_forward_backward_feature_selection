# ITSL 6.6 Conceptual 1 solution

## Task:

We perform `best subset`, `forward stepwise`, and `backward stepwise selection` on a single data
set. For each approach, we obtain p + 1 models, containing 0, 1, 2, ... , p predictors. Explain
your answers:

(a) Which of the three models with k predictors has the smallest training RSS?

(b) Which of the three models with k predictors has the smallest test RSS?

(c) True or False:
    
1. The predictors in the k-variable model identified by forward stepwise are a subset of
    the predictors in the (k +1)-variable model identified by forward stepwise selection.
2. The predictors in the k-variable model identified by backward stepwise are a subset
of the predictors in the (k + 1)-variable model identified by backward stepwise
selection.
3. The predictors in the k-variable model identified by backward stepwise are a sub-
set of the predictors in the (k + 1)-variable model identified by forward stepwise
selection.
4. The predictors in the k-variable model identified by forward stepwise are a subset
of the predictors in the (k + 1)-variable model identified by backward stepwise
selection.
5. The predictors in the k-variable model identified by best subset are a subset of the
predictors in the (k + 1)-variable model identified by best subset selection.

## Solution:

(a) Definitely, the `Best Subset` will have the smallest RSS on a training dataset. Among the
listed algorithms, `Best Subset` iterates through all possible combinations of parameters and
basically finds the best one, which makes it fit to the training dataset as much as possible.

(b) In my opinion, assuming that k is fixed, the FSS and BSS should have the smallest test RSS. 
As discussed, `Best Subset` iterates 
through all possible subsets of features and chooses the best one. However, it is possible that 
such a strong connection to the training dataset will imply to the overfitting - it will just 
fit training dataset very well, describing not the general tendencies in the data but exact
behaviour of training dataset and noise. This is why my opinion is the follows: 

- For models with big amount of predictors - FSS and BSS will perform the best. This is because 
with the increase of amount of predictors the model has more degrees of freedom, and, therefore,
chances to fit the training dataset as much as possible, not describing the general 
trends in the data increase. And while FSS and BSS look for simplest ways to describe 
the data through greedy algorithms,
`Best Selection` tries to find the exact "best" combination, which makes it vulnerable to overfitting.

- However, for the models with smaller amount of predictors, `Best Selection` might give better result.
In my opinion, with small amount of predictors linear regression has not so many chances to fit
the training dataset, and instead of noise it fits to the general trends in the data. This is why the 
problem of overfitting is unlike, and choosing the best predictors using `Best Selection` becomes a 
good option. However, FSS and BSS might perform similarly good - slightly worse, but still worthy.


(c)

1. The predictors in the k-variable model identified by forward stepwise are a subset of the 
predictors in the (k +1)-variable model identified by forward stepwise selection. 

    **Answer:** ***True**. As we only add new predictors to the model but do not remove already added features, the predictors
on the previous step are always included in the set of predictors on the next iteration.*

2. The predictors in the k-variable model identified by backward stepwise are a subset of the 
predictors in the (k + 1)-variable model identified by backward stepwise selection. 

    **Answer:** ***True**. As we only remove predictors from the model but do not add already removed features, the predictors
on the next step are always included in the set of predictors on the previous iteration.*

3. The predictors in the k-variable model identified by backward stepwise are a sub-set of the 
predictors in the (k + 1)-variable model identified by forward stepwise selection.

    **Answer:** ***False**. FSS and BSS are completely different algorithm, iterations of which are not
connected to each other anyhow but the general approach.*

4. The predictors in the k-variable model identified by forward stepwise are a subset of the predictors in 
the (k + 1)-variable model identified by backward stepwise selection.

    **Answer:** ***False**. Again, FSS and BSS are completely different algorithm, iterations of which are not
connected to each other anyhow but the general approach.*

5. The predictors in the k-variable model identified by best subset are a subset of the predictors 
in the (k + 1)-variable model identified by best subset selection.

    **Answer:** ***False**. The fact that some feature A was included in the best model by 
the Best Selection algo with some k does not guarantee that the same feature A will be included on the
next iteration with k+1 predictors.*
