# 19-Neural_Network_Charity_Analysis-lukeperrin



## Overview

This set of notebooks looks to implement neural networks to assess the potential risk or benefit of investing in different organizations. Alphabet Soup has funded many organizations and has tracked their investments, beneficiary organizations, and those organizations’ outcomes in a large and complex dataset ( [charity_data.csv](data/charity_data.csv) ).

Using this dataset, alongside neural network data analysis tools, the notebooks hope to predict whether investing in certain classifications of organizations generally yields more optimal outcomes. 

*N.B. The notebooks were written to be used within the Google Colab notebook environment due to issues getting Tensorflow to run locally in Jupyter due to Mac M1 hardware limitations.*



## Results

- ### Data Preprocessing

  - **What variable(s) are considered the target(s) for your model?**

    *The primary target for the model was the `IS_SUCCESSFUL` value (boolean), representative of the outcome of interest.*

  - **What variable(s) are considered to be the features for your model?**

    *All columns that were kept in the analyzed dataframe (i.e. all but `EIN` and `NAME`) are considered to be features for this model. Namely these are:*

    `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`, and `IS_SUCCESSFUL`.

  - **What variable(s) are neither targets nor features, and should be removed from the input data?**

    *As mentioned above, `EIN` and `NAME` were removed from the input data as they do not offer any meaningful value to the dataset and model.*

- ### Compiling, Training, and Evaluating the Model

  - **How many neurons, layers, and activation functions did you select for your neural network model, and why?**

    *I decided to replicate the module’s strategy and ratio of these quantities demonstrated in  [DeepLearning_Tabular_withCheckpoints.ipynb](practice_notebooks/DeepLearning_Tabular_withCheckpoints.ipynb). Using these similar ratios, I scaled their values to account for the dataset size.* 

  - **Were you able to achieve the target model performance?**

    *Unfortunately, my model performed just short of the 75% goal, only reaching 72.64%.*

  - **What steps did you take to try and increase model performance?**

    *I made four attempts to optimize the model, only learning of three ways that inhibited its performance. These optimization strategies (found in the [optimization_attempts](optimization_attempts) directory ) are tabulated below and in comparison to the original model ( [AlphabetSoupCharity.ipynb](AlphabetSoupCharity.ipynb) ):*

    | Optimization | Changed Parameter                          | Change from original model                                   |
    | ------------ | ------------------------------------------ | ------------------------------------------------------------ |
    | 1            | `epochs`                                   | x5  (`epochs=500`)                                           |
    | 2            | “counts” for buckets                       | /100 (`counts=50`)                                           |
    | 3            | activation functions for the hidden layers | used `“selu”` instead of `“relu”`                            |
    | 4            | Input data                                 | additional dropped columns, changed bucketed data from `APPLICATION_TYPE` and `CLASSIFICATION` to `USE_CASE` and `INCOME_AMT`; and dropped all `INCOME_AMT = “0”` |

## Summary

The results of the four optimizations mentioned above are tabulated likewise below:

| Optimization/Attempt | Loss       | Accuracy   |
| -------------------- | ---------- | ---------- |
| 1                    | 0.5906     | 0.7254     |
| 2                    | 0.5590     | 0.7285     |
| 3                    | 0.5531     | 0.7243     |
| 4                    | 0.7788     | 0.5937     |
| **ORIGINAL**         | **0.5559** | **0.7264** |

From the results above, it is evident that the only successful optimization attempt was in increasing the number of epochs from 100 to 500. Regardless, none of these optimizations met the goal of 75% accuracy.

Would I restrategize building a model for this dataset, I would likely spend more time transforming some of the quantitative features into boolean metrics of successful outcomes (e.g. `INCOME_AMT > ASK_AMT*100`) and bucketing the categorical features into boolean values in order to inform a random forest decision tree model. This seems like a simpler path to potentially higher accuracy values.

A neural network model would likely perform better than the best random forest model, but due to the complexity and variety of the input and its parameters, random forest seems to be less involved than neural networks for this dataset. If I were to keep attempting to use neural networks for this purpose, I would likely defer to an optimization/pruning toolkit (like mentioned [here](https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html)) to simplify some of the complexities.
