# Ribonanza

Code for Stanford Ribonanza RNA Folding Kaggle competition

## Roadmap

### First iteration

* Assume independence over DMS and 2A3 reactivity
* Use simple many-to-many models that predict reactivity for both DMS and 2A3 on sequences that pass the S/N filter
    - Consider k-fold cross-validation for all sequences that occur once
    - Consider fixed splits for sequences with multiple occurences: high S/N sequences are added to the training and low S/N sequences to the test set 
* Explore other metadata

### Second iteration 

* Use transformers