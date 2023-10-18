# Ribonanza

Code for Stanford Ribonanza RNA Folding Kaggle competition

## Roadmap

### First iteration

* Assume independence over DMS and 2A3 reactivity
* Use simple many-to-one models that predict reactivity for both DMS and 2A3 on sequences that pass the S/N filter
    - Consider k-fold cross-validation for all sequences that occur once
    - Consider fixed splits where high S/N sequences are added to the training and low S/N sequences to the test set 
* Consider different context windows and apply ensembling
* Explore other metadata

### Second iteration 

* Use transformers
* ...