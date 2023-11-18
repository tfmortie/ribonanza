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


#### AB tests brainstorm 15/11
- [ ] data = split
- [ ] mse vs mae als loss
- [ ] samplen van ys obv reactivity error
- [ ] inverse balance loss weighten obv reactivity error
- [ ] multi-output regression or 2 sep models
- [ ] DNA language model pre-training or not
- [ ] 


#### Gaetan's notes

- [ ] it appears a new version of the [data](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data) is available.
- [ ] `reactivity_error`: Ideas to use: (1) Model a distribution instead of point estimates (concern: will this really help MAE performance?), (2) Use to weigh loss function?, (3) sample labels according to dist?
- [ ] Signal to noise: Ideas to use: pre-train a model using all data and then fine-tune on SN-filtered-only data?
- [ ] I counted: all sequence ids are repeated in `train_data.csv`, meaning that every sequence has labels for both targets. Train a multi-output regression model? (concern: Will a single model be better than two "specialized ones"?)
- [ ] `Ribonanza_bpp_files`: Is information on basepair interaction ==> add to attention matrix?
- [ ] `eterna_openknot_metadata`: See below
- [ ] `sequence_libraries`: See below
- [ ] `supplementary_silico_predictions`: For a significant percentage (>130 000), we have predicted folding data from external model: Ideas: (1) we can add this to inputs (extra channels aside from ATCG), (2) pre-train a model on this task, fine-tune on our task.
- [ ] Biologically: Does using MSA inputs make sense here?
- [ ] Dealing with larger sequences: (1) self-supervised pre-training task?, (2) sometimes just concat two seq during training (this does not make sense biologically but might help model)
- [ ] Training and val sequences duplicates more noise less noise.



Sequences in `eterna_openknot_metadata`:
- 2001 Positives240-2000.tsv - matched fasta file in sequence lib    
- 2730 puzzle_11318423_RYOP50_with_description.tsv - matched fasta file in sequence lib   
- 2174  puzzle_11387276_RYOP90_with_description.tsv - matched fasta file in sequence lib   
- 120001 puzzle_11627601_with_descriptions_PLUS_RFAM.tsv - matched fasta file in sequence lib   
- 250419 puzzle_11836497_with_description.tsv - no matched file ??? Check if the sequences here are in the data somewhere?
- 336820 'puzzle 12378132.tsv' - no matched file ??? Check if the sequences here are in the data somewhere?

Sequences in `sequence_libraries`:
- 1000000 1Mround2_all.fasta - NO METADATA
- 15000 230601_GPN_library_RCK_edit.fasta - NO METADATA
- 1000000 DasLabBigLib1M_sequences_SUBLIBRARY.fa - NO METADATA
- 6000 FINAL390_clean.fasta - NO METADATA
- 2000 Positives240-2000.tsv.fa - HAS METADATA -- same as in eterna_openknot_metadata
- 2729 pseudoknot50_puzzle_11318423.tsv.RNA_sequences.fa - HAS METADATA -- same as in eterna_openknot_metadata, nothing interesting though
- 2173 pseudoknot90_puzzle_11387276.tsv.RNA_sequences.fa - HAS METADATA -- same as in eterna_openknot_metadata, nothing interesting though
- 120000 puzzle_11627601_Final120k_from_ETERNA.tsv_SUBLIBRARY.fa - HAS METADATA -- same as in eterna_openknot_metadata
- 2499 SL5_library_with_rescues_control.fa - NO METADATA

Sequences in `supplementary_silico_predictions`:
- 15000 GPN15k_silico_predictions
- 2729 PK50_silico_predictions
- 2173 PK90_silico_predictions
- 120000 R1_silico_predictions