from itertools import product
i = 8
data = [
        "version_1.csv", 
        "version_2.xlsx", 
        "version_3.txt", 
        "version_4.json",
        "version_5.tsv"
      ]

tech = [
        "Random Subsampling", 
        "K-fold Cross Validation", 
        "Leave-one-out CrossValidation", 
        "Leave-p-out Cross Validation", 
        "Stratified Cross Validation", 
        "Stratified Shuffle Split", 
        "Bootstrap"
      ]

A, B, C = list(product(data, tech, tech))[i]
print("A:", A, "B:", B, "C:", C)