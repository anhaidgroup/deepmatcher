# Datasets

## Structured

### Fodors-Zagats

This dataset contains restaurant data from Fodors and from Zagat and was obtained from [here](http://www.cs.utexas.edu/users/ml/riddle/data.html). It has 6 attributes which are name, address, city, phone number, type, and restaurant class. It contains 946 labeled tuple pairs of which 68 (7.2%) are matches. The original dataset obtained from the source contained two tables, one each for Fodors and Zagat, and a list of golden matches indicating which tuple pairs referred to the same restaurant. To create the version of the dataset used in our experiments which contain both matches and non-matches, we use the following procudere:

1. Given the two tables (tableA.csv & tableB.csv), perform dataset specific blocking to obtain a candidate set C
2. For each tuple pair in set C, if the pair is present in the golden matches file (gold.csv), mark the pair as a match. Else, mark the pair as a non-match.
3. Randomly split the labeled candidate set C into 3 sets, i.e., train, validation, and test, with ratio 3:1:1.

[Download Fodors-Zagats dataset]()

### DBLP-ACM

This dataset contains bibliographic data from DBLP and ACM and was obtained from [here](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). It has 4 attributes which are title, authors, publication venue, and year. It contains 12,363 labeled tuple pairs of which 2,220 (17.6%) are matches. The original dataset obtained from the source contained two tables, one each for DBLP and ACM, and a list of golden matches indicating which tuple pairs referred to the same publication. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#Fodors-Zagats).

[Download DBLP-ACM dataset]()

### Amazon-Google

This dataset contains product data from Amazon and Google. It has 3 attributes which are product title, manufacturer, and price. It contains 11,460 labeled tuple pairs of which 1,167 (10.2%) are matches. The original dataset contained one additional attribute "description" which was removed to use this as a structured dataset. Further the original dataset obtained from the source contained two tables, one each for Amazon and Google, and a list of golden matches indicating which tuple pairs referred to the same product. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#Fodors-Zagats).

[Download Amazon-Google dataset]()

