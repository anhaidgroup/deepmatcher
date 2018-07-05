# Datasets for DeepMatcher paper

Datasets listed in this page were used for the experimental study in [Deep Learning for Entity Matching](http://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf) published in SIGMOD 2018. Each data instance in each dataset is a labeled tuple pair, where each tuple pair comes from the 2 tables being matched, say table A and table B. We assume that both the tables being matched have the same schema.

The table below summarizes all the datasets. Here's a brief description of some of the columns:
- Size: Number of labeled tuple pairs in the dataset.
- \# Pos.: Number of positive instances, i.e., tuple pairs marked as a match in the dataset.
- \# Attr.: Number of attributes in the tables being matched (note that both tables have same schema)

The "Browse" links point to the preprocessed versions of the datasets which were used for experiments. The "Download" links provide a compressed zip of the preprocessed data. The "Raw" links provide a compressed zip of the raw unprocessed data obtained from data source. The "Info" links point to the dataset details.

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Dataset</th>
      <th>Domain</th>
      <th>Size</th>
      <th># Pos.</th>
      <th># Attr.</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=7> Structured</td>
      <td>BeerAdvo-RateBeer</td>
      <td>beer</td>
      <td>450</td>
      <td>68</td>
      <td>4</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/beer_exp_data.zip">Download</a> | 
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/beer_raw_data.zip">Raw</a> |
      <a href="#beeradvo-ratebeer">Info</a></td>
    </tr>
    <tr>
      <td>iTunes-Amazon<sub>1</sub></td>
      <td>music</td>
      <td>539</td>
      <td>132</td>
      <td>8</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/itunes_amazon_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/itunes_amazon_raw_data.zip">Raw</a> |
      <a href="#itunes-amazon">Info</a></td>
    </tr>
    <tr>
      <td>Fodors-Zagats</td>
      <td>restaurant</td>
      <td>946</td>
      <td>110</td>
      <td>6</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/fodors_zagat_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/fodors_zagat_raw_data.zip">Raw</a> |
      <a href="#fodors-zagats">Info</a></td>
    </tr>
    <tr>
      <td>DBLP-ACM<sub>1</sub></td>
      <td>citation</td>
      <td>12,363</td>
      <td>2,220</td>
      <td>4</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/dblp_acm_raw_data.zip">Raw</a> |
      <a href="#dblp-acm">Info</a></td>
    </tr>
    <tr>
      <td>DBLP-Scholar<sub>1</sub></td>
      <td>citation</td>
      <td>28,707</td>
      <td>5,347</td>
      <td>4</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_raw_data.zip">Raw</a> |
      <a href="#dblp-scholar">Info</a></td>
    </tr>
    <tr>
      <td>Amazon-Google</td>
      <td>software</td>
      <td>11,460</td>
      <td>1,167</td>
      <td>3</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_raw_data.zip">Raw</a> |
      <a href="#amazon-google">Info</a></td>
    </tr>
    <tr>
      <td>Walmart-Amazon<sub>1</sub></td>
      <td>electronics</td>
      <td>10,242</td>
      <td>962</td>
      <td>5</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_raw_data.zip">Raw</a> |
      <a href="#walmart-amazon">Info</a></td>
    </tr>
    <tr>
      <td rowspan=2>Textual</td>
      <td>Abt-Buy</td>
      <td>product</td>
      <td>9,575</td>
      <td>1,028</td>
      <td>3</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_raw_data.zip">Raw</a> |
      <a href="#abt-buy">Info</a></td>
    </tr>
    <tr>
      <td>Company</td>
      <td>company</td>
      <td>112,632</td>
      <td>28,200</td>
      <td>1</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/company_exp_data.zip">Download</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/company_raw_data.zip">Raw</a> |
      <a href="#company">Info</a></td>
    </tr>
    <tr>
      <td rowspan=4>Dirty</td>
      <td>iTunes-Amazon<sub>2</sub></td>
      <td>music</td>
      <td>539</td>
      <td>132</td>
      <td>8</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/iTunes-Amazon/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/iTunes-Amazon/dirty_itunes_amazon_exp_data.zip">Download</a> |
      <a href="#itunes-amazon-1">Info</a></td>
    </tr>
    <tr>
      <td>DBLP-ACM<sub>2</sub></td>
      <td>citation</td>
      <td>12,363</td>
      <td>2,220</td>
      <td>4</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-ACM/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-ACM/dirty_dblp_acm_exp_data.zip">Download</a> |
      <a href="#dblp-acm-1">Info</a></td>
    </tr>
    <tr>
      <td>DBLP-Scholar<sub>2</sub></td>
      <td>citation</td>
      <td>28,707</td>
      <td>5,347</td>
      <td>4</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-GoogleScholar/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/DBLP-GoogleScholar/dirty_dblp_scholar_exp_data.zip">Download</a> |
      <a href="#dblp-scholar-1">Info</a></td>
    </tr>
    <tr>
      <td>Walmart-Amazon<sub>2</sub></td>
      <td>electronics</td>
      <td>10,242</td>
      <td>962</td>
      <td>5</td>
      <td><a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/Walmart-Amazon/exp_data/">Browse</a> |
      <a href="http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/Walmart-Amazon/dirty_walmart_amazon_exp_data.zip">Download</a> |
      <a href="#walmart-amazon-1">Info</a></td>
    </tr>
  </tbody>
</table>

**Notes:** 
- The `tableA.csv` and `tableB.csv` files in the provided experimental data may not directly correspond to the original tables being matched. You can think of `tableA.csv` as containing all the "left" tuples and `tableB.csv` as containing all the "right" tuples. This is done so as to distribute the data in a reasonably compact but readable form.
- The dirty EM datasets were generated from the preprocessed versions of the corresponding structured EM datasets. As a result there are no raw versions of the dirty datasets.

## Batch download links:
### Preprocessed Data
- [Download all preprocessed structured datasets](http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured.zip)
- [Download all preprocessed textual datasets](http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual.zip)
- [Download all preprocessed dirty datasets](http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty.zip)
### Raw Data
- [Download all raw structured datasets](http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/StructuredRaw.zip)
- [Download all raw textual datasets](http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/TextualRaw.zip)

# Dataset Details

## Structured

### BeerAdvo-RateBeer

This dataset contains beer data from BeerAdvocate and RateBeer and was obtained from [here](https://sites.google.com/site/anhaidgroup/useful-stuff/data). It was created by students in the CS 784 data science class at UW-Madison, Fall 2015, as a part of their class project. To create the data set, students

1. Crawled HTML pages from the two websites
2. Extracted tuples from the HTML pages to create two tables, one per site
3. Performed blocking on these tables (to remove obviously non-matched tuple pairs), producing a set of candidate tuple pairs
4. Took a random sample of pairs from the above set and labeled the pairs in the sample as "match" / "non-match".

For the purpose of performing experiments for this work, we split the set of labeled tuple pairs into 3 sub-sets, i.e., train, validation, and test, with ratio 3:1:1.

### iTunes-Amazon

This dataset contains music data from iTunes and Amazon and was obtained from [here](https://sites.google.com/site/anhaidgroup/useful-stuff/data). This was also created by students in the CS 784 data science class at UW-Madison. The dataset was created in the same manner as [BeerAdvo-RateBeer](#beeradvo-ratebeer).

### Fodors-Zagats

This dataset contains restaurant data from Fodors and from Zagat and was obtained from [here](http://www.cs.utexas.edu/users/ml/riddle/data.html). The original dataset obtained from the source contained two tables, one each for Fodors and Zagat, and a list of golden matches indicating which tuple pairs referred to the same restaurant. To create the version of the dataset used in our experiments which contain both matches and non-matches, we use the following procudere:

1. Given the two tables (tableA.csv & tableB.csv), perform dataset specific blocking to obtain a candidate set C
2. For each tuple pair in set C, if the pair is present in the golden matches file (gold.csv), mark the pair as a match. Else, mark the pair as a non-match.
3. Randomly split the labeled candidate set C into 3 sets, i.e., train, validation, and test, with ratio 3:1:1.

### DBLP-ACM

This dataset contains bibliographic data from DBLP and ACM and was obtained from [here](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). The original dataset obtained from the source contained two tables, and a list of golden matches. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#fodors-zagats).

### DBLP-Scholar

This dataset contains bibliographic data from DBLP and Google Scholar and was obtained from [here](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). The original dataset obtained from the source contained two tables, and a list of golden matches. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#fodors-zagats).

### Amazon-Google

This dataset contains product data from Amazon and Google and was obtained from [here](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). The original dataset contained two tables, and a list of golden matches. Further, the original dataset contained one additional attribute "description" which contained long blobs of text. This attribute was removed so as to use this as a structured dataset. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#fodors-zagats).

### Walmart-Amazon

This dataset contains product data from Walmart and Amazon and was obtained from [here](https://sites.google.com/site/anhaidgroup/useful-stuff/data). The original dataset contained two tables, and a list of golden matches. Further, the original dataset contained one additional attribute "proddescrlong" which contained long blobs of text. This attribute was removed so as to use this as a structured dataset. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#fodors-zagats).

## Textual

### Abt-Buy

This dataset contains product data from Abt.com and Buy.com and was obtained from [here](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution). The original dataset contained two tables, and a list of golden matches. To create the version of the dataset used in our experiments we used the same procedure as in the case of [Fodors-Zagats](#fodors-zagats).

### Company

This dataset consists of pairs `(a,b)`, where `a` is the text of a Wikipedia page describing a company and `b` is the text of a company’s homepage. We created matching pairs in this dataset by crawling Wikipedia pages describing companies, then following company URLs in those pages to retrieve company homepages. To generate the non-matching pairs, for each matching pair `(a,b)`, we fix `a` and form three negative pairs <code>(a,b<sub>1</sub>)</code>, <code>(a,b<sub>2</sub>)</code>, and <code>(a,b<sub>3</sub>)</code> where b<sub>1</sub>, b<sub>2</sub>, and b<sub>3</sub> are the top-3 most similar pages other than `b` in the company homepage collection, calculated based on Okapi BM25 rankings.

In the case of company home pages and wikipedia pages, the title of the document often contains the name of the company. Further the document often begins by mentioning the company name. In general matching web documents in the wild is a complex task and such hints are unlikely to be present. To simulate a general web document matching scenario, we remove the first 20 tokens from the parsed HTML of each document and consider only the rest of the document for the purpose of matching. This essentially removes the document title and also any mentions of the company name from the beginning of the first paragraph.

## Dirty

### iTunes-Amazon

This dataset contains music data from iTunes and Amazon and was obtained by modifying the [structured iTunes-Amazon](#itunes-amazon) dataset to simulate dirty data. Specifically, for each attribute other than "title", we randomly moved each value to the attribute "title" in the same tuple with 50% probability. This simulates a common kind of dirty data seen in the wild while keeping the modifications simple.

### DBLP-ACM

This dataset contains bibliographic data from DBLP and ACM and was obtained by modifying the [structured DBLP-ACM](#dblp-acm) dataset to simulate dirty data. The procedure for generating this dataset is the same as that for [dirty iTunes-Amazon](#itunes-amazon-1).

### DBLP-Scholar

This dataset contains bibliographic data from DBLP and Google Scholar and was obtained by modifying the [structured DBLP-Scholar](#dblp-scholar) dataset to simulate dirty data. The procedure for generating this dataset is the same as that for [dirty iTunes-Amazon](#itunes-amazon-1).

### Walmart-Amazon

This dataset contains product data from Walmart and Amazon and was obtained by modifying the [structured Walmart-Amazon](#walmart-amazon) dataset to simulate dirty data. The procedure for generating this dataset is the same as that for [dirty iTunes-Amazon](#itunes-amazon-1).
