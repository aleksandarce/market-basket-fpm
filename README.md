# Market Basket Analysis: Apriori vs Eclat vs FP-Growth

A comparison of three frequent-pattern mining algorithms on grocery orders: what each one finds and how long it takes. Apriori and FP-Growth come from `mlxtend`, and **Eclat is my own implementation** using vertical TID-sets.

This is the pattern-mining part of a CSE4063 Fundamentals of Data Mining project under the supervision of Dr. Mustafa Agaoglu at Marmara University (spring 2026), done by a group of three (Hikmet Topak, Yusuf Islam Altunkaynak and me). The full project also covers clustering. This repo contains only the pattern-mining part.

## Data and settings

- **Dataset:** the Kaggle "E-commerce Dataset for Predictive Marketing 2023" (about 2M product lines, 200,000 orders, 21 departments, 134 products). It isn't included here because of its size.
- **Baskets:** each order is one basket. Repeated products in a basket are counted once, baskets with fewer than 2 items are dropped, and a random sample of 50,000 baskets is used (seed 42).
- **Parameters:** minimum support 1%, minimum confidence 30%, itemsets of up to 3 items.
- **Two levels:** items are either the 21 departments, or the 50 most frequent products.

## Results

| Level | Algorithm | Runtime | Frequent itemsets | Rules |
|---|---|---|---|---|
| Departments | Apriori | 0.12 s | 607 | 1,271 |
| Departments | FP-Growth | 0.30 s | 607 | 1,271 |
| Departments | Eclat (own) | 0.35 s | 607 | 1,271 |
| Top 50 products | Apriori | 1.83 s | 2,130 | 2,448 |
| Top 50 products | FP-Growth | 2.22 s | 2,130 | 2,448 |
| Top 50 products | Eclat (own) | 1.41 s | 2,130 | 2,448 |

- **All three algorithms find identical itemsets and rules at both levels.** That is the check that my Eclat implementation is correct.
- **The ranking changes between levels:** Apriori is fastest on departments and Eclat on products. With runtimes this short, I can't draw a general conclusion about which algorithm is best. Only the correctness result is solid.

**Strongest rules by lift**

| Level | If a basket has | it also tends to have | Confidence | Lift |
|---|---|---|---|---|
| Departments | pantry, personal care | household | 36.8% | 2.31 |
| Products | dry pasta, packaged cheese | pasta sauce | 33.6% | 4.83 |

A lift of 4.83 means those baskets are about 4.8 times as likely to contain pasta sauce as an average basket.

## Files

```
fpm.py      # loading, cleaning, basket building, the three algorithms, rules, plots
results/    # runtime tables, top rules and plots for both levels
```

## Run

```bash
pip install -r requirements.txt
# put the dataset at data/ecommerce.csv
python fpm.py
```

Settings (level, sample size, support, confidence) are in the configuration block at the top of `fpm.py`. Set `LEVEL` to `"category"` for departments or `"product"` for products. Results are written to `outputs/`, including the full itemset and rule lists, which aren't committed here.
