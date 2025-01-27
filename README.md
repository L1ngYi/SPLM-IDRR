# Soft Prompting with Logical Semantics Enhancement for Implicit Discourse Relation Recognition

The Code for the  "**A Soft Prompt Learning Method for Implicit Discourse Relation Recognition**".

## **Dependence Installation**

```
pip install -r requirements.txt
```

If you need more info about dependence,you can read **piplist.txt**.

## Data

#### Pre-training data

The pre-training data comes from [PLSE](https://github.com/lalalamdbf/PLSE_IDRR).

#### Prompt-tuning data

We use PDTB 2.0  to evaluate our models. If you have bought data from LDC, please put the PDTB data in *src/data/pdtb2* respectively.

## Data Preprocessing

- run the following commands, respectively:

```
unzip ./src/data/explicit_data/explicit_data.zip -d ./src/data/explicit_data
```

```
sh ./scripts/data_preprocess_pdtb2_4.sh
```

```
sh ./scripts/data_preprocess_pdtb2_11.sh
```

## Pre-training

-  run the following command:


```
sh ./scripts/pretrain_plse.sh
```

## Prompt-tuning

- For 4-way classification on PDTB 2.0, run the following command:

```
sh ./scripts/train_pdtb_4.sh
```

- For 11-way classification on PDTB 2.0, run the following command:

```
sh ./scripts/train_pdtb_11.sh
```

## Change template

Find the code file in the following path

```
./src/prompt-tuning/prompt_config.py
```

Modify the code at line 38

```python
text='{"soft"}{"soft"}{"soft"}{"soft"}{"placeholder":"text_a"}{"mask"}{"placeholder":"text_b"}'
```

## Change learning rate

Find the scripts file in the following path

```
./scripts/train_pdtb_4.sh
```

Modify the code

```
  --p_tuning_learning_rate 1e-3\
  --learning_rate 1e-5 
```

## Bibliography

If you find this repo useful, please cite our paper.
