# 🌳 Learning Decision Trees 🌳

This project showcases a Python solution for Assignment 3 of the CS 540 Introduction to Artificial Intelligence course at the *University of Wisconsin-Madison*, adapted from a **CS 20551 Introduction to AI** course at the *Open University of Israel*  (2024a) as mmn (`Matalat Manche`) 18. Originally based on Java code from the assignment, this reimplementation in Python achieved a perfect score! 🎯

<p align = "center">
    <img src="cover.png" width="600"/>
</p>

---

- [📖 Overview](#-overview)
- [🛠 Technical Details](#-technical-details)
  - [`my_DecisionTreeImpl.py`](#my_decisiontreeimplpy)
  - [`my_HW3.py`](#my_hw3py)
  - [`my_Instance.py`](#my_instancepy)
  - [`my_DecisionTree.py` \& `my_DataSet.py`](#my_decisiontreepy--my_datasetpy)
- [🚀 How to Run](#-how-to-run)
- [📈 Example Usage](#-example-usage)
  - [Tennis Data Set](#tennis-data-set)
  - [Loans Data Set](#loans-data-set)
    - [Mode 0](#mode-0)
    - [Mode 1](#mode-1)
    - [Mode 2](#mode-2)
    - [Mode 3](#mode-3)

---

## 📖 Overview

The implementation of the decision tree is a testament to the robustness and versatility of Python, transitioning from Java while aiming to maintain the original code structure and interface integrity.

## 🛠 Technical Details

### `my_DecisionTreeImpl.py`

At the core, this module encapsulates the essence of decision tree learning, adhering closely to the algorithms outlined in academic literature, with a sprinkle of personal insights for optimization. 🤖 The transition from Java's rigid type system to Python's dynamic nature posed challenges but ultimately led to a more flexible and readable codebase.

### `my_HW3.py`

This script is the conductor of the symphony, ensuring each part of the decision tree's implementation plays together harmoniously. Through meticulous refactoring, the script now boasts enhanced clarity and structure, guiding the user through the process seamlessly.

### `my_Instance.py`

A crucial component, this class ensures individual instances are managed efficiently, adopting a strategy that aligns with the data's inherent structure while providing easy access and manipulation capabilities.

### `my_DecisionTree.py` \& `my_DataSet.py`

These modules lay the groundwork for the decision tree's operation, translating complex Java logic into Pythonic elegance. The journey from static typing to dynamic required both creativity and precision, ensuring the algorithm's integrity remains intact.

---

## 🚀 How to Run

Execute your decision tree adventure with the following command, substituting `<F>` with your desired mode of operation:

```bash
python3 my_HW3.py <F> <train-set-file> <test-set-file>

```

Where `<F>` represents one of the following modes:

- `mode 0`: Revel in the mutual information of each attribute at the root.

- `mode 1`: Witness the birth of a decision tree from your training set.

- `mode 2`: Predict the future of your test set with your tree.

- `mode 3`: Validate your tree's wisdom against the test set.

---

## 📈 Example Usage

Dive into the examples to see the decision tree tackle real-world datasets, from tennis matches to loan approvals, demonstrating its versatility and accuracy across diverse scenarios.

### Tennis Data Set

The following output are from the tennis data set (`Tennis.txt`), taken from p.222 of the studing material.

Note the Attributes are:

- A1: Outlook
- A2: Temperature
- A3: Humidity
- A4: Wind
- Label: Play

The train and test files are identical, so the accuracy is 1.0 as expected.

```plaintext
// tennis.txt
%%,Yes,No
##,A1,Sunny,Overcast,Rain
##,A2,Hot,Mild,Cool
##,A3,High,Normal
##,A4,Weak,Strong
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No
```

---

Outout of the program - matching the expected results from p.223:

```bash
dorpascal@Mac-mini HW3_Skeleton % python3.11 my_HW3.py 0 tennis.txt tennis.txt
A1 0.24675
A2 0.02922
A3 0.15184
A4 0.04813
dorpascal@Mac-mini HW3_Skeleton % python3.11 my_HW3.py 1 tennis.txt tennis.txt
└── [A1=?]
    │ Sunny
    ├── [A3=?]
    │   │ High
    │   ├── Label: No
    │   │ Normal
    │   └── Label: Yes
    │ Overcast
    ├── Label: Yes
    │ Rain
    └── [A4=?]
        │ Weak
        ├── Label: Yes
        │ Strong
        └── Label: No
dorpascal@Mac-mini HW3_Skeleton % python3.11 my_HW3.py 2 tennis.txt tennis.txt
Classification of Instance: label=No, attributes=0, 0, 0, 0 = No - Correct? True
Classification of Instance: label=No, attributes=0, 0, 0, 1 = No - Correct? True
Classification of Instance: label=Yes, attributes=1, 0, 0, 0 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=2, 1, 0, 0 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=2, 2, 1, 0 = Yes - Correct? True
Classification of Instance: label=No, attributes=2, 2, 1, 1 = No - Correct? True
Classification of Instance: label=Yes, attributes=1, 2, 1, 1 = Yes - Correct? True
Classification of Instance: label=No, attributes=0, 1, 0, 0 = No - Correct? True
Classification of Instance: label=Yes, attributes=0, 2, 1, 0 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=2, 1, 1, 0 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=0, 1, 1, 1 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=1, 1, 0, 1 = Yes - Correct? True
Classification of Instance: label=Yes, attributes=1, 0, 1, 0 = Yes - Correct? True
Classification of Instance: label=No, attributes=2, 1, 0, 1 = No - Correct? True
dorpascal@Mac-mini HW3_Skeleton % python3.11 my_HW3.py 3 tennis.txt tennis.txt
1.00000
```

---

### Loans Data Set

The following output are from the loan application data set (`prune_train.txt` and `prune_test.txt`). Note that the name of the files probably should be `loan_train.txt` and `loan_test.txt` as the data set is about loans, but I kept the original names.

#### Mode 0

```bash
dorpascal@Mac-mini HW3_Skeleton % python3 my_HW3.py 0 prune_train.txt prune_test.txt
A1 0.08282
A2 0.03319
A3 0.06140
A4 0.01948
A5 0.00907
A6 0.02554
A7 0.00004
A8 0.00303
A9 0.00000
A10 0.00496
```

#### Mode 1

```bash
dorpascal@Mac-mini HW3_Skeleton % python3 my_HW3.py 1 prune_train.txt prune_test.txt
└── [A1=?]
    │ x
    ├── [A3=?]
    │   │ a
    │   ├── [A5=?]
    │   │   │ h
    │   │   ├── Label: B
    │   │   │ s
    │   │   ├── Label: G
    │   │   │ u
    │   │   └── Label: G
    │   │ c
    │   ├── [A2=?]
    │   │   │ n
    │   │   ├── [A5=?]
    │   │   │   │ h
    │   │   │   ├── [A4=?]
    │   │   │   │   │ o
    │   │   │   │   ├── Label: G
    │   │   │   │   │ f
    │   │   │   │   └── Label: B
    │   │   │   │ s
    │   │   │   ├── Label: G
    │   │   │   │ u
    │   │   │   └── [A4=?]
    │   │   │       │ r
    │   │   │       ├── Label: G
    │   │   │       │ o
    │   │   │       └── Label: B
    │   │   │ b
    │   │   ├── [A6=?]
    │   │   │   │ c
    │   │   │   ├── [A5=?]
    │   │   │   │   │ h
    │   │   │   │   ├── [A8=?]
    │   │   │   │   │   │ 1
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ 2
    │   │   │   │   │   └── [A9=?]
    │   │   │   │   │       │ y
    │   │   │   │   │       ├── [A4=?]
    │   │   │   │   │       │   │ o
    │   │   │   │   │       │   └── [A7=?]
    │   │   │   │   │       │       │ 1
    │   │   │   │   │       │       └── [A10=?]
    │   │   │   │   │       │           │ y
    │   │   │   │   │       │           └── Label: G
    │   │   │   │   │       │ n
    │   │   │   │   │       └── Label: G
    │   │   │   │   │ s
    │   │   │   │   ├── Label: G
    │   │   │   │   │ u
    │   │   │   │   └── Label: G
    │   │   │   │ l
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── Label: B
    │   │   │   │   │ o
    │   │   │   │   └── [A5=?]
    │   │   │   │       │ s
    │   │   │   │       ├── [A9=?]
    │   │   │   │       │   │ y
    │   │   │   │       │   ├── Label: G
    │   │   │   │       │   │ n
    │   │   │   │       │   └── [A7=?]
    │   │   │   │       │       │ 1
    │   │   │   │       │       └── [A8=?]
    │   │   │   │       │           │ 2
    │   │   │   │       │           └── [A10=?]
    │   │   │   │       │               │ y
    │   │   │   │       │               └── Label: G
    │   │   │   │       │ n
    │   │   │   │       ├── Label: G
    │   │   │   │       │ u
    │   │   │   │       └── Label: G
    │   │   │   │ r
    │   │   │   ├── Label: G
    │   │   │   │ n
    │   │   │   └── Label: G
    │   │   │ m
    │   │   ├── [A6=?]
    │   │   │   │ c
    │   │   │   ├── Label: G
    │   │   │   │ l
    │   │   │   ├── Label: G
    │   │   │   │ r
    │   │   │   └── [A8=?]
    │   │   │       │ 1
    │   │   │       ├── Label: G
    │   │   │       │ 2
    │   │   │       ├── Label: B
    │   │   │       │ 3
    │   │   │       └── Label: G
    │   │   │ g
    │   │   ├── Label: G
    │   │   │ w
    │   │   └── Label: G
    │   │ d
    │   ├── [A7=?]
    │   │   │ 1
    │   │   ├── [A5=?]
    │   │   │   │ h
    │   │   │   ├── Label: B
    │   │   │   │ s
    │   │   │   ├── Label: G
    │   │   │   │ u
    │   │   │   └── Label: G
    │   │   │ 2
    │   │   └── [A2=?]
    │   │       │ n
    │   │       ├── Label: G
    │   │       │ b
    │   │       ├── Label: B
    │   │       │ g
    │   │       ├── Label: B
    │   │       │ w
    │   │       └── Label: B
    │   │ e
    │   ├── [A8=?]
    │   │   │ 1
    │   │   ├── [A2=?]
    │   │   │   │ n
    │   │   │   ├── [A9=?]
    │   │   │   │   │ y
    │   │   │   │   ├── Label: G
    │   │   │   │   │ n
    │   │   │   │   └── [A6=?]
    │   │   │   │       │ c
    │   │   │   │       ├── Label: G
    │   │   │   │       │ l
    │   │   │   │       ├── [A5=?]
    │   │   │   │       │   │ s
    │   │   │   │       │   ├── Label: G
    │   │   │   │       │   │ u
    │   │   │   │       │   └── [A10=?]
    │   │   │   │       │       │ y
    │   │   │   │       │       ├── Label: B
    │   │   │   │       │       │ n
    │   │   │   │       │       └── Label: G
    │   │   │   │       │ r
    │   │   │   │       ├── [A5=?]
    │   │   │   │       │   │ s
    │   │   │   │       │   ├── [A4=?]
    │   │   │   │       │   │   │ o
    │   │   │   │       │   │   └── [A7=?]
    │   │   │   │       │   │       │ 1
    │   │   │   │       │   │       └── [A10=?]
    │   │   │   │       │   │           │ y
    │   │   │   │       │   │           └── Label: B
    │   │   │   │       │   │ u
    │   │   │   │       │   └── Label: G
    │   │   │   │       │ n
    │   │   │   │       └── Label: G
    │   │   │   │ b
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── [A5=?]
    │   │   │   │   │   │ h
    │   │   │   │   │   ├── [A6=?]
    │   │   │   │   │   │   │ c
    │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │       │ 1
    │   │   │   │   │   │       └── [A9=?]
    │   │   │   │   │   │           │ y
    │   │   │   │   │   │           └── [A10=?]
    │   │   │   │   │   │               │ y
    │   │   │   │   │   │               └── Label: B
    │   │   │   │   │   │ s
    │   │   │   │   │   ├── [A6=?]
    │   │   │   │   │   │   │ l
    │   │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │   │ r
    │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │       │ 1
    │   │   │   │   │   │       └── [A9=?]
    │   │   │   │   │   │           │ n
    │   │   │   │   │   │           └── [A10=?]
    │   │   │   │   │   │               │ y
    │   │   │   │   │   │               └── Label: B
    │   │   │   │   │   │ u
    │   │   │   │   │   └── Label: G
    │   │   │   │   │ o
    │   │   │   │   ├── [A5=?]
    │   │   │   │   │   │ h
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ s
    │   │   │   │   │   ├── [A6=?]
    │   │   │   │   │   │   │ c
    │   │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │   │ l
    │   │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │   │ r
    │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │       │ 1
    │   │   │   │   │   │       ├── [A9=?]
    │   │   │   │   │   │       │   │ y
    │   │   │   │   │   │       │   ├── Label: G
    │   │   │   │   │   │       │   │ n
    │   │   │   │   │   │       │   └── [A10=?]
    │   │   │   │   │   │       │       │ y
    │   │   │   │   │   │       │       └── Label: G
    │   │   │   │   │   │       │ 2
    │   │   │   │   │   │       └── Label: G
    │   │   │   │   │   │ u
    │   │   │   │   │   └── [A6=?]
    │   │   │   │   │       │ l
    │   │   │   │   │       ├── Label: B
    │   │   │   │   │       │ r
    │   │   │   │   │       └── Label: G
    │   │   │   │   │ f
    │   │   │   │   └── Label: B
    │   │   │   │ m
    │   │   │   ├── [A5=?]
    │   │   │   │   │ h
    │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │ r
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ o
    │   │   │   │   │   └── Label: B
    │   │   │   │   │ s
    │   │   │   │   ├── Label: G
    │   │   │   │   │ u
    │   │   │   │   └── Label: G
    │   │   │   │ g
    │   │   │   ├── Label: G
    │   │   │   │ w
    │   │   │   └── Label: G
    │   │   │ 2
    │   │   ├── [A2=?]
    │   │   │   │ n
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── Label: B
    │   │   │   │   │ o
    │   │   │   │   └── [A9=?]
    │   │   │   │       │ y
    │   │   │   │       ├── Label: B
    │   │   │   │       │ n
    │   │   │   │       └── Label: G
    │   │   │   │ b
    │   │   │   ├── [A6=?]
    │   │   │   │   │ c
    │   │   │   │   ├── [A5=?]
    │   │   │   │   │   │ h
    │   │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │   │ r
    │   │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │   │ o
    │   │   │   │   │   │   └── Label: B
    │   │   │   │   │   │ s
    │   │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │   │ r
    │   │   │   │   │   │   ├── Label: B
    │   │   │   │   │   │   │ o
    │   │   │   │   │   │   └── Label: G
    │   │   │   │   │   │ u
    │   │   │   │   │   └── Label: G
    │   │   │   │   │ r
    │   │   │   │   ├── Label: G
    │   │   │   │   │ n
    │   │   │   │   └── Label: G
    │   │   │   │ m
    │   │   │   └── Label: B
    │   │   │ 3
    │   │   └── Label: G
    │   │ n
    │   └── [A2=?]
    │       │ n
    │       ├── Label: B
    │       │ b
    │       ├── [A7=?]
    │       │   │ 1
    │       │   ├── Label: G
    │       │   │ 2
    │       │   └── Label: B
    │       │ m
    │       └── Label: G
    │ n
    ├── [A3=?]
    │   │ a
    │   ├── [A6=?]
    │   │   │ c
    │   │   ├── [A2=?]
    │   │   │   │ b
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── Label: G
    │   │   │   │   │ o
    │   │   │   │   └── [A5=?]
    │   │   │   │       │ s
    │   │   │   │       ├── Label: B
    │   │   │   │       │ u
    │   │   │   │       └── Label: G
    │   │   │   │ m
    │   │   │   └── Label: G
    │   │   │ l
    │   │   ├── Label: B
    │   │   │ n
    │   │   └── Label: B
    │   │ c
    │   ├── [A5=?]
    │   │   │ h
    │   │   ├── Label: G
    │   │   │ s
    │   │   ├── [A4=?]
    │   │   │   │ r
    │   │   │   ├── Label: G
    │   │   │   │ o
    │   │   │   ├── [A2=?]
    │   │   │   │   │ n
    │   │   │   │   ├── Label: G
    │   │   │   │   │ b
    │   │   │   │   ├── [A7=?]
    │   │   │   │   │   │ 1
    │   │   │   │   │   ├── [A6=?]
    │   │   │   │   │   │   │ c
    │   │   │   │   │   │   ├── [A8=?]
    │   │   │   │   │   │   │   │ 2
    │   │   │   │   │   │   │   └── [A9=?]
    │   │   │   │   │   │   │       │ y
    │   │   │   │   │   │   │       └── [A10=?]
    │   │   │   │   │   │   │           │ y
    │   │   │   │   │   │   │           └── Label: B
    │   │   │   │   │   │   │ l
    │   │   │   │   │   │   └── [A9=?]
    │   │   │   │   │   │       │ y
    │   │   │   │   │   │       ├── Label: G
    │   │   │   │   │   │       │ n
    │   │   │   │   │   │       └── [A8=?]
    │   │   │   │   │   │           │ 2
    │   │   │   │   │   │           └── [A10=?]
    │   │   │   │   │   │               │ y
    │   │   │   │   │   │               └── Label: B
    │   │   │   │   │   │ 2
    │   │   │   │   │   └── Label: G
    │   │   │   │   │ g
    │   │   │   │   └── Label: G
    │   │   │   │ f
    │   │   │   └── Label: B
    │   │   │ n
    │   │   ├── Label: B
    │   │   │ u
    │   │   └── Label: G
    │   │ d
    │   ├── [A2=?]
    │   │   │ n
    │   │   ├── Label: G
    │   │   │ b
    │   │   └── Label: B
    │   │ e
    │   ├── [A5=?]
    │   │   │ h
    │   │   ├── [A6=?]
    │   │   │   │ c
    │   │   │   ├── Label: G
    │   │   │   │ l
    │   │   │   ├── [A2=?]
    │   │   │   │   │ n
    │   │   │   │   ├── Label: G
    │   │   │   │   │ b
    │   │   │   │   └── [A4=?]
    │   │   │   │       │ r
    │   │   │   │       ├── [A7=?]
    │   │   │   │       │   │ 1
    │   │   │   │       │   └── [A8=?]
    │   │   │   │       │       │ 1
    │   │   │   │       │       └── [A9=?]
    │   │   │   │       │           │ y
    │   │   │   │       │           └── [A10=?]
    │   │   │   │       │               │ y
    │   │   │   │       │               └── Label: B
    │   │   │   │       │ o
    │   │   │   │       └── Label: B
    │   │   │   │ r
    │   │   │   ├── Label: G
    │   │   │   │ n
    │   │   │   └── [A2=?]
    │   │   │       │ b
    │   │   │       └── [A4=?]
    │   │   │           │ f
    │   │   │           └── [A7=?]
    │   │   │               │ 1
    │   │   │               └── [A8=?]
    │   │   │                   │ 1
    │   │   │                   └── [A9=?]
    │   │   │                       │ y
    │   │   │                       └── [A10=?]
    │   │   │                           │ y
    │   │   │                           └── Label: B
    │   │   │ s
    │   │   ├── [A2=?]
    │   │   │   │ n
    │   │   │   ├── [A9=?]
    │   │   │   │   │ y
    │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │ r
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ o
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ f
    │   │   │   │   │   └── Label: B
    │   │   │   │   │ n
    │   │   │   │   └── Label: B
    │   │   │   │ b
    │   │   │   ├── [A8=?]
    │   │   │   │   │ 1
    │   │   │   │   ├── [A9=?]
    │   │   │   │   │   │ y
    │   │   │   │   │   ├── [A6=?]
    │   │   │   │   │   │   │ c
    │   │   │   │   │   │   ├── Label: B
    │   │   │   │   │   │   │ l
    │   │   │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │   │   │ o
    │   │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │   │       │ 1
    │   │   │   │   │   │   │       └── [A10=?]
    │   │   │   │   │   │   │           │ y
    │   │   │   │   │   │   │           └── Label: B
    │   │   │   │   │   │   │ r
    │   │   │   │   │   │   ├── Label: B
    │   │   │   │   │   │   │ n
    │   │   │   │   │   │   └── Label: B
    │   │   │   │   │   │ n
    │   │   │   │   │   └── [A6=?]
    │   │   │   │   │       │ c
    │   │   │   │   │       ├── [A4=?]
    │   │   │   │   │       │   │ r
    │   │   │   │   │       │   ├── Label: G
    │   │   │   │   │       │   │ o
    │   │   │   │   │       │   └── [A7=?]
    │   │   │   │   │       │       │ 1
    │   │   │   │   │       │       └── [A10=?]
    │   │   │   │   │       │           │ y
    │   │   │   │   │       │           └── Label: G
    │   │   │   │   │       │ l
    │   │   │   │   │       ├── [A7=?]
    │   │   │   │   │       │   │ 1
    │   │   │   │   │       │   ├── [A4=?]
    │   │   │   │   │       │   │   │ r
    │   │   │   │   │       │   │   ├── [A10=?]
    │   │   │   │   │       │   │   │   │ y
    │   │   │   │   │       │   │   │   └── Label: B
    │   │   │   │   │       │   │   │ o
    │   │   │   │   │       │   │   └── Label: B
    │   │   │   │   │       │   │ 2
    │   │   │   │   │       │   └── [A4=?]
    │   │   │   │   │       │       │ o
    │   │   │   │   │       │       └── [A10=?]
    │   │   │   │   │       │           │ y
    │   │   │   │   │       │           └── Label: G
    │   │   │   │   │       │ r
    │   │   │   │   │       ├── [A4=?]
    │   │   │   │   │       │   │ r
    │   │   │   │   │       │   ├── Label: B
    │   │   │   │   │       │   │ o
    │   │   │   │   │       │   └── [A10=?]
    │   │   │   │   │       │       │ y
    │   │   │   │   │       │       ├── [A7=?]
    │   │   │   │   │       │       │   │ 1
    │   │   │   │   │       │       │   └── Label: B
    │   │   │   │   │       │       │ n
    │   │   │   │   │       │       └── Label: G
    │   │   │   │   │       │ n
    │   │   │   │   │       └── [A7=?]
    │   │   │   │   │           │ 1
    │   │   │   │   │           ├── Label: G
    │   │   │   │   │           │ 2
    │   │   │   │   │           └── [A4=?]
    │   │   │   │   │               │ f
    │   │   │   │   │               └── [A10=?]
    │   │   │   │   │                   │ y
    │   │   │   │   │                   └── Label: B
    │   │   │   │   │ 2
    │   │   │   │   └── Label: B
    │   │   │   │ m
    │   │   │   ├── Label: B
    │   │   │   │ g
    │   │   │   ├── Label: G
    │   │   │   │ w
    │   │   │   └── Label: G
    │   │   │ n
    │   │   ├── Label: G
    │   │   │ u
    │   │   └── [A6=?]
    │   │       │ c
    │   │       ├── [A2=?]
    │   │       │   │ b
    │   │       │   ├── Label: G
    │   │       │   │ m
    │   │       │   └── [A7=?]
    │   │       │       │ 1
    │   │       │       ├── Label: B
    │   │       │       │ 2
    │   │       │       └── Label: G
    │   │       │ l
    │   │       ├── Label: G
    │   │       │ r
    │   │       ├── [A9=?]
    │   │       │   │ y
    │   │       │   ├── Label: G
    │   │       │   │ n
    │   │       │   └── [A2=?]
    │   │       │       │ n
    │   │       │       ├── Label: G
    │   │       │       │ b
    │   │       │       └── [A4=?]
    │   │       │           │ r
    │   │       │           ├── Label: G
    │   │       │           │ o
    │   │       │           └── [A7=?]
    │   │       │               │ 1
    │   │       │               ├── [A8=?]
    │   │       │               │   │ 1
    │   │       │               │   └── [A10=?]
    │   │       │               │       │ y
    │   │       │               │       └── Label: B
    │   │       │               │ 2
    │   │       │               └── Label: B
    │   │       │ n
    │   │       └── Label: B
    │   │ n
    │   └── [A6=?]
    │       │ c
    │       ├── [A2=?]
    │       │   │ n
    │       │   ├── Label: G
    │       │   │ b
    │       │   └── Label: B
    │       │ l
    │       ├── Label: B
    │       │ r
    │       ├── Label: G
    │       │ n
    │       └── Label: B
    │ b
    ├── [A6=?]
    │   │ c
    │   ├── [A2=?]
    │   │   │ n
    │   │   ├── Label: G
    │   │   │ b
    │   │   ├── [A4=?]
    │   │   │   │ r
    │   │   │   ├── [A8=?]
    │   │   │   │   │ 1
    │   │   │   │   ├── Label: G
    │   │   │   │   │ 2
    │   │   │   │   ├── Label: B
    │   │   │   │   │ 3
    │   │   │   │   └── Label: G
    │   │   │   │ o
    │   │   │   ├── [A3=?]
    │   │   │   │   │ c
    │   │   │   │   ├── [A5=?]
    │   │   │   │   │   │ h
    │   │   │   │   │   ├── Label: B
    │   │   │   │   │   │ s
    │   │   │   │   │   └── [A8=?]
    │   │   │   │   │       │ 1
    │   │   │   │   │       ├── Label: B
    │   │   │   │   │       │ 2
    │   │   │   │   │       └── Label: G
    │   │   │   │   │ d
    │   │   │   │   ├── Label: G
    │   │   │   │   │ e
    │   │   │   │   └── [A8=?]
    │   │   │   │       │ 1
    │   │   │   │       ├── [A9=?]
    │   │   │   │       │   │ y
    │   │   │   │       │   ├── [A5=?]
    │   │   │   │       │   │   │ s
    │   │   │   │       │   │   ├── [A7=?]
    │   │   │   │       │   │   │   │ 1
    │   │   │   │       │   │   │   ├── [A10=?]
    │   │   │   │       │   │   │   │   │ y
    │   │   │   │       │   │   │   │   └── Label: B
    │   │   │   │       │   │   │   │ 2
    │   │   │   │       │   │   │   └── Label: B
    │   │   │   │       │   │   │ u
    │   │   │   │       │   │   └── Label: B
    │   │   │   │       │   │ n
    │   │   │   │       │   └── [A10=?]
    │   │   │   │       │       │ y
    │   │   │   │       │       ├── [A5=?]
    │   │   │   │       │       │   │ s
    │   │   │   │       │       │   ├── Label: G
    │   │   │   │       │       │   │ u
    │   │   │   │       │       │   └── [A7=?]
    │   │   │   │       │       │       │ 1
    │   │   │   │       │       │       └── Label: B
    │   │   │   │       │       │ n
    │   │   │   │       │       └── Label: B
    │   │   │   │       │ 2
    │   │   │   │       └── Label: G
    │   │   │   │ f
    │   │   │   └── Label: G
    │   │   │ m
    │   │   ├── [A3=?]
    │   │   │   │ a
    │   │   │   ├── Label: G
    │   │   │   │ c
    │   │   │   ├── Label: G
    │   │   │   │ d
    │   │   │   ├── [A5=?]
    │   │   │   │   │ h
    │   │   │   │   ├── Label: B
    │   │   │   │   │ s
    │   │   │   │   └── Label: G
    │   │   │   │ e
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── Label: B
    │   │   │   │   │ o
    │   │   │   │   └── [A8=?]
    │   │   │   │       │ 1
    │   │   │   │       ├── Label: B
    │   │   │   │       │ 2
    │   │   │   │       └── Label: G
    │   │   │   │ n
    │   │   │   └── Label: B
    │   │   │ g
    │   │   ├── [A3=?]
    │   │   │   │ c
    │   │   │   ├── Label: G
    │   │   │   │ e
    │   │   │   └── Label: B
    │   │   │ w
    │   │   └── Label: G
    │   │ l
    │   ├── [A8=?]
    │   │   │ 1
    │   │   ├── [A5=?]
    │   │   │   │ h
    │   │   │   ├── Label: B
    │   │   │   │ s
    │   │   │   ├── [A2=?]
    │   │   │   │   │ n
    │   │   │   │   ├── [A3=?]
    │   │   │   │   │   │ e
    │   │   │   │   │   └── [A4=?]
    │   │   │   │   │       │ o
    │   │   │   │   │       └── [A7=?]
    │   │   │   │   │           │ 1
    │   │   │   │   │           └── [A9=?]
    │   │   │   │   │               │ n
    │   │   │   │   │               └── [A10=?]
    │   │   │   │   │                   │ y
    │   │   │   │   │                   └── Label: G
    │   │   │   │   │ b
    │   │   │   │   ├── [A3=?]
    │   │   │   │   │   │ c
    │   │   │   │   │   ├── [A4=?]
    │   │   │   │   │   │   │ o
    │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │       │ 1
    │   │   │   │   │   │       └── [A9=?]
    │   │   │   │   │   │           │ n
    │   │   │   │   │   │           └── [A10=?]
    │   │   │   │   │   │               │ y
    │   │   │   │   │   │               └── Label: B
    │   │   │   │   │   │ d
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ e
    │   │   │   │   │   └── [A9=?]
    │   │   │   │   │       │ y
    │   │   │   │   │       ├── [A7=?]
    │   │   │   │   │       │   │ 1
    │   │   │   │   │       │   ├── Label: B
    │   │   │   │   │       │   │ 2
    │   │   │   │   │       │   └── Label: G
    │   │   │   │   │       │ n
    │   │   │   │   │       └── Label: G
    │   │   │   │   │ m
    │   │   │   │   └── Label: B
    │   │   │   │ n
    │   │   │   ├── Label: B
    │   │   │   │ u
    │   │   │   └── [A2=?]
    │   │   │       │ n
    │   │   │       ├── Label: G
    │   │   │       │ b
    │   │   │       ├── Label: B
    │   │   │       │ m
    │   │   │       ├── Label: B
    │   │   │       │ w
    │   │   │       └── Label: G
    │   │   │ 2
    │   │   └── Label: G
    │   │ r
    │   ├── [A8=?]
    │   │   │ 1
    │   │   ├── [A2=?]
    │   │   │   │ n
    │   │   │   ├── [A5=?]
    │   │   │   │   │ h
    │   │   │   │   ├── Label: G
    │   │   │   │   │ s
    │   │   │   │   └── Label: B
    │   │   │   │ b
    │   │   │   ├── [A4=?]
    │   │   │   │   │ r
    │   │   │   │   ├── [A5=?]
    │   │   │   │   │   │ h
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ s
    │   │   │   │   │   ├── [A3=?]
    │   │   │   │   │   │   │ e
    │   │   │   │   │   │   └── [A7=?]
    │   │   │   │   │   │       │ 1
    │   │   │   │   │   │       └── [A9=?]
    │   │   │   │   │   │           │ n
    │   │   │   │   │   │           └── [A10=?]
    │   │   │   │   │   │               │ y
    │   │   │   │   │   │               └── Label: B
    │   │   │   │   │   │ u
    │   │   │   │   │   └── Label: G
    │   │   │   │   │ o
    │   │   │   │   └── Label: G
    │   │   │   │ m
    │   │   │   ├── Label: G
    │   │   │   │ g
    │   │   │   ├── [A5=?]
    │   │   │   │   │ s
    │   │   │   │   ├── Label: G
    │   │   │   │   │ u
    │   │   │   │   └── [A3=?]
    │   │   │   │       │ e
    │   │   │   │       └── [A4=?]
    │   │   │   │           │ o
    │   │   │   │           └── [A7=?]
    │   │   │   │               │ 1
    │   │   │   │               └── [A9=?]
    │   │   │   │                   │ n
    │   │   │   │                   └── [A10=?]
    │   │   │   │                       │ y
    │   │   │   │                       └── Label: B
    │   │   │   │ w
    │   │   │   └── Label: G
    │   │   │ 2
    │   │   ├── [A5=?]
    │   │   │   │ h
    │   │   │   ├── Label: B
    │   │   │   │ s
    │   │   │   ├── [A3=?]
    │   │   │   │   │ c
    │   │   │   │   ├── [A2=?]
    │   │   │   │   │   │ n
    │   │   │   │   │   ├── Label: G
    │   │   │   │   │   │ b
    │   │   │   │   │   └── Label: B
    │   │   │   │   │ d
    │   │   │   │   ├── Label: G
    │   │   │   │   │ e
    │   │   │   │   └── Label: B
    │   │   │   │ u
    │   │   │   └── [A7=?]
    │   │   │       │ 1
    │   │   │       ├── [A3=?]
    │   │   │       │   │ c
    │   │   │       │   ├── [A2=?]
    │   │   │       │   │   │ b
    │   │   │       │   │   └── [A4=?]
    │   │   │       │   │       │ o
    │   │   │       │   │       └── [A9=?]
    │   │   │       │   │           │ n
    │   │   │       │   │           └── [A10=?]
    │   │   │       │   │               │ y
    │   │   │       │   │               └── Label: B
    │   │   │       │   │ e
    │   │   │       │   └── Label: G
    │   │   │       │ 2
    │   │   │       └── Label: G
    │   │   │ 3
    │   │   └── [A2=?]
    │   │       │ n
    │   │       ├── Label: G
    │   │       │ b
    │   │       └── Label: B
    │   │ n
    │   └── [A3=?]
    │       │ a
    │       ├── [A2=?]
    │       │   │ n
    │       │   ├── Label: G
    │       │   │ b
    │       │   ├── Label: B
    │       │   │ m
    │       │   └── Label: B
    │       │ c
    │       ├── Label: G
    │       │ d
    │       ├── [A2=?]
    │       │   │ b
    │       │   ├── Label: B
    │       │   │ m
    │       │   └── [A4=?]
    │       │       │ o
    │       │       ├── Label: B
    │       │       │ f
    │       │       └── Label: G
    │       │ e
    │       ├── [A4=?]
    │       │   │ r
    │       │   ├── Label: B
    │       │   │ o
    │       │   ├── Label: B
    │       │   │ f
    │       │   └── [A5=?]
    │       │       │ h
    │       │       ├── [A2=?]
    │       │       │   │ b
    │       │       │   └── [A7=?]
    │       │       │       │ 1
    │       │       │       └── [A8=?]
    │       │       │           │ 1
    │       │       │           └── [A9=?]
    │       │       │               │ y
    │       │       │               └── [A10=?]
    │       │       │                   │ y
    │       │       │                   └── Label: B
    │       │       │ s
    │       │       ├── [A2=?]
    │       │       │   │ n
    │       │       │   ├── [A7=?]
    │       │       │   │   │ 1
    │       │       │   │   └── [A8=?]
    │       │       │   │       │ 1
    │       │       │   │       └── [A9=?]
    │       │       │   │           │ y
    │       │       │   │           └── [A10=?]
    │       │       │   │               │ y
    │       │       │   │               └── Label: B
    │       │       │   │ m
    │       │       │   └── [A7=?]
    │       │       │       │ 1
    │       │       │       └── [A8=?]
    │       │       │           │ 1
    │       │       │           └── [A9=?]
    │       │       │               │ n
    │       │       │               └── [A10=?]
    │       │       │                   │ y
    │       │       │                   └── Label: B
    │       │       │ n
    │       │       └── Label: G
    │       │ n
    │       └── Label: B
    │ g
    └── [A3=?]
        │ a
        ├── [A2=?]
        │   │ n
        │   ├── Label: G
        │   │ b
        │   ├── Label: B
        │   │ w
        │   └── Label: G
        │ c
        ├── [A4=?]
        │   │ r
        │   ├── Label: B
        │   │ o
        │   ├── Label: G
        │   │ f
        │   └── Label: G
        │ d
        ├── Label: B
        │ e
        ├── Label: G
        │ n
        └── Label: G
```

---

#### Mode 2

```bash
dorpascal@Mac-mini HW3_Skeleton % python3 my_HW3.py 2 prune_train.txt prune_test.txt
Classification of Instance: label=B, attributes=1, 1, 2, 2, 1, 3, 1, 1, 1, 0 = B - Correct? True
Classification of Instance: label=B, attributes=1, 1, 0, 1, 1, 0, 0, 1, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 0, 0, 1 = G - Correct? True
Classification of Instance: label=G, attributes=3, 1, 3, 1, 3, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 3, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 2, 1, 1, 1, 1, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=3, 1, 3, 0, 3, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 3, 1, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 2, 1, 1, 0, 1, 1, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 3, 0, 1, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=B, attributes=1, 1, 1, 2, 3, 3, 0, 1, 1, 0 = G - Correct? False
Classification of Instance: label=B, attributes=2, 2, 0, 2, 0, 3, 0, 1, 0, 0 = B - Correct? True
Classification of Instance: label=B, attributes=1, 0, 3, 1, 1, 0, 0, 0, 1, 0 = B - Correct? True
Classification of Instance: label=G, attributes=3, 1, 2, 1, 1, 0, 0, 1, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=2, 1, 3, 2, 1, 3, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=B, attributes=1, 1, 3, 0, 1, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 2, 0, 3, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 0, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 1, 2, 1, 1, 0, 0, 1, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 3, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 4, 1, 1, 2, 0, 1, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 3, 1, 0, 1, 1, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 0, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 1, 3, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 1, 3, 2, 1, 1, 1, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=3, 1, 3, 1, 1, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 1, 1, 3, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 1, 1, 2, 0, 3, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 1, 1, 1, 0, 0, 1, 0 = B - Correct? True
Classification of Instance: label=G, attributes=3, 1, 1, 1, 1, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 2, 3, 1, 1, 2, 1, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 1, 1, 1, 1, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=3, 0, 1, 0, 1, 3, 0, 1, 1, 0 = B - Correct? False
Classification of Instance: label=G, attributes=1, 1, 2, 1, 3, 1, 0, 1, 1, 0 = B - Correct? False
Classification of Instance: label=B, attributes=0, 3, 1, 2, 1, 3, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=2, 1, 3, 1, 0, 3, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 3, 1, 3, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=B, attributes=1, 1, 2, 1, 1, 3, 0, 1, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=1, 4, 3, 1, 1, 1, 1, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 2, 1, 3, 1, 2, 0, 0 = G - Correct? False
Classification of Instance: label=B, attributes=0, 1, 3, 1, 0, 0, 0, 0, 0, 0 = G - Correct? False
Classification of Instance: label=B, attributes=2, 4, 1, 2, 1, 3, 1, 1, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 1, 3, 1, 1, 3, 1, 0, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 1, 0, 1, 2, 0, 2, 0, 0 = G - Correct? False
Classification of Instance: label=B, attributes=1, 1, 1, 0, 1, 1, 0, 0, 0, 0 = G - Correct? False
Classification of Instance: label=B, attributes=2, 0, 3, 1, 1, 1, 0, 0, 0, 0 = G - Correct? False
Classification of Instance: label=B, attributes=2, 1, 4, 2, 0, 3, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 3, 3, 1, 1, 1, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 3, 3, 1, 1, 1, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 0, 3, 0, 1, 2, 0, 0, 1, 0 = B - Correct? False
Classification of Instance: label=G, attributes=1, 1, 3, 1, 0, 1, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=G, attributes=2, 3, 1, 1, 0, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 0, 1, 3, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 1, 3, 1, 3, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 0, 2, 0, 3, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=G, attributes=1, 1, 1, 0, 1, 2, 1, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 4, 0, 1, 3, 1, 1, 1, 0 = B - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 2, 1, 3, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 2, 3, 1, 1, 0, 0, 0, 1, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 3, 2, 0, 3, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 2, 3, 1, 1, 0, 0, 1, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=2, 2, 1, 1, 1, 3, 1, 1, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 0, 1, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 1, 3, 1, 3, 0, 1, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 0, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 4, 1, 1, 1, 0, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 1, 1, 1, 2, 0, 1, 1, 0 = B - Correct? True
Classification of Instance: label=G, attributes=3, 1, 2, 1, 1, 1, 0, 0, 1, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 0, 3, 1, 1, 0, 0, 0, 1, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 3, 3, 1, 0, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 0, 2, 3, 3, 0, 0, 1, 0 = B - Correct? True
Classification of Instance: label=G, attributes=2, 4, 3, 1, 1, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 1, 3, 2, 1, 0, 1, 1 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 3, 2, 0, 3, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=1, 3, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 2, 2, 1, 3, 1, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 1, 3, 2, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 1, 1, 1, 1, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 1, 1, 1, 0, 0, 1, 1, 1 = G - Correct? True
Classification of Instance: label=G, attributes=2, 4, 1, 1, 1, 0, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 3, 0, 2, 1, 3, 1, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 4, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=3, 0, 3, 1, 0, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 3, 4, 1, 1, 0, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 2, 0, 1, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 1, 0, 1, 1, 0, 1, 1, 1 = G - Correct? False
Classification of Instance: label=B, attributes=2, 1, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 0, 3, 0, 3, 0, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 4, 1, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 1, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 0, 2, 2, 1, 3, 1, 0, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=3, 1, 4, 1, 0, 0, 0, 1, 0, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 0, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 2, 0, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 1, 1, 1, 0, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 1, 3, 1, 1, 1, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=2, 0, 1, 1, 1, 2, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 3, 1, 0, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 2, 3, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 3, 1, 3, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 0, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 4, 1, 1, 0, 0, 0, 0, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 3, 1, 0, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 2, 1, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 2, 0, 3, 1, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 3, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 0, 0, 1, 0, 2, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=2, 0, 3, 1, 3, 1, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 1, 1, 3, 2, 1, 1, 1, 0 = B - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 2, 2, 1, 3, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 2, 0, 3, 0, 2, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=1, 1, 3, 2, 0, 3, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 0, 1, 1, 1, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 3, 1, 1, 3, 0, 0, 1, 0 = B - Correct? True
Classification of Instance: label=B, attributes=3, 2, 3, 2, 3, 3, 1, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 2, 2, 0, 1, 1, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 1, 0, 0, 2, 0, 0, 0, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 1, 0, 0, 0, 1 = G - Correct? True
Classification of Instance: label=G, attributes=3, 1, 1, 2, 1, 3, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 0, 1, 1, 0, 1, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 0, 1, 2, 1, 2, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=1, 1, 3, 1, 1, 0, 0, 0, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=2, 0, 2, 2, 0, 3, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 2, 1, 1, 0, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 1, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=2, 1, 3, 1, 1, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 2, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 3, 1, 0, 1, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=0, 1, 2, 1, 1, 2, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 3, 1, 1, 1, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 3, 3, 1, 1, 1, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 0, 2, 1, 1, 0, 1, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 3, 1, 3, 2, 1, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 1, 1, 1, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 1, 1, 1, 3, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=B, attributes=0, 1, 3, 1, 1, 0, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=B, attributes=2, 1, 3, 1, 1, 1, 0, 0, 1, 0 = G - Correct? False
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 2, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 2, 2, 0, 0, 0, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=B, attributes=1, 1, 2, 1, 1, 0, 0, 1, 0, 0 = B - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 0, 3, 1, 3, 1, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 0, 3, 1, 1, 0, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 0, 3, 2, 1, 3, 1, 0, 0, 0 = B - Correct? False
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 2, 0, 1, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=2, 1, 3, 1, 1, 2, 0, 0, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 3, 2, 0, 3, 0, 0, 0, 0 = B - Correct? False
Classification of Instance: label=G, attributes=0, 0, 3, 0, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 3, 1, 3, 2, 1, 0, 1, 1 = G - Correct? True
Classification of Instance: label=G, attributes=0, 1, 1, 1, 1, 3, 0, 1, 1, 0 = G - Correct? True
Classification of Instance: label=G, attributes=0, 0, 3, 1, 1, 0, 0, 0, 0, 0 = G - Correct? True
Classification of Instance: label=G, attributes=1, 1, 3, 1, 0, 0, 0, 0, 1, 0 = G - Correct? True
```

#### Mode 3

```bash
dorpascal@Mac-mini HW3_Skeleton % python3 my_HW3.py 3 prune_train.txt prune_test.txt
0.67485
```
