# Amazon Market Basket Analysis
### Customer Purchasing Behavior and Product Recommendation Report

**Author:** Mrudula  
**Domain:** E-Commerce | Customer Analytics  
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, MLxtend

---

## Project Overview

This project analyses Amazon Customer Behaviour Survey data to uncover
purchasing patterns, segment customers, and provide actionable
recommendations for personalized shopping experiences.

**Key Business Questions Answered:**
- What products are frequently purchased together?
- How do customers vary in their purchasing behaviour?
- Which customers are at risk of churning?
- What drives satisfaction and repeat purchases?

---

## Project Structure
amazon-market-basket-analysis/

│

├── data/

│   ├── raw/                    # Original Amazon.csv dataset

│   └── processed/              # Cleaned dataset (auto-generated)

│

├── notebooks/

│   └── amazon_analysis.ipynb   # Full analysis notebook

│

├── src/                        # Modular Python source code

│   ├── clean_data.py           # Task 1: Data cleaning

│   ├── helpers.py              # Shared utility functions

│   ├── build_features.py       # Task 3: Feature engineering

│   ├── clustering.py           # Task 3: KMeans clustering

│   └── plot_helpers.py         # Tasks 2,3,4,5: Visualizations

│

├── reports/

│   └── figures/                # All output charts (auto-generated)

│

├── config/

│   └── config.yaml             # Project parameters

│

├── main.py                     # Pipeline entry point

├── requirements.txt            # Python dependencies

└── README.md                   # Project documentation
---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/amazon-market-basket-analysis.git
cd amazon-market-basket-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python main.py
```

### 4. Or Open the Notebook
```bash
jupyter notebook notebooks/amazon_analysis.ipynb
```

---

## Tasks Completed

| Task | Description | Output |
|------|-------------|--------|
| **Task 1** | Data Cleaning & Preparation | `data/processed/Amazon_cleaned.csv` |
| **Task 2** | Descriptive Behavior Analysis | Charts 01–07 |
| **Task 3** | Customer Segmentation (Rule-based + KMeans) | Charts 08–10 |
| **Task 4** | Recommendation & Review Insights | Charts 11–13 |
| **Task 5** | Summary Dashboard | `00_summary_dashboard.png` |

---

## Key Findings

- **800 customers** surveyed across diverse demographics
- **Clothing & Fashion** is the most purchased category
- **High shipping cost** is the #1 cart abandonment reason
- Average shopping satisfaction: **3.01 / 5** — room for improvement
- **3 customer segments** identified:
  - 🟢 **Frequent Buyers** — high frequency, high satisfaction
  - 🟡 **Occasional Shoppers** — moderate frequency and satisfaction  
  - 🔴 **At-Risk Customers** — low satisfaction, churn risk

---

## Recommendations

1. **Improve recommendation algorithms** — customers who find
   recommendations helpful report higher satisfaction scores
2. **Introduce free shipping thresholds** — reduces #1 abandonment reason
3. **Target At-Risk customers** with personalised discount notifications
4. **Enhance semantic search** — keyword search dominates usage
5. **Incentivise review reading** — heavy reviewers show better
   rating accuracy perception

---

## Dataset

**Amazon Customer Behaviour Survey**  
- 800 survey responses  
- 24 columns covering demographics, purchase habits,
  browsing behaviour, review engagement, and satisfaction

---

## Video Presentation

[Add your Google Drive video link here]

---

## Connect

**GitHub:** [your-github-profile]  
**LinkedIn:** [your-linkedin-profile]