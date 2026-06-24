# California Property Price Prediction

## Project Overview

This project aims to develop a machine learning model capable of predicting residential property sale prices in California using historical Multiple Listing Service (MLS) transaction data.

The project is being completed as part of a 12-week data science internship focused on real estate analytics and predictive modeling. The objective is to predict a property's final sale price (ClosePrice) based on its characteristics, including property size, location, number of bedrooms, bathrooms, lot size, and other relevant features.

## Project Objectives

* Explore and understand California residential property sales data.
* Build a reproducible preprocessing pipeline.
* Train and compare multiple machine learning models.
* Evaluate model performance using R², MAPE, and MdAPE.
* Engineer additional predictive features.
* Develop an optional Streamlit application for property price prediction.
* Deliver documentation, presentation materials, and deployment-ready code.

## Repository Structure

```text
california-property-price-prediction/
├── notebooks/
├── src/
├── app/
├── models/
├── reports/
├── data/
├── README.md
├── requirements.txt
└── .gitignore
```

## Data Source

The datasets used in this project are provided through a private FTP server and are not included in this repository.

Only observations meeting the following criteria are used:

* PropertyType = Residential
* PropertySubType = SingleFamilyResidence

## Project Timeline

| Week | Topic                    |
| ---- | ------------------------ |
| 1    | Orientation & Setup      |
| 2    | Data Exploration         |
| 3    | Data Preprocessing       |
| 4    | Baseline Model           |
| 5    | Model Comparison         |
| 6    | Feature Engineering      |
| 7    | Advanced Models          |
| 8    | Evaluation               |
| 9    | Streamlit Application    |
| 10   | Documentation            |
| 11   | Presentation Preparation |
| 12   | Final Delivery           |
