# **SHapley Additive exPlanations**

## Explainability of AutoML Models with SHAP (SHapley Additive exPlanations)

A company produces hospital items through one of its factories in Brazil. 
Each the factory has several industrial equipment that periodically requires maintenance.
The company collected historical data associating different metrics (predictor variables) the need for equipment maintenance (yes or no). 
The idea is to have a Machine model Learning capable of predicting when each machine will require maintenance and thus avoiding downtime unscheduled

<div align="center">
<p float="left">
    <img src="/images/.png" width="1000" height="500"/>
</p>
</div>

***
## 1. BUSINESS PROBLEMS

One of the great challenges of those who work as a Data Scientist and create data models Machine Learning is the choice (trade-off) of the algorithm to be used. 
Essentially we have two options:

1. Use simple and easily explained algorithms, such as Logistic Regression.
2. Use powerful algorithms that achieve much higher accuracy, but at a cost of losing interpretability, such as Gradient Boosting or Support Vector Machines. Those models are often called “black boxes”, which implies that you know what goes into and what comes out, but there's no way to understand what's really going on behind the scenes.

This is a dilemma that is part of everyday life as a Data Scientist and, as Machine Learning is increasingly used in companies, the concern with the explainability of the model becomes basic requirement.

That said, this project aims to present an alternative that can help explain how predictive models make predictions.

SHAP – which stands for SHpley Additive exPlanations – is probably the state of the art in Machine Learning explainability. 
This algorithm was first published in 2017 by [Lundberg and Lee](https://arxiv.org/abs/1705.07874) and it's a brilliant way to do reverse engineering the output of any predictive algorithm.

In a nutshell, SHAP values ​​are used whenever you have a template complex (it could be a GBM, XGBoost, a Neural Network or anything that takes a few resources as input and produces some predictions as output) and you want to understand which decisions the model has made.
That is, predictive models respond to “how much”. SHAP answers the “why”.

Therefore, when we apply SHAP analysis to the model, we produce an array of SHAP values. The higher the SHAP value, the greater the probability of the positive prediction and vice versa.
Furthermore, a SHAP value greater than zero leads to an increase in probability, a value less than zero leads to a decrease in probability.

In this project, we will also use the AutoML.

AutoML or Automated Machine Learning is the process of automating the tasks of the development of Machine Learning models. 
With AutoML, Data Scientists can create ML models with high scale, efficiency and productivity, while giving model quality support.
The development of the traditional Machine Learning model is intensive use of resources, requiring significant domain knowledge and time to produce and compare dozens of models.
With AutoML, you'll speed up the time it takes to get models from Production-ready ML with great ease and efficiency.
The [link](https://www.automl.org/automl/) below has more details on AutoML, which is studied in the Machine course Learning from Data Scientist Training.


***
## 2. BUSINESS ASSUMPTIONS

For this project I will create a mass of dummy data, which represents data real.

Below the data dictionary:

|                                                 |                                                            |
|:-----------------------------------------------:|:----------------------------------------------------------:|
|Predictor Variable 1 (productivity)              |Overall Equipment Effectiveness: This is a measure of productivity, which describes the of the time a machine works at peak performance. The metric is a product of themachine availability, performance and quality|
|Predictor Variable 2 (yield)                     |First Pass Yield: This is the share of products that leave the production and that are free of defects and meet specifications without the need for any rectification work.|
|Predictor Variable 3 (cost)                      |Energy cost per unit: This is the cost of electricity, steam, oil or gas required to produce a particular unit of product at the factory.|
|Predictor Variable 4 (priority)                  |Equipment priority when entering the maintenance period (Low, Medium, High).|
|Predictor Variable 5 (efficiency)                |The amount of product that a machine produces during a specific period. That metric can also be applied to the entire production line to check its efficiency.|
|**Target Variable (maintenance)**                |0 means the equipment does not require maintenance (no) and 1 means the equipment requires maintenance (yes)|

***
