## `Churn Classification with LLM Integration`
> The marketing team can seamlessly interact with the API using natural language queries. The LLM extracts relevant features from these unstructured inputs, converting them into structured data. This structured data is then processed through the pipeline and fed into the churn classifier model, which predicts whether a customer will churn (True for Exited) or not (False for Not Exited).
---------------------------

## `Project Files`
<img src="./artifacts/project-content.png" alt="content" width="450" height="600">

---------------------------

## `Diagram of Steps`
<img src="./artifacts/diagram.png" alt="diagram" width="400" height="600">

---------------------------

<h2 align="center"> Documentation </h2>

### `1. Tools and Language used`
* Programming: `Python=3.10.9`
* Deployment: `FastAPI & uvicorn`
* Machine Learning: `Scikit-Learn`
* LLM model: `google/gemma-1.1-2b-it`
---------------------------
### `2. Preparing the Enviroment`
`Open Terminal and Run the following commands`
``` python
# create a new environment with the name "churn" and the specified Python version
conda create --name churn python==3.10.9 notebook

# activate the environment
conda activate churn

# install the required packages from the requirements.txt file
pip install -r requirements.txt
```
---------------------------
### `3. Dataset`
* Using the dataset [Data](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) for `churn classification`.
* The column `Exited` column in dataset refers to target and it is already encoded. `1 = Exited` & `0 = Not Exited`
* This data set contains details of a bank's customers and the target variable is a `binary` variable reflecting the fact 
`whether the customer left the bank (closed his account) or he continues to be a customer`.
* The used evaluation metric here will be `F1-score`. It is the significant metric over other metrics as data is `binary and imbalanced`.
---------------------------

### `4. Churn-Notebook`
The Notebook is fully automated and can be run over any machine with any OS.

**Parts of Churn-Notebook as follows:**

* **Read the Dataset**

* **Exploratory Data Analysis**
    * Univariate Viz
    * Bivariate Viz
    * Handling Outliers 

* **Split to train & test**
    * Apply stratification on target
    * Split with 80% Train & 20% Test 

* **Data Preprocessing**
    * Numerical Columns: Imputing using median & Standardization
    * Categorical Columns: Imputing using mode & One-Hot Encoding (OHE)
    * Ready Columns: Just imputing using mode
    * Building a `combined pipeline` and dump it locally

* **Dealing with Imbalancing:** We have 3 methods
    1. Do not take that effect and work on original data
    2. Apply `class-weight` argument in most algorithms
    3. Apply `SMOTE` algorithm to generate more data for the minority class

* **Building Models**
    1. Logistic Regression Model
        * Considering both class-weight & SMOTE for two trials
        * SMOTE with Logistic is the best in Logistic models
    2. Naive Bayes Model
        * Considering using SMOTE for imbalanced data
        * Using Gaussian NB
    3. Random Forest Model
        * Considering both class-weight & SMOTE for two trials
        * SMOTE with RF is the best in RF models
        * Tuning RF using GridSearchCV with some search spaces

* **Final Comparison**
    * Dump the `Tuned-RF` model as it is the best

---------------------------
### `5. LLMs-Notebook`
* The Notebook is fully automated and can be run over any machine with `CPU` only with around `needed System RAM ~ 12 GB`.
* **Parts of LLMs-Notebook as following**
    * Prepare `Pydantic Response` to parse the output in a proper way
    * Using `google/gemma-1.1-2b-it` an open-source model on `HugggingFace` without quantization, and only `CPU with around needed ~ 12 GB RAM`
    * Create a function that `engineers the prompt` for the LLM and provides a `one-shot learning` example.
    * Some Functions for parsing and validate the output to be in a proper way and ready for processing and classification

<br />

* Here is some examples for marketing team to pass to the model.

   > `Mohammed Agoor is a 27-year-old male from the Spain with a credit score of 700. He has been with the bank for 5 years, has a balance of 5000.0 USD, holds 2 products, owns a credit card, is an active member, and earns an estimated salary of 100000.0 USD.` 

    <br />

    > `Maher is a 28-year-old male from the France with a credit score of 620. He has been with the bank for 2 years, has a balance of 5400.0 USD, holds 3 products, owns a credit card, is an active member, and earns an estimated salary of 105000.0 USD.`
---------------------------
### `6. Utils File`
* This file combines functions for the LLM to extract the necessary features in `Pydantic format`. These features are then fed into the previously dumped processing `pipeline`, and finally, into the `classifier` to predict customer churn.
---------------------------
### `7. Main File`
* This file contains the router for running the API. The deployment uses `FastAPI`, and the endpoint to be called is `/predict`.

* You can run the API via: `uvicorn main:app --reload`
* Open the Swagger Docs via: `localhost:8000/docs`
* You can also use `Postman` to send request or use `curl` like following

``` bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'KeyToken={}&Text={}
```
---------------------------

### `Usage Example`
1. Run in your Terminal `uvicorn main:app --reload`
2. Open browser and go to `localhost:8000/docs`
3. Pass the required param `KeyToken`: it is a ssecret get, you can get it from `env` file
4. Pass the required param `Text`: Try one on the same template of above exmaples
5. The respoonse is `Pydantic class` with extracted features and prediction.
---------------------------

### `API Testing on Swagger`
<img src="./artifacts/swagger.jpeg" alt="swagger" width="400" height="600">

---------------------------

### `API Testing on Postman`
<img src="./artifacts/postman.png" alt="postman" width="600" height="600">

* You can found a json file for postman API testing here: `./artifacts/churn.postman_collection.json`
---------------------------
