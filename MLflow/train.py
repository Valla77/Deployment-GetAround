import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, plot_importance
import time


# Loading dataset
df = pd.read_csv("https://storage.googleapis.com/jedha-projects/get_around_pricing_project.csv", index_col=0)

# Dropping rows with anomaly
df = df[(df['mileage'] >= 0) & (df['engine_power'] > 0)]

# Splitting dataset into X features and Target variable
target = 'rental_price_per_day'
Y = df[target]
X = df.drop(target, axis = 1)

# categorizing features
numeric_features = []
categorical_features = []
for i,t in X.dtypes.items():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
    else :
        categorical_features.append(i)

print('Found numeric features ', numeric_features)
print('Found categorical features ', categorical_features)

# Split our training set and our test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Features preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocessings on train set
print("Performing preprocessings on train set...")
print(X_train.head())
X_train = preprocessor.fit_transform(X_train)
print('...Done.')
print(X_train[0:5]) 
print()

# Preprocessings on test set
print("Performing preprocessings on test set...")
print(X_test.head()) 
X_test = preprocessor.transform(X_test) 
print('...Done.')
print(X_test[0:5,:])

# Set your variables for your environment
EXPERIMENT_NAME="getaround-mlflow-experiment"

# Set tracking URI to your Heroku application
os.environ["APP_URI"]="https://getaround-mlflowapp-val-5ecb428bcb6e.herokuapp.com/"
mlflow.set_tracking_uri(os.environ["APP_URI"])

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Time execution
start_time = time.time()

# Call mlflow autolog
mlflow.sklearn.autolog()


print("Linear Regression Training ...")
run_name = 'linear_regression'
with mlflow.start_run(run_name=run_name) as run:
    model_lr = LinearRegression()
    model_lr.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_lr.predict(X_train)
    Y_test_pred = model_lr.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    mlflow.end_run()

print("Random Forest Training ...")
run_name = 'random_forest'
with mlflow.start_run(run_name=run_name) as run:
    model_rf = RandomForestRegressor(max_depth=10)
    model_rf.fit(X_train, Y_train)
    
    print("Training done.")
    Y_train_pred = model_rf.predict(X_train)
    Y_test_pred = model_rf.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    mlflow.end_run()

print("Ridge Training ...")
run_name = 'ridge'
with mlflow.start_run(run_name=run_name) as run:
    model = Ridge(alpha=1)
    model.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
   
    mlflow.end_run()

print("Lasso Training ...")
run_name = 'lasso'
with mlflow.start_run(run_name=run_name) as run:
    model_lasso =  Lasso(alpha=1)
    model_lasso.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_lasso.predict(X_train)
    Y_test_pred = model_lasso.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    
    mlflow.end_run()

print("GridSearchCV RandomForest Training ...")
# Grid of values to be tested
run_name = 'random_forest_gridsearch'
with mlflow.start_run(run_name=run_name) as run:
    params_rf = {
    'max_depth': [10, 12, 14, 16, 18, 20],
    'min_samples_split': [2, 4, 8, 10, 12, 14, 16],
    'n_estimators': [60, 80, 100, 200, 300, 400, 500]
    }
    rf = RandomForestRegressor()
    model_gridrf = GridSearchCV(rf, params_rf, cv=5, verbose=True, n_jobs=-1)
    model_gridrf.fit(X_train, Y_train)
    print("Training done.")
    Y_train_pred = model_gridrf.predict(X_train)
    Y_test_pred = model_gridrf.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
    mlflow.log_param("best_params", model_gridrf.best_params_)
    
    mlflow.end_run()

print("XGBRegressor Training ...")
run_name = 'xgbr'
with mlflow.start_run(run_name=run_name) as run:
    model_xgb = XGBRegressor(n_estimators=200, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, alpha=0.1, random_state=42)
    model_xgb.fit(X_train, Y_train)
    
    print("Training done.")
    Y_train_pred = model_xgb.predict(X_train)
    Y_test_pred = model_xgb.predict(X_test)
    mlflow.log_metric("training_r2_score",r2_score(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_absolute_error",mean_absolute_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_mean_squared_error",mean_squared_error(Y_train, Y_train_pred))
    mlflow.log_metric("training_root_mean_squared_error",mean_squared_error(Y_train, Y_train_pred, squared=False))
    mlflow.log_metric("testing_r2_score",r2_score(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_absolute_error",mean_absolute_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_mean_squared_error",mean_squared_error(Y_test, Y_test_pred))
    mlflow.log_metric("testing_root_mean_squared_error",mean_squared_error(Y_test, Y_test_pred, squared=False))
   
    mlflow.end_run()

print("All training is done!")
print(f"---Total training time: {time.time()-start_time}")



