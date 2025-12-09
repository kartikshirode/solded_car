import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv('autos.csv')


df.drop(
    columns= ['index', 'dateCreated', 'dateCrawled', 'lastSeen', 'nrOfPictures', 'postalCode', 'name', 'seller', 'offerType', 'abtest', 'model'],
    inplace=True,
    axis=1
)

year_current = datetime.now().year
month_current = datetime.now().month

df['TimeforRegistration'] = ((year_current - df['yearOfRegistration']) * 12 + (month_current - df['monthOfRegistration']))

df.drop(
    columns=['yearOfRegistration', 'monthOfRegistration'],
    inplace=True,
    axis= 1
)

for i in df.select_dtypes(include="object"):
    df[i] = df[i].fillna("unknown")

df["vehicleType"] = df["vehicleType"].replace({
    "kleinwagen": "small_car",
    "limousine": "sedan",
    "kombi": "wagon",
    "cabrio": "convertible",
    "andere": "other"
})

df["gearbox"] = df["gearbox"].replace({
    "manuell": "manual",
    "automatik": "automatic"
})

df["fuelType"] = df["fuelType"].replace({
    "benzin": "petrol",
    "diesel": "diesel",
    "elektro": "electric",
    "hybrid": "hybrid",
    "lpg": "lpg",
    "cng": "cng",
    "andere": "other"
})

df["notRepairedDamage"] = df["notRepairedDamage"].replace({
    "ja": "yes",
    "nein": "no"
})

brand_count = df['brand'].value_counts()
common_brand = brand_count[brand_count >= 5100].index

df["brand"] = df["brand"].where(df["brand"].isin(common_brand),'Other')

df = df[(df['price'] >= 500) & (df['price'] <= 40000)]
df = df[(df['powerPS'] >= 30) & (df['powerPS'] <= 600)]
df = df[(df['TimeforRegistration'] >= 0) & (df['TimeforRegistration'] <= 420)]
df = df[(df['kilometer'] >= 0) & (df['kilometer'] <= 300000)]

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_fea = ['powerPS', 'kilometer', 'TimeforRegistration']
catgo_fea = ['vehicleType', 'gearbox', 'fuelType', 'brand', 'notRepairedDamage']

num_trans = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

catgo_trans = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('Hotencode', OneHotEncoder(handle_unknown='ignore'))
    ]
)

combine = ColumnTransformer(
    transformers=[
        ('num', num_trans, num_fea),
        ('catgo', catgo_trans, catgo_fea)
    ]
)   

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

best_mae = float("inf")
best_model = None
best_model_name = None

for name, reg in models.items():
    pipe = Pipeline(
        steps=[
            ('processing', combine),
            ('model', reg)
        ]
    )
    pipe.fit(X_train, y_train_log)
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    print(f"\n{name}")
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    if mae < best_mae:
        best_mae = mae
        best_model = pipe
        best_model_name = name



import joblib
joblib.dump(
    {
        "model": best_model,
        "metric": "MAE",
        "mae": best_mae,
        "name": best_model_name
    },
    "car_model1.joblib"
)

print(f"Saved model: {best_model_name} with MAE = {best_mae}")