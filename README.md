# Ex-06-Feature-Transformation

## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
### STEP 1: Read the given Data

### STEP 2: Clean the Data Set using Data Cleaning Process

### STEP 3: Apply Feature Transformation techniques to all the features of the data set

### STEP 4: Save the data to the file

## PROGRAM AND OUTPUT:
```
NAME : Jagan A
REG NO: 212221230037
```
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/3805dbef-ab5f-49b4-bdbc-fe3293454984)

```
df.skew()
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/3fd9e28a-5d82-430c-a732-19146cea2861)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/08006ad8-ed48-4c26-81c1-244c8849e0ef)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/3eb5ce3a-7993-4bcf-a0ab-c367bd13e564)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/40c1142e-890f-4dc8-92ef-241dfef3f96d)

```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/69f74411-a8f6-4949-a7d7-92cb57f50e09)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/8cea9b87-b6e3-496e-94e3-a76bcfb36ae3)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/8e2eebec-bdaf-497e-bf01-f0a907735840)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/f267eaa2-6272-4edd-bd46-e30e38511fbf)

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/718226f5-5b1e-4d4c-aca8-562fd2444016)

```
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/f2126b6f-adf7-4454-a4c6-158d861e90e2)

```
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/JoyceBeulah/Ex-06-Feature-Transformation/assets/118343698/16d129d9-e2bc-46d1-a86a-b2e607814af4)

## RESULT:
Thus,Feature transformation is performed and executed successfully for the given dataset.

