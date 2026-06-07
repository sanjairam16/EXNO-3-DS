## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       ~~~
        import pandas as pd
        df=pd.read_csv("/content/Encoding Data.csv")
        df
       ~~~
  <img width="650" height="446" alt="e-3 1" src="https://github.com/user-attachments/assets/a0f044b5-6c0c-4976-8c0b-3bd7f2a577d2" />
  ~~~
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
~~~
<img width="650" height="446" alt="e-3 1" src="https://github.com/user-attachments/assets/75db711c-bf61-4071-bcbe-acdda2551264" />
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
<img width="650" height="446" alt="e-3 1" src="https://github.com/user-attachments/assets/58eded04-fc12-44a5-bfd8-f56b3b1318eb" />
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
~~~
    from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
<img width="521" height="348" alt="e3-4" src="https://github.com/user-attachments/assets/141f3f07-ac76-4c0e-97d4-d7d9ae1d03c1" />
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
<img width="733" height="336" alt="e3-5" src="https://github.com/user-attachments/assets/0d759b7d-f89e-4868-a507-ba1a73228b18" />
~~~
pip install --upgrade category_encoders
~~~
<img width="824" height="254" alt="e3-6" src="https://github.com/user-attachments/assets/1c85845d-db54-4162-9797-b6692bd48f69" />
~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
~~~
<img width="670" height="407" alt="e3-7" src="https://github.com/user-attachments/assets/f94088d2-7b79-4fae-95c8-d643017881fb" />
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
~~~
<img width="750" height="411" alt="e3-8" src="https://github.com/user-attachments/assets/adfddda9-5f4b-41b4-8192-afb12861769e" />
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="750" height="411" alt="e3-9" src="https://github.com/user-attachments/assets/d4aadd8c-4117-4100-9fa0-1e525e735e3c" />
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="750" height="411" alt="e3-10" src="https://github.com/user-attachments/assets/bb1ece98-9855-4eda-b2d2-3a0e482bcf14" />
```
df.skew()
```
<img width="750" height="411" alt="e3-10" src="https://github.com/user-attachments/assets/80a9428f-a3d4-4d83-bcd3-fdf0e605e53c" />

```
np.log(df["Highly Positive Skew"])
```
<img width="819" height="392" alt="e3-11" src="https://github.com/user-attachments/assets/f72adeed-2417-421f-b2d3-1f800f69f0ab" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="510" height="177" alt="e3-12" src="https://github.com/user-attachments/assets/f6988fb2-b726-4201-84c9-4765c5eacf45" />
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="508" height="216" alt="e3-13" src="https://github.com/user-attachments/assets/30c15e2a-f198-4387-b5b4-ce2654e9e7b6" />

```
np.square(df["Highly Positive Skew"])

```


<img width="553" height="221" alt="e3-14" src="https://github.com/user-attachments/assets/2ea18f4a-9d4f-4102-83e4-02f8fcb7bdfa" />
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="541" height="220" alt="e3-15" src="https://github.com/user-attachments/assets/034ec085-6e1f-438a-ba82-7f90c2bc79b0" />
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```

<img width="822" height="394" alt="e3-16" src="https://github.com/user-attachments/assets/e97af11f-bac7-4bfb-811a-2f03bb74d7e3" />
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="815" height="410" alt="e3-17" src="https://github.com/user-attachments/assets/119b6f3f-dbde-4860-8ba3-e24164a7f73a" />

```

```
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
