{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Spliting Dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
      "0       1   69        1               2        2              1   \n",
      "1       1   74        2               1        1              1   \n",
      "2       0   59        1               1        1              2   \n",
      "3       1   63        2               2        2              1   \n",
      "4       0   63        1               2        1              1   \n",
      "\n",
      "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
      "0                1         2         1         2                  2         2   \n",
      "1                2         2         2         1                  1         1   \n",
      "2                1         2         1         2                  1         2   \n",
      "3                1         1         1         1                  2         1   \n",
      "4                1         1         1         2                  1         2   \n",
      "\n",
      "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  LUNG_CANCER  \n",
      "0                    2                      2           2            1  \n",
      "1                    2                      2           2            1  \n",
      "2                    2                      1           2            0  \n",
      "3                    1                      2           2            0  \n",
      "4                    2                      1           1            0  \n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv('lungcancer.csv')\n",
    "print(data.head())\n",
    "X = data.drop(columns=['LUNG_CANCER'])  # Replace 'target' with actual target column name\n",
    "y = data['LUNG_CANCER']  # Replace 'target' with actual target column name\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=255, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Bayes Accuracy: 85.882353%\n"
     ]
    }
   ],
   "source": [
    "# Standardize the dataset\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Naive Bayes\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "y_pred_nb = nb.predict(X_test)\n",
    "accuracy_nb = accuracy_score(y_test, y_pred_nb)\n",
    "print(f'Naive Bayes Accuracy: {accuracy_nb * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>0</td>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>305</th>\n",
       "      <td>1</td>\n",
       "      <td>70</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>306</th>\n",
       "      <td>1</td>\n",
       "      <td>58</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>307</th>\n",
       "      <td>1</td>\n",
       "      <td>67</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>308</th>\n",
       "      <td>1</td>\n",
       "      <td>62</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>309 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0         1   69        1               2        2              1   \n",
       "1         1   74        2               1        1              1   \n",
       "2         0   59        1               1        1              2   \n",
       "3         1   63        2               2        2              1   \n",
       "4         0   63        1               2        1              1   \n",
       "..      ...  ...      ...             ...      ...            ...   \n",
       "304       0   56        1               1        1              2   \n",
       "305       1   70        2               1        1              1   \n",
       "306       1   58        2               1        1              1   \n",
       "307       1   67        2               1        2              1   \n",
       "308       1   62        1               1        1              2   \n",
       "\n",
       "     CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  \\\n",
       "0                  1         2         1         2                  2   \n",
       "1                  2         2         2         1                  1   \n",
       "2                  1         2         1         2                  1   \n",
       "3                  1         1         1         1                  2   \n",
       "4                  1         1         1         2                  1   \n",
       "..               ...       ...       ...       ...                ...   \n",
       "304                2         2         1         1                  2   \n",
       "305                1         2         2         2                  2   \n",
       "306                1         1         2         2                  2   \n",
       "307                1         2         2         1                  2   \n",
       "308                1         2         2         2                  2   \n",
       "\n",
       "     COUGHING  SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  \\\n",
       "0           2                    2                      2           2   \n",
       "1           1                    2                      2           2   \n",
       "2           2                    2                      1           2   \n",
       "3           1                    1                      2           2   \n",
       "4           2                    2                      1           1   \n",
       "..        ...                  ...                    ...         ...   \n",
       "304         2                    2                      2           1   \n",
       "305         2                    2                      1           2   \n",
       "306         2                    1                      1           2   \n",
       "307         2                    2                      1           2   \n",
       "308         1                    1                      2           1   \n",
       "\n",
       "     LUNG_CANCER  \n",
       "0              1  \n",
       "1              1  \n",
       "2              0  \n",
       "3              0  \n",
       "4              0  \n",
       "..           ...  \n",
       "304            1  \n",
       "305            1  \n",
       "306            1  \n",
       "307            1  \n",
       "308            1  \n",
       "\n",
       "[309 rows x 16 columns]"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decission Tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 87.843137%\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train, y_train)\n",
    "y_pred_dt = dt.predict(X_test)\n",
    "accuracy_dt = accuracy_score(y_test, y_pred_dt)\n",
    "print(f'Decision Tree Accuracy: {accuracy_dt * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 87.843137%\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred_rf = rf.predict(X_test)\n",
    "accuracy_rf = accuracy_score(y_test, y_pred_rf)\n",
    "print(f'Random Forest Accuracy: {accuracy_rf * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Accuracy: 86.274510%\n"
     ]
    }
   ],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)\n",
    "y_pred_knn = knn.predict(X_test)\n",
    "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
    "print(f'KNN Accuracy: {accuracy_knn * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "SVM\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy: 83.529412%\n"
     ]
    }
   ],
   "source": [
    "svm = SVC(kernel='linear')\n",
    "svm.fit(X_train, y_train)\n",
    "y_pred_svm = svm.predict(X_test)\n",
    "accuracy_svm = accuracy_score(y_test, y_pred_svm)\n",
    "print(f'SVM Accuracy: {accuracy_svm * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic Regression \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 88.627451%\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred_lr = lr.predict(X_test)\n",
    "accuracy_lr = accuracy_score(y_test, y_pred_lr)\n",
    "print(f'Logistic Regression Accuracy: {accuracy_lr * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ploting Accuracy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Abhishek Karnam\\AppData\\Local\\Temp\\ipykernel_30640\\3753162025.py:17: FutureWarning: \n",
      "\n",
      "Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.\n",
      "\n",
      "  sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette=light_colors)\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "accuracy_scores = {\n",
    "    'Naive Bayes': accuracy_nb,\n",
    "    'Decision Tree': accuracy_dt,\n",
    "    'Random Forest': accuracy_rf,\n",
    "    'KNN': accuracy_knn,\n",
    "    'SVM': accuracy_svm,\n",
    "    'Logistic Regression': accuracy_lr\n",
    "}\n",
    "\n",
    "# Define a light color palette\n",
    "light_colors = [\"lightblue\", \"lightgreen\", \"lightcoral\", \"lightskyblue\", \"lightpink\", \"lightyellow\"]\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette=light_colors)\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Model Performance Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
