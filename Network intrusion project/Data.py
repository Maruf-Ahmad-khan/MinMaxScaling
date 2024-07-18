import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Solution:
     def __init__(self, df):
        self.df = df

    # Read the file
     def Read_data(self):
        return self.df.head()

    # Find the shape
     def Shape_data(self):
        return self.df.shape

    # Find the list of the column
     def List_data(self):
        return self.df.columns.tolist()

    # Find the info
     def Info(self):
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        buffer.close()
        return info_str

    # Find the Descriptive stats of the data
     def Describe(self):
        return self.df.describe(include='object')

    # Calculate total number of rows
     def Total_Number_Rows(self):
        return self.df.shape[0]

    # Find the count of Count "Class"
     def Count_of_Class(self):
        plt.figure(figsize=(12, 6))
        sns.countplot(x=self.df['class'])
        plt.title("Frequency of the class", fontsize=15, weight="bold")
        plt.grid(True)
        plt.savefig('class_frequency_plot.png')
        plt.close()

    # Find the Count of anomaly and normal
     def Anomly_Normal_Counts(self):
        return self.df['class'].value_counts().reset_index(name="Counts").sort_values(by="Counts")

    # Label encoding
     def Label_Encode(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                label_encoder = LabelEncoder()
                self.df[col] = label_encoder.fit_transform(self.df[col])
                
     # Delete the cols   
     def Delete(self):
          return self.df.drop(['num_outbound_cmds'], axis = 1, inplace = True)
     
     # Read the top 5 rows again
     def Read_next(self):
          return self.df.head()
          
     # Select the features
     def Feature_select(self):
          return self.df.drop(['class'], axis = 1)
      
     # Select the target variable
     def Select_target_feature(self):
          return self.df['class']
     
     def select_features_using_rfe(self, n_features_to_select=10):
        """
        Select features using Recursive Feature Elimination (RFE) with a RandomForestClassifier.

        Parameters:
        n_features_to_select (int): The number of features to select. Default is 10.

        Returns:
        list: A list of selected feature names.
        """
        # Get the feature set and target variable
        X_train = self.Feature_select()
        Y_train = self.Select_target_feature()

        # Initialize the RandomForestClassifier
        rfc = RandomForestClassifier()
        
        # Initialize RFE with the RandomForestClassifier
        rfe = RFE(rfc, n_features_to_select=n_features_to_select)

        # Fit RFE
        rfe.fit(X_train, Y_train)

        # Create a feature map of selected features
        feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
        selected_features = [v for i, v in feature_map if i]

        return selected_features
     def prepare_data(self, selected_features, test_size=0.3, random_state=2):
        """
        Prepare the data by selecting features, standardizing, and splitting into train/test sets.

        Parameters:
        selected_features (list): List of selected feature names.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.3.
        random_state (int): Seed used by the random number generator. Default is 2.

        Returns:
        tuple: Split and standardized data (x_train, x_test, y_train, y_test).
        """
        # Select the features
        X = self.df[selected_features]
        Y = self.Select_target_feature()

        # Standardize the features
        scale = StandardScaler()
        X = scale.fit_transform(X)

        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        
        return x_train, x_test, y_train, y_test
     
    # Save results to CSV
     def Save_Results(self, results_dict, filename):
        results_df = pd.DataFrame()
        for key, value in results_dict.items():
            if isinstance(value, pd.DataFrame):
                value.columns = [f"{key}_{col}" for col in value.columns]
                results_df = pd.concat([results_df, value.reset_index(drop=True)], axis=1)
            else:
                results_df[key] = pd.Series(value)
        results_df.to_csv(filename, index=False)
