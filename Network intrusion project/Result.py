from Data import Solution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\mk744\OneDrive - Poornima University\Documents\Python Scripts\Train_data.csv')
sol = Solution(df)

# Prepare results dictionary
results = {}

print("Label Encoder")
results['Label_Encode'] = sol.Label_Encode()
print(results['Label_Encode'], "\n")

print("Dropped Cols")
results['Delete'] = sol.Delete()
print(results['Delete'], '\n')

print("Read the file again")
results['Read_next'] = sol.Read_next()
print(results['Read_next'])

print("Feature selection")
results['Feature_select'] = sol.Feature_select()
print(results['Feature_select'], "\n")

print("Target feature selection")
results['Select_target_feature'] = sol.Select_target_feature()
print(results['Select_target_feature'])

selected_features = sol.select_features_using_rfe(n_features_to_select=10)
print(selected_features)

x_train, x_test, y_train, y_test = sol.prepare_data(selected_features)
print(x_train, x_test, y_train, y_test)
# Save results to a CSV file
# sol.Save_Results(results, 'Feature_Engineering_Results.csv')

# print("Results have been saved to 'Feature_Engineering_Results.csv'")



