from Data import Solution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\mk744\OneDrive - Poornima University\Documents\Python Scripts\Train_data.csv')
sol = Solution(df)

# Prepare results dictionary
results = {}

# Get and save the top five rows of the data
results['Top_five_rows'] = sol.Read_data()
print("Top five rows of the data : ")
print(results['Top_five_rows'], "\n")

# Get and save the shape of the data
results['Shape'] = sol.Shape_data()
print("Shape of the data")
print(results['Shape'], "\n")

# Get and save the list of columns
results['Column_list'] = sol.List_data()
print("Column in list")
print(results['Column_list'], "\n")

# Get and save the info of the data
results['Info'] = [sol.Info()]
print("Info of the data")
print(results['Info'], "\n")

# Get and save the descriptive statistics
results['Describe'] = sol.Describe()
print(sol.Describe(), "\n")

# Get and save the total number of rows
results['Total_number_of_rows'] = [sol.Total_Number_Rows()]
print("Total number of rows")
print(results['Total_number_of_rows'], "\n")

# Show the frequency graph of class and save it
print("Show the frequency graph of class")
sol.Count_of_Class()

# Get and save the counts of anomaly and normal
results['Anomaly_and_Normal_Counts'] = sol.Anomly_Normal_Counts()
print("Counts of Anomaly and Normal")
print(results['Anomaly_and_Normal_Counts'], "\n")

# Perform label encoding
sol.Label_Encode()

# Save results to a CSV file
sol.Save_Results(results, 'analysis_results.csv')

print("Results have been saved to 'analysis_results.csv' and 'class_frequency_plot.png'")
