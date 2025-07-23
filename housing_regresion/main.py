from pathlib import Path
import matplotlib.pyplot as plt
import funtions as f

housing = f.load_housing_data()

print("########################################################")
print(housing.info())

print("########################################################")
print(housing["ocean_proximity"].value_counts())

print("########################################################")
print(housing.describe())


housing.hist(bins=50, figsize=(12, 8))
plt.savefig("housing_histogram.png")



