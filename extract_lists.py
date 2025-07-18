import pandas as pd

car = pd.read_csv("Cleaned_Car_data.csv")

print("companies =", sorted(car['company'].unique()))
print("car_models =", sorted(car['name'].unique()))
print("years =", sorted(car['year'].unique(), reverse=True))
print("fuel_types =", sorted(car['fuel_type'].unique()))
