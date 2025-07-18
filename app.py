from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enables CORS globally

# Load the trained model and car dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
#car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    #companies = sorted(car['company'].unique())
    #car_models = sorted(car['name'].unique())
    #years = sorted(car['year'].unique(), reverse=True)
    #fuel_types = car['fuel_type'].unique()
    companies=['Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Force', 'Ford', 'Hindustan', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Land', 'Mahindra', 'Maruti', 'Mercedes', 'Mini', 'Mitsubishi', 'Nissan', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    car_models = ['Audi A3 Cabriolet', 'Audi A4 1.8', 'Audi A4 2.0', 'Audi A6 2.0', 'Audi A8', 'Audi Q3 2.0', 'Audi Q5 2.0', 'Audi Q7', 'BMW 3 Series', 'BMW 5 Series', 'BMW 7 Series', 'BMW X1', 'BMW X1 sDrive20d', 'BMW X1 xDrive20d', 'Chevrolet Beat', 'Chevrolet Beat Diesel', 'Chevrolet Beat LS', 'Chevrolet Beat LT', 'Chevrolet Beat PS', 'Chevrolet Cruze LTZ', 'Chevrolet Enjoy', 'Chevrolet Enjoy 1.4', 'Chevrolet Sail 1.2', 'Chevrolet Sail UVA', 'Chevrolet Spark', 'Chevrolet Spark 1.0', 'Chevrolet Spark LS', 'Chevrolet Spark LT', 'Chevrolet Tavera LS', 'Chevrolet Tavera Neo', 'Datsun GO T', 'Datsun Go Plus', 'Datsun Redi GO', 'Fiat Linea Emotion', 'Fiat Petra ELX', 'Fiat Punto Emotion', 'Force Motors Force', 'Force Motors One', 'Ford EcoSport', 'Ford EcoSport Ambiente', 'Ford EcoSport Titanium', 'Ford EcoSport Trend', 'Ford Endeavor 4x4', 'Ford Fiesta', 'Ford Fiesta SXi', 'Ford Figo', 'Ford Figo Diesel', 'Ford Figo Duratorq', 'Ford Figo Petrol', 'Ford Fusion 1.4', 'Ford Ikon 1.3', 'Ford Ikon 1.6', 'Hindustan Motors Ambassador', 'Honda Accord', 'Honda Amaze', 'Honda Amaze 1.2', 'Honda Amaze 1.5', 'Honda Brio', 'Honda Brio V', 'Honda Brio VX', 'Honda City', 'Honda City 1.5', 'Honda City SV', 'Honda City VX', 'Honda City ZX', 'Honda Jazz S', 'Honda Jazz VX', 'Honda Mobilio', 'Honda Mobilio S', 'Honda WR V', 'Hyundai Accent', 'Hyundai Accent Executive', 'Hyundai Accent GLE', 'Hyundai Accent GLX', 'Hyundai Creta', 'Hyundai Creta 1.6', 'Hyundai Elantra 1.8', 'Hyundai Elantra SX', 'Hyundai Elite i20', 'Hyundai Eon', 'Hyundai Eon D', 'Hyundai Eon Era', 'Hyundai Eon Magna', 'Hyundai Eon Sportz', 'Hyundai Fluidic Verna', 'Hyundai Getz', 'Hyundai Getz GLE', 'Hyundai Getz Prime', 'Hyundai Grand i10', 'Hyundai Santro', 'Hyundai Santro AE', 'Hyundai Santro Xing', 'Hyundai Sonata Transform', 'Hyundai Verna', 'Hyundai Verna 1.4', 'Hyundai Verna 1.6', 'Hyundai Verna Fluidic', 'Hyundai Verna Transform', 'Hyundai Verna VGT', 'Hyundai Xcent Base', 'Hyundai Xcent SX', 'Hyundai i10', 'Hyundai i10 Era', 'Hyundai i10 Magna', 'Hyundai i10 Sportz', 'Hyundai i20', 'Hyundai i20 Active', 'Hyundai i20 Asta', 'Hyundai i20 Magna', 'Hyundai i20 Select', 'Hyundai i20 Sportz', 'Jaguar XE XE', 'Jaguar XF 2.2', 'Jeep Wrangler Unlimited', 'Land Rover Freelander', 'Mahindra Bolero DI', 'Mahindra Bolero Power', 'Mahindra Bolero SLE', 'Mahindra Jeep CL550', 'Mahindra Jeep MM', 'Mahindra KUV100', 'Mahindra KUV100 K8', 'Mahindra Logan', 'Mahindra Logan Diesel', 'Mahindra Quanto C4', 'Mahindra Quanto C8', 'Mahindra Scorpio', 'Mahindra Scorpio 2.6', 'Mahindra Scorpio LX', 'Mahindra Scorpio S10', 'Mahindra Scorpio S4', 'Mahindra Scorpio SLE', 'Mahindra Scorpio SLX', 'Mahindra Scorpio VLX', 'Mahindra Scorpio Vlx', 'Mahindra Scorpio W', 'Mahindra TUV300 T4', 'Mahindra TUV300 T8', 'Mahindra Thar CRDe', 'Mahindra XUV500', 'Mahindra XUV500 W10', 'Mahindra XUV500 W6', 'Mahindra XUV500 W8', 'Mahindra Xylo D2', 'Mahindra Xylo E4', 'Mahindra Xylo E8', 'Maruti Suzuki 800', 'Maruti Suzuki A', 'Maruti Suzuki Alto', 'Maruti Suzuki Baleno', 'Maruti Suzuki Celerio', 'Maruti Suzuki Ciaz', 'Maruti Suzuki Dzire', 'Maruti Suzuki Eeco', 'Maruti Suzuki Ertiga', 'Maruti Suzuki Esteem', 'Maruti Suzuki Estilo', 'Maruti Suzuki Maruti', 'Maruti Suzuki Omni', 'Maruti Suzuki Ritz', 'Maruti Suzuki S', 'Maruti Suzuki SX4', 'Maruti Suzuki Stingray', 'Maruti Suzuki Swift', 'Maruti Suzuki Versa', 'Maruti Suzuki Vitara', 'Maruti Suzuki Wagon', 'Maruti Suzuki Zen', 'Mercedes Benz A', 'Mercedes Benz B', 'Mercedes Benz C', 'Mercedes Benz GLA', 'Mini Cooper S', 'Mitsubishi Lancer 1.8', 'Mitsubishi Pajero Sport', 'Nissan Micra XL', 'Nissan Micra XV', 'Nissan Sunny', 'Nissan Sunny XL', 'Nissan Terrano XL', 'Nissan X Trail', 'Renault Duster', 'Renault Duster 110', 'Renault Duster 110PS', 'Renault Duster 85', 'Renault Duster 85PS', 'Renault Duster RxL', 'Renault Kwid', 'Renault Kwid 1.0', 'Renault Kwid RXT', 'Renault Lodgy 85', 'Renault Scala RxL', 'Skoda Fabia', 'Skoda Fabia 1.2L', 'Skoda Fabia Classic', 'Skoda Laura', 'Skoda Octavia Classic', 'Skoda Rapid Elegance', 'Skoda Superb 1.8', 'Skoda Yeti Ambition', 'Tata Aria Pleasure', 'Tata Bolt XM', 'Tata Indica', 'Tata Indica V2', 'Tata Indica eV2', 'Tata Indigo CS', 'Tata Indigo LS', 'Tata Indigo LX', 'Tata Indigo Marina', 'Tata Indigo eCS', 'Tata Manza', 'Tata Manza Aqua', 'Tata Manza Aura', 'Tata Manza ELAN', 'Tata Nano', 'Tata Nano Cx', 'Tata Nano GenX', 'Tata Nano LX', 'Tata Nano Lx', 'Tata Sumo Gold', 'Tata Sumo Grande', 'Tata Sumo Victa', 'Tata Tiago Revotorq', 'Tata Tiago Revotron', 'Tata Tigor Revotron', 'Tata Venture EX', 'Tata Vista Quadrajet', 'Tata Zest Quadrajet', 'Tata Zest XE', 'Tata Zest XM', 'Toyota Corolla', 'Toyota Corolla Altis', 'Toyota Corolla H2', 'Toyota Etios', 'Toyota Etios G', 'Toyota Etios GD', 'Toyota Etios Liva', 'Toyota Fortuner', 'Toyota Fortuner 3.0', 'Toyota Innova 2.0', 'Toyota Innova 2.5', 'Toyota Qualis', 'Volkswagen Jetta Comfortline', 'Volkswagen Jetta Highline', 'Volkswagen Passat Diesel', 'Volkswagen Polo', 'Volkswagen Polo Comfortline', 'Volkswagen Polo Highline', 'Volkswagen Polo Highline1.2L', 'Volkswagen Polo Trendline', 'Volkswagen Vento Comfortline', 'Volkswagen Vento Highline', 'Volkswagen Vento Konekt', 'Volvo S80 Summum']
    years = [2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1995]
    fuel_types = ['Diesel', 'LPG', 'Petrol']

    companies.insert(0, 'Select Company')

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Extract form values
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    # Make prediction
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=[[car_model, company, year, driven, fuel_type]])

    prediction = model.predict(input_df)
    output = np.round(prediction[0], 2)

    return str(output)


if __name__ == '__main__':
    app.run(debug=True)
