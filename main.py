import numpy as np
from flask import Flask, redirect, url_for, render_template, request, session
import requests
import urllib
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler
import os
import sys

# System level operations (like loading files)
sys.path.append(os.path.abspath("./model"))
from load import *  # Ensure you have this module for loading your model

# Initializing Flask app
app = Flask(__name__)
app.secret_key = 'my secret and not your secret'

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  # You can replace this with your email provider's SMTP server
SMTP_PORT = 587
EMAIL_SENDER = "vijayshankarvnair21@gmail.com"
EMAIL_PASSWORD = "qeidvajvgqmsvfxr"  # Or use app-specific password
EMAIL_SUBJECT = "Flood Risk Alert"

# Setup scheduler
scheduler = BackgroundScheduler()
scheduler.start()


# Store email in session
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/grantaccess', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        session['email'] = email  # Store email in session

        # Get the user's location and date
        location = request.form['location']
        year = request.form['date'][:4]
        datetemp = datetime.strptime(request.form['date'], '%Y-%m-%d').strftime('%d/%m/%y')

        # Fetch weather details for the user's location
        location_details = fetch_location_details(location)

        # If location details are found, send a detailed email
        if location_details:
            city, state, lat, lon = location_details
            weather_info = fetch_weather_details(lat, lon)
            flood_risk = get_flood_risk(weather_info['precipitation'])
            email_body = generate_initial_email_body(city, state, weather_info, flood_risk)
            send_email(email, email_body)

        # Redirect to appropriate response page based on flood risk
        if flood_risk == 'high':
            return redirect(url_for('response_page_high', location=location, date=datetemp, year=year,
                                    flood_risk=flood_risk, condition=weather_info['condition'], temp=weather_info['temp']))
        elif flood_risk == 'medium':
            return redirect(url_for('response_page_medium', location=location, date=datetemp, year=year,
                                    flood_risk=flood_risk, condition=weather_info['condition'], temp=weather_info['temp']))
        else:
            return redirect(url_for('response_page_low', location=location, date=datetemp, year=year,
                                    flood_risk=flood_risk, condition=weather_info['condition'], temp=weather_info['temp']))
    else:
        return redirect(url_for('index'))


def fetch_location_details(location):
    # Fetch the location details (latitude, longitude, city, state)
    url = 'http://dev.virtualearth.net/REST/v1/Locations?'
    key = 'AozIVsiQ675xXwo2NwGtEuv8vtcQ098NSmpCuV1QAl7nFQ9wfjtcwSI_gdbH4sZV'
    cr = 'IN'
    results = url + urllib.parse.urlencode(({'CountryRegion': cr, 'locality': location, 'key': key}))
    response = requests.get(results)
    parser = response.json()

    if parser['statusDescription'] == 'OK':
        location_data = parser['resourceSets'][0]['resources'][0]['address']
        lat = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][0]
        lon = parser['resourceSets'][0]['resources'][0]['point']['coordinates'][1]
        city = location_data['locality']
        state = location_data.get('adminDistrict', 'Unknown State')
        return city, state, lat, lon
    else:
        return None


def fetch_weather_details(lat, lon):
    # Fetch the current weather details
    key = 'e31020243ddd05cc3d37ad5f4816190f'
    current_weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}'
    current_weather_response = requests.get(current_weather_url)
    current_weather_data = current_weather_response.json()

    if current_weather_data['cod'] == 200:
        current_temp = current_weather_data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
        current_condition = current_weather_data['weather'][0]['description']
        humidity = current_weather_data['main']['humidity']
        pressure = current_weather_data['main']['pressure']
        wind_speed = current_weather_data['wind']['speed']

        # Fetch precipitation from forecast data
        forecast_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={key}'
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        total_precipitation = 0.0
        for entry in forecast_data['list']:
            if 'rain' in entry and '3h' in entry['rain']:
                total_precipitation += entry['rain']['3h']

        return {
            'temp': round(current_temp, 2),
            'condition': current_condition,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': total_precipitation / 4  # Average precipitation for forecast
        }
    else:
        return None


def get_flood_risk(precipitation):
    # Determine risk level based on precipitation
    if precipitation > 50:  # Example threshold for high risk
        return "high"
    elif 20 < precipitation <= 50:  # Medium risk threshold
        return "medium"
    else:
        return "low"


def generate_initial_email_body(city, state, weather_info, flood_risk):
    # Generate the email body with the weather information and flood risk status
    if weather_info:
        temp = weather_info['temp']
        condition = weather_info['condition']
        humidity = weather_info['humidity']
        pressure = weather_info['pressure']
        wind_speed = weather_info['wind_speed']
        precipitation = round(weather_info['precipitation'], 2)

        return f"""
        Welcome to the Flood Prediction System!

        We have received your registration for the flood prediction system.

        Here are the current weather details for {city}, {state}:
        - Temperature: {temp}°C
        - Condition: {condition}
        - Humidity: {humidity}%
        - Pressure: {pressure} hPa
        - Wind Speed: {wind_speed} m/s
        - Precipitation: {precipitation} mm (forecasted)

        Based on the current weather data, the flood risk for your area is: {flood_risk}

        Stay tuned for your flood risk alert based on this weather data.

        Thank you!
        Flood Prediction System
        """
    else:
        return "Unable to fetch weather information. Please try again later."


def send_email(to_email, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = to_email
    msg['Subject'] = EMAIL_SUBJECT
    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, to_email, text)


@app.route('/response_page_low')
def response_page_low():
    location = request.args.get('location')
    date = request.args.get('date')
    year = request.args.get('year')
    flood_risk = request.args.get('flood_risk')
    condition = request.args.get('condition')
    temp = request.args.get('temp')
    return render_template('response0.html', location=location, date=date, year=year, flood_risk=flood_risk,
                           condition=condition, temp=temp)


@app.route('/response_page_medium')
def response_page_medium():
    location = request.args.get('location')
    date = request.args.get('date')
    year = request.args.get('year')
    flood_risk = request.args.get('flood_risk')
    condition = request.args.get('condition')
    temp = request.args.get('temp')
    return render_template('response1.html', location=location, date=date, year=year, flood_risk=flood_risk,
                           condition=condition, temp=temp)


@app.route('/response_page_high')
def response_page_high():
    location = request.args.get('location')
    date = request.args.get('date')
    year = request.args.get('year')
    flood_risk = request.args.get('flood_risk')
    condition = request.args.get('condition')
    temp = request.args.get('temp')
    return render_template('response2.html', location=location, date=date, year=year, flood_risk=flood_risk,
                           condition=condition, temp=temp)


if __name__ == "__main__":
    app.run(debug=True)
