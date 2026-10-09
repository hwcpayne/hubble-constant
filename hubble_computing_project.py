#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 21:29:29 2025

@author: henrypayne
"""
# Imports
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Defining constants
c = 2.9979e8
lambda_e = 486.1e-9

# Extracting the observation numbers and distances from the dist_data.txt file. Information on how to use len() .strip() and .split() was taken from online sources such as: https://www.mygreatlearning.com/blog/strip-in-python/#:~:text=Python%20strip()%20is%20a,copy%20of%20the%20original%20string. 

# Creating empty lists.
obs_numbers = []
distances = []

f = open("dist_data.txt") # Opens the distance file.

for line in f: # Goes through the file one line at a time.
    line = line.strip() # Removes trailing spaces for clean proccessing.
    if line == "" or line.startswith("#") or line.startswith("User_id"): # Checks for blank lines, comments, or headers.
        continue # These points are skipped so the program jumps back to the top of the loop and reads the next line. 
    parts = line.split() # Breaks the line into its columns, user_id, timestamp, observation_number, distance, instrument_response. https://realpython.com/python-split-string/#:~:text=split()%20method%20can%20split,breaks%20with%20the%20keepends%20parameter.
    if len(parts) < 5: # If the line is incomplete it is skipped. https://realpython.com/len-python-function/
        continue 
    if parts[4] != "1": # Selects only the good data.
        continue
    obs_numbers.append(int(parts[2])) # Stores the 3rd collumn observation numbers.
    distances.append(float(parts[3])) # Stores the 4th column distances.

f.close() # Closes the file


# Extract the frequency and intensity data into two separate arrays from the SpectralData_HBeta.csv file

spec_data = np.loadtxt("SpectralData_Hbeta.csv", delimiter=",", comments="#") # The file is a Comma-Separated-Values so this marks the commas as the separators and the hashtags as preceding the comments.

obs_header = spec_data[0, 0::2].astype(int) # Selects every 2nd column fromm the first row and creates an array of the values.

freqs_all = spec_data[1:, 0::2] # Starting at row 1 and continues through all remaining rows, and starts at collumn 0 and takes every second collumn to compile all the frequency values into an array.
ints_all  = spec_data[1:, 1::2] # Same as line before but for intensity, starting at column 0 since this is the first column with an inensity value.

def get_spectrum(obs_number): # Defines the get_spectrum function
    index = None # Creates a placeholder variable with no initial value
    for i in range(len(obs_header)): # Loops through every column position in the header https://www.geeksforgeeks.org/python/how-to-access-index-in-for-loop-python
        if obs_header[i] == obs_number: # Checkes whether the observation ID storeed at column i matches the desired one
            index = i # Stores the column index where the match was found
            break # Stops the column index where the match was found
    if index is None: # Checks if no match was found after scanning the entire header
        return None, None # Returns a failure signal (no spectrum for observation number)
    f = freqs_all[:, index] # Extracts the entire frequency column for that observation
    s = ints_all[:, index] # Same but for intensity
    return f, s # Returns two seperate arrays featuring the intensity and frequency for that observation
    

# Define the fit function

def fit_func(f, a, mu, sig, m, c0): # Defines the model function used for fitting
    gaus = a * np.exp(-(f - mu)**2 / (2 * sig**2))
    line = m * f + c0
    return gaus + line # Adds the gausian and linear lines together to produce the full intensity model that gets fitted to the data


# Get the recessional velocity value for each observation number

def velocity_from_obs_number(obs_num): # Defines the function that will return a galaxy's recessional velocity given its observation number.
    f, s = get_spectrum(obs_num) # Gets the frequency and intensity spectrum for the chosen galaxy
    if f is None: # Provides a route for the case where an observation number isnt found
        return None 

    max_index = 0 # Assumes tha the first point has the maximum intensity
    for i in range(1, len(s)): # Scans through all the intensity values from the first to the last value   https://sparkbyexamples.com/python/get-index-of-max-of-list-in-python
        if s[i] > s[max_index]: # If the current point it greater than the stored maximum
            max_index = i # Set the max index as equal to that value

    mu0  = f[max_index] # Initial guess for the line centre = frequency at highest intensity
    a0   = s[max_index] - s[0] # Initial guess for amplitude  = max - min intensity values
    sig0 = (f[-1] - f[0]) / 20 # Initial guess for width = total frequency range/20
    m0   = 0.0 # Initial guess for gradient of background line = flat
    c0   = s[0] # Initial guess for background intecept = smallest intensity value

    p0 = [a0, mu0, sig0, m0, c0] # Collects all the initial guesses into a single array

    # Iterate to find the parameter that produces the smallest overall error
    
    popt, pcov = curve_fit(fit_func, f, s, p0) # Fits the gaussian and straight line model to the spectrum
    
    mu = popt[1]

    lambda_obs = c / mu # Converts the fitted frequency to the observed wavelength
    z = (lambda_obs - lambda_e) / lambda_e # Calculates a redshift value
    v_ms = c * z # Converts the redshift value to a recessional velocituy
    v_kms = v_ms / 1000.0 # Converts to km/s

    return v_kms # Returns the galaxy's recessional velocity


# Pair the calcuated recessional velocity with the corresponding distance value for each obvervation

velocities = [] # Creates an empty list that will store the recessional velocity for each valid galaxy
final_distances = [] # Does the same but for final distances

for obs, dist in zip(obs_numbers, distances): # The Zip function pairs up the items from the two lists. The for loop goes through each new pair one by one. https://realpython.com/python-zip-function
    v = velocity_from_obs_number(obs) # Calls the function to compute the galaxy's recessional velocity from its spectrum
    if v is None: # Checks for any galaxy where a recessional velocity could not be calculated
        continue # Skips these error cases
    velocities.append(v) # Stored the computed velocity in the previously created empty velocity list
    final_distances.append(dist) # Does the same but for final distances
    
    
# Plot the final graph

plt.scatter(final_distances, velocities) # Plots a graph of each data point (the pairings created by the zip function)
    
coeffs, cov = np.polyfit(final_distances, velocities, 1, cov=True) # Fits the straight line v = H0D + b using the method of least squares, the slope and intercept are stored in the coeffs variable and the covariacnce matrix in the cov variable
x_line = np.linspace(min(final_distances), max(final_distances), 200) # Creates 200 evenly spaced x-values so that the fitted line is smooth
y_line = np.polyval(coeffs, x_line) # Evaluetes the best fit line at those 200 distances
plt.plot(x_line, y_line) # Plots the fitted Hubble line on top of the scatter

plt.xlabel("Distance of Galaxy from Earth (Mpc)") # Labels the x-axis
plt.ylabel("Recessional Veclocity of Distant Galaxy (km/s)") # Labels the y-axis
plt.grid() # Adds gridlines
plt.savefig("HubbleDiagram.png", dpi=200) # Saves the plot as an image file
plt.show() # Displays the graph


# Return the final value

H0 = coeffs[0] # Extracts the gradient from the first item in the coeffs variable
H0_err = np.sqrt(cov[0, 0]) # Extracts the top left value in the covariance matrix (variance) and square roots it to get the standard deviation
print("H0 =", H0, "+/-", H0_err, "km/s/Mpc") # Prints the measured Hubble constant with its error.