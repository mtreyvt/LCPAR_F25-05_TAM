#------------------------------------------------------------------------------
#'TimeGating.py'                                LCPART VT-ECE
#                                                      8Feb24
# Desc. of file here... 
#------------------------------------------------------------------------------
from gnuradio import analog
from gnuradio import blocks
from gnuradio import network
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.filter import window
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------
# Desc. of function here 
#------------------------------------------------------------------------------

def print_and_return_data(data):
    #print("Data: " + str(data))
    plt.plot(data)
    plt.show()
    return data
#--------------------------------------------------------------------------EoF

#------------------------------------------------------------------------------
# Desc. of function here 
# single row list to multiple rows for each frequency
#------------------------------------------------------------------------------

def format_data(data, num_freqs):
    formated_data = []
    row_length = len(data)//num_freqs
    for i in range(0, len(data), row_length):
    	formated_data.append(data[i:i+row_length])
    formated_data = np.array(formated_data)
    return formated_data
#--------------------------------------------------------------------------EoF
#------------------------------------------------------------------------------
# Desc. of function here 
# calculates the synthetic pulses, by Calculating the Fourier series coefficients for a square wave.
# needs the list of frequencies, not just the number of frequencies
#------------------------------------------------------------------------------
    
def synthetic_pulse(frequencies, duration):
    N = 1000  # Number of terms in the Fourier series
    t = np.linspace(0, duration, N, endpoint=False)  # Time array
    coefficients = []
    for freq in frequencies:
        # Calculate Fourier series coefficient for each frequency
        n = np.arange(1, N)
        coeff = 2 / (np.pi * n) * (1 - np.cos(2 * np.pi * freq * duration / (n * np.pi))) * np.sin(np.pi * n / 2)
        coefficients.append(np.sum(coeff))

    return np.abs(coefficients)
#--------------------------------------------------------------------------EoF

#------------------------------------------------------------------------------
# Desc. of function here 
# produce synthetic output by multiplying syntetic pulses by channle measurments
#------------------------------------------------------------------------------
    
def synthetic_output(pulse, data, num_freqs):
    data = np.array(data)
    output = np.zeros_like(data)
    for freq in range(data.shape[0]):
    	for i in range(data.shape[1]):
    		output[freq,i] = data[freq,i]*pulse[freq]
    plt.plot(output[0,:])
    plt.show()
    return output


#------------------------------------------------------------------------------
# Desc. of function here 
# format data and then preform the IFFT
#------------------------------------------------------------------------------

def to_time_domain(data, num_freqs):
    #frequency_domain_data = np.array(format_data(data, num_freqs))
    time_domain_data = np.fft.ifft(data.T, axis =1) #shoul it be .T
    plt.plot(time_domain_data[0,:]) #will only plot real portion
    plt.show()
    return time_domain_data
#--------------------------------------------------------------------------EoF

#------------------------------------------------------------------------------
# Desc. of function here 
#------------------------------------------------------------------------------

def time_gate(data):
    #print("Data: " + str(data))
    for i in range(len(data)):
    	if(i>30 or i<10):
    		data[i] = np.complex128(0+0j)
    return data
    
    
#--------------------------------------------------------------------------EoF
