#------------------------------------------------------------------------------
#'RadioFunctions.py'                                Hearn WSU-ECE
#                                                   17apr23
# Open-Source Antenna Pattern Measurement System
#
# RadioFunctions-contains the helper functions for the radio system that are
# not part of any other class
#  
# Performs the following project-specific functions:
#   LoadParams = imports 'json' file with inputs
#   InitMotor   
#   OpenDatafile
#   rms
#   do_single
#   do_AMscan
#   do_AMmeas
#   do_NSmeas
#   get_plot_data
#   PlotFile()
#   PlotFiles()
#
#   RxRadio
#   Tx Radio
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# WSU-ECE legal statement here
#------------------------------------------------------------------------------
import numpy as np
from PlotGraph import PlotGraph
import json
import RxRadio
import TxRadio
import TimeGating
from MotorController import MotorController
from PolarPlot import plot_polar_patterns, plot_patterns
import matplotlib.pyplot as plt
import time
#------------------------------------------------------------------------------
def LoadParams(filename=None):
    """ Load parameters file
        parameters are a dictionary saved in a json file
        if filename is not given then a default file will be used
        any parameters not given in the file will be used from the default file
        if the file cannot be found it will raise an exception
    """
    try:
        defaults=json.load(open("params_default.json"))
    except Exception as e:
        print("params_default.json file is missing")
        raise e 
    if filename==None:
        return defaults
    try:
        params=json.load(open(filename))
    except Exception as e:
        print("Failed to load parameter file {:s}".format(filename))
        raise e
    #--------------------------------------------------------------------------
    # print(params)
    # go through all parameters given in the params file and
    # overwrite the defaults with any that are given
    #--------------------------------------------------------------------------
    for p in defaults:
        if p in params:
            defaults[p]=params[p]
        else:
            print("Parameter {:s} not specified in {:s} using default of ".format(p,filename),defaults[p])
    #--------------------------------------------------------------------------        
    # make sure freqency is within hackrf range
    #--------------------------------------------------------------------------
    if defaults["frequency"] < 30e6 or defaults["frequency"] > 6e9:
        #raise Excpetion("Frequency {:e} out of range".format(defaults["frequency"]))
        raise Exception("Frequency {:e} out of range".format(defaults["frequency"]))
    return defaults
#------------------------------------------------------------------------------
def InitMotor(params):
    motor_controller = MotorController(
    params["usb_port"],
    params["baudrate"])
    try:
        motor_controller.connect()
        print("Success: Motor controller fully connected.")
    except Exception as e:
        print("Error: Motor controller not responding, verify connections.")
        raise e
    motor_controller.reset_orientation()
    return motor_controller
#------------------------------------------------------------------------------
def OpenDatafile(params):
    filename= time.strftime("%d-%b-%Y_%H-%M-%S") + params["filename"]
    datafile_fp = open(filename, 'w')
    datafile_fp.write(params["notes"]+"\n")
    datafile_fp.write("% Mast Angle, Arm Angle, Background RSSI, Transmission RSSI\n")
    return datafile_fp
#------------------------------------------------------------------------------
def rms(data):
    """ return the rms of a data vector """
    return np.sqrt(np.square(data).mean())
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_single(Tx=True):
    params=LoadParams()
    if Tx:
        radio_tx_graph = TxRadio.RadioFlowGraph(
            params["tx_radio_id"], 
            params["frequency"], 
            params["tx_freq_offset"])
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"],
        numSamples=10000)
    if Tx:
        radio_tx_graph.start()
    radio_rx_graph.start()
    radio_rx_graph.wait()
    if Tx:
        radio_tx_graph.stop()
    rxd=radio_rx_graph.vector_sink_0.data()
    rxd2 =radio_rx_graph.vector_sink_1.data()
    plt.plot(rxd)
    plt.show()
    return rms(rxd)
    
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_singleTG(params):
    #params=LoadParams()
    Tx=True
    
    #create freq list based on params, needs to be 3 sig figs
    freq_list = np.linspace(params["lower_frequency"],
    	params["upper_frequency"], params["freq_steps"])
    freq_list_round = np.zeros_like(freq_list) #round frequency array to 3 sig figs
    for i,val in enumerate(freq_list):         #otherwise SDR will not measure data
    	rounding_factor = -int(np.floor(np.log10(np.abs(val)))-2)
    	freq_list_round[i] = round(val, rounding_factor)
    freq_list = freq_list_round
    rxd = np.array([])
    rxd_i = np.array([])
    for freq in freq_list:  #freq control
    	if Tx:
    	    radio_tx_graph = TxRadio.RadioFlowGraph(
    	        params["tx_radio_id"], 
    	        freq, 
    	        params["tx_freq_offset"])
    	radio_rx_graph = RxRadio.RadioFlowGraph(
    	    params["rx_radio_id"], 
    	    freq, 
    	    params["rx_freq_offset"],
    	    numSamples=10000)
    	if Tx:
    	    radio_tx_graph.start()
    	radio_rx_graph.start()
    	radio_rx_graph.wait()
    	if Tx:
    	    radio_tx_graph.stop()
    	rxd1=rms(radio_rx_graph.vector_sink_0.data())
    	rxd_i1=rms(radio_rx_graph.vector_sink_1.data())
    	rxd = np.append(rxd,rxd1)
    	rxd_i = np.append(rxd_i,rxd_i1)
    complex_rxd = np.vectorize(complex)(rxd, rxd_i) #creats complex vector of RSSI measurements
	
    #post prossesing script does this better
    TGavg = [[complex_rxd],[complex_rxd]]
    TGavg = np.squeeze(np.array(TGavg).T)
    pulses = TimeGating.synthetic_pulse(freq_list, 2.5e-8)
    print(pulses.shape)
    synth_out = TimeGating.synthetic_output(pulses, TGavg, params["freq_steps"])
    flipped_synth_out = np.fliplr(synth_out)
    double_sided_synth_out = np.concatenate((flipped_synth_out, synth_out), axis=0)
    print(double_sided_synth_out.shape)
    #TGantenna_data = TimeGating.to_time_domain(double_sided_synth_out, params["freq_steps"])
    TGantenna_data = TimeGating.to_time_domain(synth_out, params["freq_steps"])
    print(TGantenna_data.shape)
    data = np.fft.fft(TGantenna_data[0,:])#, axis = 1)
    print(data.shape)
    plt.plot(data)
    plt.show()

#    plt.plot(rxd)
#    plt.show()
#    plt.plot(
    return rms(rxd)
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def do_AMscan(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    time_gating_enabled = params["time_gate"]
    print("Time gating enabled: " + str(time_gating_enabled))
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"])
    AMantenna_data   = []
    radio_tx_graph.start()
    time.sleep(3)                                             # Tx latency
    print("Moving to start angle")
    motor_controller.rotate_mast(params["mast_start_angle"]);
    print("Collecting data while moving to end angle")
    radio_rx_graph.start()
    motor_controller.rotate_mast(params["mast_end_angle"]);
    radio_rx_graph.stop()
    radio_tx_graph.stop();                                    # stop Tx
    print("Finished collection, return to 0")                 #
    motor_controller.rotate_mast(0);                          # Reset AUT
    antenna_data=radio_rx_graph.vector_sink_0.data()
    if time_gating_enabled:
        antenna_data = TimeGating.print_and_return_data(antenna_data)
    n=len(antenna_data)
    print("read {:d} data_points".format(n))
    antenna_pow = np.square(antenna_data)
    numangles = params["mast_end_angle"]-params["mast_start_angle"] 
    binsize=int(n/numangles)
    print("binsize= {:d}".format(binsize))
    avg=np.zeros(numangles)
    for i in range(numangles):
        avg[i]=np.sqrt(np.square(
            antenna_data[i*binsize:(i+1)*binsize]).sum()/binsize)
    angles = range(int(params["mast_start_angle"]), int(params["mast_end_angle"]),1)
    arm_angle = np.zeros(len(avg));
    background_rssi = np.zeros(len(avg));
    plt.plot(antenna_pow)
    plt.show()
    plt.plot(avg)
    plt.show()
    print("avg {:d}".format(len(avg)),binsize)
    for i in range(len(avg)):
        datafile.write(
                str(angles[i]) + ',' + 
                str(arm_angle[i]) + ',' + 
                str(background_rssi[i]) + ',' + 
                str(avg[i]) + ','+
                str(arm_angle[i]) + '\n' #place holder
                )
        AMantenna_data.append((angles[i], arm_angle[i], 
            background_rssi[i], avg[i], arm_angle[i]))

    datafile.close();
    print("datafile closed")

    return AMantenna_data
#-----------------------------------------------------------------------------
#def do_AMscan_slow(params):
#------------------------------------------------------------------------------
def do_AMmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"],
        params["rx_freq_offset"], 
        numSamples=params["rx_samples"])
    antenna_data = []

    mast_angles = np.linspace(
        params["mast_start_angle"], 
        params["mast_end_angle"], 
        params["mast_steps"])
    arm_angles = np.linspace(params["arm_start_angle"], 
        params["arm_end_angle"], 
        params["arm_steps"])
    radio_tx_graph.start()
    time.sleep(3)                         # Tx latency 
    for mast_angle in mast_angles:        # azimuth control
        for arm_angle in arm_angles:      # elevation control (under constr)
            background_rssi = 0.0
            transmission_rssi = 0.0
            #
            print("Target Mast Angle: "+str(mast_angle))
            print("Target Arm Angle: "+str(arm_angle))
            print("Moving antenna...")
            motor_controller.rotate_mast(mast_angle)
            motor_controller.rotate_arm(arm_angle)
            print("Movement complete")
            #------------------------------------------------------------------
            # transmission rssi reading
            #------------------------------------------------------------------
            print("Taking transmitted signal sample...")
            radio_rx_graph.start()
            radio_rx_graph.wait()
            #radio_rx_graph.stop()
            # get data from the receiver and reset its output vector
            data=radio_rx_graph.vector_sink_0.data()
            radio_rx_graph.vector_sink_0.reset()
            radio_rx_graph.blocks_head_0.reset()
            #------------------------------------------------------------------
            #originally trimmed like this NW
            #data_points = delete(data_points, range(399000));
            #------------------------------------------------------------------
            print("read {:d} data_points".format(len(data)))
            transmission_rssi=np.sqrt(np.square(data).mean())
            print("Transmission RSSI: {:.3e}".format(transmission_rssi))
            print("Saving samples")
            datafile.write(
                str(mast_angle) + ',' + 
                str(arm_angle) + ',' + 
                str(background_rssi) + ',' + 
                str(transmission_rssi) + '\n'
                )
            antenna_data.append((mast_angle, arm_angle, 
                background_rssi, transmission_rssi))
    print("Returning mast and arm to home position...")
    motor_controller.rotate_mast(0)
    motor_controller.rotate_arm(0)
    print("Mast and arm should now be in home position")
    datafile.close();
    print("datafile closed")
    print("Scan completed")
    radio_tx_graph.stop()
    radio_tx_graph.wait()
    #
    return antenna_data
    
#==============================================================================

def do_AMTGmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    antenna_data = []
    mast_angles = np.linspace(
        params["mast_start_angle"], 
        params["mast_end_angle"], 
        params["mast_steps"])
    arm_angles = np.linspace(params["arm_start_angle"], 
        params["arm_end_angle"], 
        params["arm_steps"])
    #create a list of frequencies, needs to be rounded to 3 sig figs or'
    #radio doesn't work
    freq_list = np.linspace(params["lower_frequency"],
    	params["upper_frequency"], params["freq_steps"])
    freq_list_round = np.zeros_like(freq_list) #round frequency array to 3 sig figs
    for i,val in enumerate(freq_list):         #otherwise SDR will not measure data
    	rounding_factor = -int(np.floor(np.log10(np.abs(val)))-2)
    	freq_list_round[i] = round(val, rounding_factor)
    freq_list = freq_list_round
    print(freq_list)
 
    for freq in freq_list:                  # Freq Control
    	radio_rx_graph = RxRadio.RadioFlowGraph(
        	params["rx_radio_id"], 
        	freq,
        	params["rx_freq_offset"], 
        	numSamples=params["rx_samples"])
    	radio_tx_graph = TxRadio.RadioFlowGraph(
            	params["tx_radio_id"], 
        	freq, 
        	params["tx_freq_offset"])
    	radio_tx_graph.start()
    	time.sleep(3)                      # TX latency
    	    
    	for mast_angle in mast_angles:        # azimuth control
        	for arm_angle in arm_angles:      # elevation control (under constr)
            		background_rssi = 0.0
            		transmission_rssi = 0.0
            		
            		print("Target Mast Angle: "+str(mast_angle))
            		print("Target Arm Angle: "+str(arm_angle))
            		print("Moving antenna...")
            		motor_controller.rotate_mast(mast_angle)
            		motor_controller.rotate_arm(arm_angle)
            		print("Movement complete")
            		#------------------------------------------------------------------
            		# transmission rssi reading
            		#------------------------------------------------------------------
            		print("Taking transmitted signal sample...")
            		radio_rx_graph.start()
            		radio_rx_graph.wait()
            		#radio_rx_graph.stop()
            		# get data from the receiver and reset its output vector
            		data=radio_rx_graph.vector_sink_0.data()
            		data_i=radio_rx_graph.vector_sink_1.data()
            		min_length = min(len(data), len(data_i)) #make arrays equal length
            		antenna_data_complex = np.vectorize(complex)(data[:min_length], data_i[:min_length])
            		radio_rx_graph.vector_sink_0.reset()
            		radio_rx_graph.blocks_head_0.reset()
            		radio_rx_graph.vector_sink_1.reset()
            		radio_rx_graph.blocks_head_1.reset()
            		#------------------------------------------------------------------
            		#originally trimmed like this NW
            		#data_points = delete(data_points, range(399000));
            		#------------------------------------------------------------------
            		print("read {:d} data_points".format(len(data)))
            		transmission_rssi=np.sqrt(np.square(data).mean())
            		transmission_rssi_i=np.sqrt(np.square(data_i).mean())
            		rssi_i = np.vectorize(complex)(transmission_rssi, transmission_rssi_i)
            		print("Transmission RSSI: {:.3e}".format(transmission_rssi))
            		print("Saving samples")
            		datafile.write(
            		    str(mast_angle) + ',' + 
            		    str(arm_angle) + ',' + 
            		    str(background_rssi) + ',' + 
            		    str(rssi_i) + '\n' #this is the RSSI with the imaginary part in it
            		    #','.join(map(str,antenna_data_complex)) + '\n'
            		    )
            		antenna_data.append((mast_angle, arm_angle, 
            		    background_rssi, transmission_rssi, rssi_i))
    	print("Returning mast and arm to home position...")
    	motor_controller.rotate_mast(0)
    	motor_controller.rotate_arm(0)
    	print("Mast and arm should now be in home position")
#    	datafile.close();
#    	print("datafile closed")
#    	print("Scan completed")
    	radio_tx_graph.stop()
    	radio_tx_graph.wait()

#    print("Returning mast and arm to home position...")
#    motor_controller.rotate_mast(0)
#    motor_controller.rotate_arm(0)
#    print("Mast and arm should now be in home position")
    datafile.close()
    print("datafile closed")
    print("scan complete")
    #
    return antenna_data

#==============================================================================

def do_AMTGscan(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    AMantenna_data   = []
    
    # #creates a frequency list based on params, needs to be 3 sig figs
    # freq_list = np.linspace(params["lower_frequency"],
    # 	params["upper_frequency"], params["freq_steps"])
    # freq_list_round = np.zeros_like(freq_list) #round frequency array to 3 sig figs

    # #removing duplicates introduced by rounding so we scan once
    # freq_list = np.unique(freq_list_round)

    #new frequency rounding value
    freq_lin = np.linspace(params["lower_frequency"],
                           params["upper_frequency"],
                           params["freq_steps"])
    
    freq_rounded = np.array([round_sig(v,3) for v in freq_lin], dtype = float)
    freq_list = np.unique(freq_rounded)

    
    for i,val in enumerate(freq_list):         #otherwise SDR will not measure data
    	rounding_factor = -int(np.floor(np.log10(np.abs(val)))-2)
    	freq_list_round[i] = round(val, rounding_factor)
    freq_list = freq_list_round
    	
    for freq in freq_list:                  # Freq Control
    	radio_rx_graph = RxRadio.RadioFlowGraph(
        	params["rx_radio_id"], 
        	freq,
        	params["rx_freq_offset"])
    	radio_tx_graph = TxRadio.RadioFlowGraph(
            	params["tx_radio_id"], 
        	freq, 
        	params["tx_freq_offset"])
    	radio_tx_graph.start()
    	time.sleep(3)                                             # Tx latency
    	print("Moving to start angle")
    	motor_controller.rotate_mast(params["mast_start_angle"]);
    	print("Collecting data while moving to end angle")
    	radio_rx_graph.start()
    	motor_controller.rotate_mast(params["mast_end_angle"]);
    	radio_rx_graph.stop()
    	radio_tx_graph.stop();                                    # stop Tx
    	print("Finished collection, return to 0")                 #
    	motor_controller.rotate_mast(0);                          # Reset AUT
    	antenna_data=radio_rx_graph.vector_sink_0.data()#+1j*radio_rx_graph.vector_sink_1.data()
    	antenna_data_i = radio_rx_graph.vector_sink_1.data()    	
    	n=len(antenna_data)
    	print('hi')
    	print(n)
#    	print("read {:d} data_points".format(n))
#    	antenna_pow = np.square(antenna_data)
    	numangles = params["mast_end_angle"]-params["mast_start_angle"] 
    	binsize=int(n/numangles)
#    	print("binsize= {:d}".format(binsize))
    	avg=np.zeros(numangles)
    	avg_i=np.zeros(numangles)
    	for i in range(numangles):
        	#avg[i]=antenna_data[int(i*binsize/2)]
        	#avg_i[i]=antenna_data_i[int(i*binsize/2)]
        	avg[i]=np.sqrt(np.square(
        	    antenna_data[i*binsize:(i+1)*binsize]).sum()/binsize)
        	avg_i[i]=np.sqrt(np.square(
        	    antenna_data_i[i*binsize:(i+1)*binsize]).sum()/binsize)
    	angles = range(int(params["mast_start_angle"]), int(params["mast_end_angle"]),1)
    	arm_angle = np.zeros(len(avg));
    	background_rssi = np.zeros(len(avg));
    	complex_avg = np.vectorize(complex)(avg, avg_i)
    	min_length = min(len(antenna_data), len(antenna_data_i)) #make arrays equal length
    	#antenna_data_complex = np.vectorize(complex)(antenna_data[:min_length], antenna_data_i[:min_length])s
    	#print(antenna_data_complex.shape)
#    	plt.plot(antenna_pow)
#    	plt.show()
#    	plt.plot(avg)
#    	plt.show()
#    	print("avg {:d}".format(len(avg)),binsize)
    	for i in range(len(avg)):
        	datafile.write(
        	        str(angles[i]) + ',' + 
        	        str(arm_angle[i]) + ',' + 
        	        str(background_rssi[i]) + ',' + 
        	        str(complex_avg[i]) + ',' +
        	        #','.join(map(str,antenna_data_complex[i*binsize:(i+1)*binsize])) + '\n'
        	        str(avg[i]) + '\n' #changed
        	        )
        	AMantenna_data.append((angles[i], arm_angle[i], 
        	    background_rssi[i], avg[i], complex_avg[i]))

    #post prossesing script currently does this better
    pulses = TimeGating.synthetic_pulse(freq_list, 2.5e-8)
    TGdata = np.array(AMantenna_data)
    TGavg = TimeGating.format_data(TGdata[:,4],params["freq_steps"])
    TGantenna_data = TimeGating.synthetic_output(pulses, TGavg, params["freq_steps"])
    TGantenna_data = TimeGating.to_time_domain(TGantenna_data, params["freq_steps"])
    data = np.fft.fft(TGantenna_data, axis = 1)
    plt.plot(data[:,1])
    plt.title('FFT result')
    plt.show() 
    #polar plotting setup
    #take the data and put it into dB(gated_dB)
    #arrange our angles into the plot (deg)
    #plot the data 
    gated_db = _tg_data_to_dB(TGantenna_data, pick = "max")
    deg = np.arange(int(params["mast_start_angle"]),
                    int(params["mast_end_angle"]),1,dtype = float)
    plot_polar_patterns(
        deg,
        traces = [("Time_Gated",gated_dB)],
        rmin = -60.0, rmax = 0.0, rticks = (-60,-40,-20,0),
        title = "Radiation Pattern (Time-Gated-Polar)"
    )
    
    plt.plot(TGantenna_data)
    plt.show()
    datafile.close();
    print("datafile closed")

    return AMantenna_data



#==============================================================================
#------------------------------------------------------------------------------
# non-coherent noise-subtraction method (1st algorithm)
def do_NSmeas(params):
    motor_controller = InitMotor(params)
    datafile = OpenDatafile(params) 
    radio_tx_graph = TxRadio.RadioFlowGraph(
        params["tx_radio_id"], 
        params["frequency"], 
        params["tx_freq_offset"]) 
    radio_rx_graph = RxRadio.RadioFlowGraph(
        params["rx_radio_id"], 
        params["frequency"], 
        params["rx_freq_offset"], 
        numSamples=params["rx_samples"])
    antenna_data = []
    mast_angles = np.linspace(
        params["mast_start_angle"], 
        params["mast_end_angle"], 
        params["mast_steps"])
    arm_angles  = np.linspace(
        params["arm_start_angle"], 
        params["arm_end_angle"], 
        params["arm_steps"])

    for mast_angle in mast_angles:                           # azimuth
         for arm_angle in arm_angles:                        # elevation

             background_rssi = 0.0
             transmission_rssi = 0.0

             print("Target Mast Angle: "+str(mast_angle))
             print("Target Arm Angle: "+str(arm_angle))
             print("Moving antenna...")
             motor_controller.rotate_mast(mast_angle)
             motor_controller.rotate_arm(arm_angle)
             print("Movement complete")
             print("Taking background noise sample...")       # bkgrnd rssi 
             radio_rx_graph.start()
             radio_rx_graph.wait()
             #-----------------------------------------------------------------
             # get data from the receiver and reset its output vector
             # TODO the other scans use RMS, this just does average? 
             #-----------------------------------------------------------------
             data=radio_rx_graph.vector_sink_0.data()
             print("received {:d} background samples".format(len(data)))
             radio_rx_graph.vector_sink_0.reset()
             radio_rx_graph.blocks_head_0.reset()
             background_rssi = rms(data)
             #-----------------------------------------------------------------
             # Transmission rssi reading
             #-----------------------------------------------------------------
             print("Taking transmitted signal sample...")
             radio_tx_graph.start()
             time.sleep(1.3)                                  # Tx latency
             radio_rx_graph.start()
             radio_rx_graph.wait()
             radio_tx_graph.stop()
             radio_tx_graph.wait()
             # get data from the receiver and reset its output vector
             data=radio_rx_graph.vector_sink_0.data()
             print("received {:d} transmitted samples".format(len(data)))
             radio_rx_graph.vector_sink_0.reset()
             radio_rx_graph.blocks_head_0.reset()
             #-----------------------------------------------------------------
             # TODO the other scans use RMS, this just does average? 
             #-----------------------------------------------------------------
             transmission_rssi = rms(data)
             #-----------------------------------------------------------------
             # write rssi readings to file print("Saving samples")
             #-----------------------------------------------------------------
             datafile.write(
                 str(mast_angle) + ',' + 
                 str(arm_angle) + ',' + 
                 str(background_rssi) + ',' + 
                 str(transmission_rssi) + '\n'
                 )
             print("Sample angle={:f} bkgnd={:e} received={:e}".format(
                 mast_angle, background_rssi,transmission_rssi))
             antenna_data.append((mast_angle, arm_angle, 
                background_rssi, transmission_rssi))
    print("Returning mast and arm to home position...")
    motor_controller.rotate_mast(0)
    motor_controller.rotate_arm(0)
    print("Mast and arm should now be in home position")
    datafile.close()
    return antenna_data
#------------------------------------------------------------------------------
# plot functions for menu
#------------------------------------------------------------------------------
def get_plot_data(text):
    dataPoint = 0
    fileData = []
    for dataString in text:
        dataPointString = ''
        dataTuple = []
        for tempChar in dataString:
            if tempChar == ',' or tempChar == '\n':
                dataPoint = float(dataPointString)
                dataTuple.append(dataPoint)
                dataPointString = ''
            else:
                dataPointString += tempChar
        fileData.append((dataTuple[0],dataTuple[1],dataTuple[2],dataTuple[3]))
    return fileData;
def PlotFile():
    fileName = input("Enter the name of the data to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph = PlotGraph(fileData, fileName)
    plot_graph.show()
def PlotFiles():
    fileName = input("Enter the name of first file to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph1 = PlotGraph(fileData, fileName)
    fileName = input("Enter the name of the second file to plot\n")
    fr = open(fileName)
    text = fr.readlines()
    fr.close()
    text.remove(text[0])
    text.remove(text[0])
    fileData = get_plot_data(text);
    plot_graph2 = PlotGraph(fileData, fileName)
    ax1 = plt.subplot(111, projection='polar')
    ax1.set_theta_zero_location("N")
    theta1 = [angle*(np.pi/180) for angle in plot_graph1.mast_angles]
    ax1.plot(theta1, plot_graph1.rssi, label="With Gain")
    ax2 = plt.subplot(111, projection='polar')
    ax2.set_theta_zero_location("N")
    theta2 = [angle*(np.pi/180) for angle in plot_graph2.mast_angles]
    ax2.plot(theta2, plot_graph2.rssi, label="No Gain", linewidth=1)
#    
    if plot_graph1.plot_in_db == 'y':
        ax1.set_rticks([-20,-16,-12,-8,-4,0]);
        ax2.set_rticks([-20,-16,-12,-8,-4,0]);
    plt.legend(loc="lower center", bbox_to_anchor=(1, 1))
    plt.show()

def _tg_data_to_dB (td: np.ndarray, pick: str = "max",idx: int | None = None) ->np.ndarray:
    import numpy as np
    mag = np.abs(td)
    if pick == "center":
        if idx is None: 
            idx = mag.shape[1]//2
            y = mag[:,idx]
    else: 
        y = mag.max(axis=1)
    y = y/(np.max(y) if np.max(y) > 0 else 1.0)
    return 20.0 * np.log10(np.clip(y, 1e-12, None))

def round_sig(x:float, sig: int = 3) -> float: 
    if not np.isfinite(x):
        return x
    if x == 0.0:
        return 0.0
    ndigits = int(sig - 1 - math.floot(math.log10(abs(x))))
    ndigits = max(-12, min(12, ndigits))
    return round(x, ndigits)
#--------------------------------------------------------------------------EoF

