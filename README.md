# A Machine Learning Approach to the Prediction of Tidal elevation      
                                                     
**Installation Instructions:**
------------------------------

![](https://github.com/AIChris-Qian/Master-Independent-Research-Project/blob/main/Code/Figure/logo.png)


Download **Python3**

Download **Jupyter Notebook**

    pip install NumPy
    pip install Pandas
    pip install Matplotlib


**User Guide:**
------------------------------

![](https://github.com/AIChris-Qian/Master-Independent-Research-Project/blob/main/Code/Figure/software%20architecture.png)


The above shown is the **software architecture**, which consists of `Data Preparation`, `Data Processing` and `Experimental Method`.

Afterwards, **open the programme** and **compile the uploaded code**, which will be demonstrated below. 

Open **Git Bash**

    cd Desktop
          
    git clone https://github.com/acse-2019/irp-acse-cq419.git
          
Open **Jupyter Notebook**

    / Desktop / irp-acse-cq419 / Code / Notebook Code
    

### 1. Data Preparation:

   Read data from text file which is in `/ irp-acse-cq419 / Code / data`, then transform it in to `DataFrame` format.

    Index, Hour, Elevation = read_data('../ data / filename')
    print(list_transform_csv(Index, Hour, Elevation, scale, number))
  
    x_list = list_transform_csv(Index, Hour, Elevation, scale, number)['Hour']
    y_list = list_transform_csv(Index, Hour, Elevation, scale, number)['Elevation']
    Figure = onefigure_plot(0, height, x_list, y_list, 'colour_name', 'label_name’, 'title_name', True)
    print(Figure)
    
    
### 2. Data Processing:  
 
   Smooth the data which is stored in the form of `DataFrame`, then estimate the periodicity of sample data by periodic analysis.
 
    csv_data = list_transform_csv(Index, Hour, Elevation, scale, number)
    time_series = data_smooth(csv_data['Elevation'])
    data_smooth = onefigure_plot(0, height, csv_data['Hour'], time_series, 'colour_name', label_name, ‘title_name', False)
    print(data_smooth)
    
    x_csv = csv_data['Hour'][: length]
    y_csv = time_series[: length]
    x_csv1 = csv_data['Hour'][number1: number1 + length]
    y_csv1 = time_series[number1: number1 + length]
    Periodic_Analysis = twofigure_plot(0, height, x_csv, y_csv, 'colour_name', 'Period 1', x_csv1, y_csv1, 'colour_name', 'Period 2', 'Tidal periodic analysis', False)
    print(Periodic_Analysis)

    x_nor = csv_data['Hour'][: new_number]
    y_nor = data_normalization(csv_data['Elevation'])[: new_number]
    new_plot = onefigure_plot(0, 1, x_nor, y_nor, 'colour_name', 'label_name’, 'title_name', True)
    print(new_plot)
    
    
### 3. Experimental Method:

Take the normalized sample data as input data, training it under the built BP neural network framework, and then select suitable historical data as the starting data for prediction. As a result, the normalized prediction data will be obtained. Finally, transform the normalized prediction data to normal prediction data.
    
    data_A = list_transform_csv(Index, Hour, Elevation, scale, number)
    data_A_elevation = data_A['Elevation']
    data_B = list_transform_csv(Index1, Hour1, Elevation1, scale, number)
    data_B_elevation = data_B['Elevation']
    
    new_data = prediction_historical_sample(data_A_elevation, data_B_elevation, total_number, index)
    training_sample, training_label = create_sample_and_label(data_A_elevation, total_number, num_residual, num_train_sample, num_histri_data, num_histri_data, num_histri_data, 'true')

    BP = BP_Neural_Network()
    BP.test_set(training_sample, training_label, input, hidden, output, steps, learning_rate, momentum_factor, error_convergence)
    
    normolized_prediction_list, normolized_prediction_length = iteration_prediction_list( BP, new_data, len(training_sample), number, index, 'true')
    prediction_list, prediction_length = data_transfer(data_A_elevation, normolized_prediction_list)
    
    x_new_data = data_A['Hour'][: prediction_length]
    y_new_data = prediction_list
    x_new_data1 = data_B['Hour'][: prediction_length]
    y_new_data1 = data_B_elevation[: prediction_length]
    BP_method = twofigure_plot(0, height, x_new_data, y_new_data, 'colour_name', 'Prediction', x_new_data1, y_new_data1, 'colour_name', 'Observation', 'Tidal elevation analysis', True)
    print(BP_method)
    
    
**Performance Analysis:**   
------------------------------

After finishing compiling the code in the software architecture, to further quantifying performance between prediction and observation, one new measurement indicator is   introduced, calling the **correlation coefficient**.

    method0 = correlation_coefficient(y_new_data1, y_new_data1)
    method1 = correlation_coefficient(y_new_data, y_new_data1)

    fig, ax1 = plt.subplots(1, figsize=(7, 7))
    fig.tight_layout(w_pad=4)
    
    ax1.plot(y_new_data1, y_new_data1, 'colour_name', label = '(Actual Line) correlation coefficient: {:.4f}'.format(method0), markersize=5)
    ax1.plot(y_new_data1, y_new_data, 'colour_name', label = '(BP Method) correlation coefficient: {:.4f}'.format(method1), markersize=5)


**Furthermore:**
------------------------------

For **compiling code details**, please refer to `Documentation`
   
    / irp-acse-cq419 / Code / Documentation 
    
    
For **sample testing results**, please refer to `HTML Testing` or `Notebook Testing`
   
    / irp-acse-cq419 / Code / HTML Testing
    / irp-acse-cq419 / Code / Notebook Testing
