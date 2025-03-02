# Eye and Head Movement Voltage Analysis

This repository contains Python code to analyze voltage changes in response to eye and head movements. The code processes CSV files containing voltage measurements and generates graphs that show the change in voltage as a function of eye or head movements, aiding in understanding the effects of eye or head motion on voltage readings.

## Requirements

Before running the code, ensure you have the following:
- A set of CSV files formatted as `<direction><degree>_<trail number>.csv`, where:
  - `direction`: "L" for left or "R" for right
  - `degree`: 1, 2, or 3 (indicating the degree of movement)
  - `trail number`: 1, 2, 3, etc. (indicating the trial number)
  
  For example: `L1_1.csv` should be placed in the "EYES" folder.
- A calibration file to calibrate the chair movements
- Three folders: `eyes`, `head`, and `both`, which contain the respective CSV files

## Dependencies

The following Python libraries are required to run this code:
- pandas
- numpy
- statsmodels
- plotly

You can install the necessary libraries with the following command:

```bash
pip install pandas numpy statsmodels plotly
```

## Functionality

The code provides several functions to preprocess, analyze, and visualize the data:

### 1. Data Preprocessing
- `preprocess_data(file_path)`: Reads a CSV file and processes it into a DataFrame with necessary columns for voltage analysis

### 2. VOR Correction Analysis
- `find_head_degree(df, trail_type, head_intercept, head_slope)`: Determines the degree of head movement based on the VOR correction
- `calculate_vor_duration(df)`: Calculates the duration of the VOR (Vestibulo-Ocular Reflex) in the dataset

### 3. Spike and Stable Point Detection
- `find_spike_blocks(df, spike_type, ...)`: Finds blocks of data containing spikes
- `find_stable_point(spike_data, spike_type, ...)`: Identifies the stable point after a spike
- `find_vor_stable_point(spike_data, stable_point, ...)`: Identifies the stable point during the VOR phase

### 4. Degree Calculation
- `degree(trail_type)`: Calculates the degree of the movement based on the trial type (left or right)

### 5. File Processing
- `process_files(directory, spike_type, ...)`: Processes all the CSV files in a specified directory, calculates the mean differences, and outputs a DataFrame with results

### 6. Calibration
- `calibrate_chair(calibration_file_path, angle=40)`: Calibrates the chair using data from the calibration file, providing voltage-angle mappings for left and right movements

### 7. Graph Creation
- `create_graph(df, spike_type)`: Creates a graph showing the change in voltage based on eye or head movements, with a linear regression trendline and equation annotation

## Usage

To use the code, follow these steps:

1. **Data Preparation**:
   - Ensure your CSV files are named correctly and stored in the appropriate folders (`eyes`, `head`, `both`)

2. **Calibration**:
   - Provide the path to your calibration file and calibrate the chair using the `calibrate_chair()` function

3. **Processing Files**:
   - Use the `process_files()` function to process the data in the desired directory
   - This will generate a DataFrame with the calculated results, including the mean differences and degrees for each trial

4. **Create Graphs**:
   - Use the `create_graph()` function to generate a graph for either eye or head movements, visualizing the change in voltage as a function of the degrees of movement

## Example

```python
# Example usage to process files and create a graph
directory = 'eyes'  # Path to the folder containing the CSV files
spike_type = 'eyes_volt'  # 'eyes_volt' or 'head_volt'
head_intercept = 10  # Example intercept for head movement (if applicable)
head_slope = 0.5  # Example slope for head movement (if applicable)

# Process the data
results_df = process_files(directory, spike_type, head_intercept, head_slope)

# Generate a graph
create_graph(results_df, spike_type)
```

## Graph Output

The code generates scatter plots with trendlines for voltage changes over movement degrees. The trendline equation is also displayed on the graph, showing the linear relationship between movement degrees and voltage changes.

## Notes

- The code assumes the CSV files are well-formatted and that the `spike_type` exists in the data columns (`eyes_volt`, `head_volt`, or `chair_volt`)
- The results are output as a DataFrame containing the trail type, mean voltage differences, and calculated degrees

## License

This project is licensed under the MIT License - see the LICENSE file for details.
