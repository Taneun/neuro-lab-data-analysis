import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
"""
for this code to work you need to save the csv files with the following format:
<direction><degree>_<trail number>.csv
direction - L or R
Degree - 1, 2, 3
trail number - 1, 2, 3...
for example: L1_1.csv in "EYES" folder
also, you need to have the calibration file
in addition, you need 3 folders: eyes, head, and both
"""


def preprocess_data(file_path):
    """
    Preprocess the data from the CSV file
    :param file_path:
    :return:
    """
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Rename the columns
    df.columns = ['time', 'eyes_volt', 'head_volt', 'chair_volt']

    # Convert the appropriate columns to numeric
    df['seconds'] = pd.to_numeric(df['time'])
    df['eyes_volt'] = pd.to_numeric(df['eyes_volt'])  # Corrected column name
    df['head_volt'] = pd.to_numeric(df['head_volt'])  # Corrected column name
    df['chair_volt'] = pd.to_numeric(df['chair_volt'])  # Corrected column name
    return df


def find_head_degree(df, trail_type, head_intercept, head_slope, parts=20):
    """
    Find the degree of the head movement based on the VOR correction
    :param df:
    :param trail_type:
    :param head_intercept:
    :param head_slope:
    :param parts:
    :return:
    """
    part_length = len(df) // parts
    end_block_mean = df.iloc[-part_length:]['eyes_volt'].mean()
    VOR_correction = 0
    if 'R' in trail_type:
        VOR_correction = df['eyes_volt'].max() - end_block_mean
    elif 'L' in trail_type:
        VOR_correction = df['eyes_volt'].min() - end_block_mean
    return ((VOR_correction - head_intercept) / head_slope) / 5


def calculate_vor_duration(df):
    """
    Calculate the duration of the VOR in the data
    :param df:
    :return:
    """
    vor_start, vor_end = find_spike_blocks(df, 'eyes_volt', for_vor=True)
    start_time = df.iloc[vor_start]['seconds']
    end_time = df.iloc[vor_end]['seconds']
    time_diff = end_time - start_time
    return time_diff


def calculate_part_means(df, spike_type, parts):
    """
    Calculate the mean of the spike type for each part of the data
    :param df:
    :param spike_type:
    :param parts:
    :return:
    """
    part_length = len(df) // parts
    potential_spikes = []
    for i in range(parts):
        start_index = i * part_length
        end_index = start_index + part_length
        part_mean = df.loc[start_index:end_index, spike_type].mean()
        potential_spikes.append((start_index, part_mean))
    return potential_spikes


def find_max_diff_indexes(potential_spikes):
    """
    Find the indexes of the maximum difference between the potential spikes
    :param potential_spikes:
    :return:
    """
    max_diff = 0
    max_diff_indexes = None
    for i in range(len(potential_spikes) - 1):
        diff = abs(potential_spikes[i + 1][1] - potential_spikes[i][1])
        if diff > max_diff:
            max_diff = diff
            max_diff_indexes = (i, i + 1)
    return max_diff_indexes


def find_stable_point(spike_data, spike_type, start_index, moving_average_window):
    """
    Find the stable point in the data
    :param spike_data:
    :param spike_type:
    :param start_index:
    :param moving_average_window:
    :return:
    """
    spike_data['moving_avg'] = spike_data[spike_type].rolling(window=moving_average_window).mean()
    stable_point = spike_data.index[-1]
    for i in range(len(spike_data) - moving_average_window):
        if abs(spike_data['moving_avg'].iloc[i + moving_average_window] - spike_data['moving_avg'].iloc[i]) < 1e-3:
            stable_point = i + start_index + moving_average_window
            break
    return stable_point


def find_vor_stable_point(spike_data, stable_point, moving_average_window):
    """
    Find the stable point in the VOR data
    :param spike_data:
    :param stable_point:
    :param moving_average_window:
    :return:
    """
    start_vor = stable_point
    end_vor = stable_point + 1
    for i in range(len(spike_data) - moving_average_window):
        if abs(spike_data['moving_avg'].iloc[i + moving_average_window] - spike_data['moving_avg'].iloc[i]) < 1e-3:
            end_vor = i + start_vor + moving_average_window
            break
    return start_vor, end_vor


def find_spike_blocks(df, spike_type, parts=20, for_duration=False, for_vor=False):
    """
    Find the blocks of data that contain the spike
    :param df:
    :param spike_type:
    :param parts:
    :param for_duration:
    :param for_vor:
    :return:
    """
    if spike_type not in df.columns:
        return None

    potential_spikes = calculate_part_means(df, spike_type, parts)
    max_diff_indexes = find_max_diff_indexes(potential_spikes)

    start_index = potential_spikes[max_diff_indexes[0]][0]
    end_index = potential_spikes[max_diff_indexes[1]][0]
    spike_data = df.loc[start_index:end_index].copy()

    moving_average_window = 7
    stable_point = find_stable_point(spike_data, spike_type, start_index, moving_average_window)

    if for_vor:
        moving_average_window = 20
        start_vor, end_vor = find_vor_stable_point(spike_data, stable_point, moving_average_window)
        return start_vor, end_vor

    if for_duration:
        return df.loc[start_index:stable_point]

    start_block = max(0, max_diff_indexes[0] - 3)
    end_block = min(max_diff_indexes[1] + 2, len(potential_spikes) - 1)

    before_spike_blocks = potential_spikes[start_block:max_diff_indexes[0]]
    after_spike_blocks = potential_spikes[max_diff_indexes[1]:end_block]

    return before_spike_blocks, after_spike_blocks


def find_spike(df, spike_type, parts=20):
    """
    Find the spike in the data
    :param df:
    :param spike_type:
    :param parts:
    :return:
    """
    before_spike_blocks, after_spike_blocks = find_spike_blocks(df, spike_type, parts)
    before_spike = np.mean([block[1] for block in before_spike_blocks])
    after_spike = np.mean([block[1] for block in after_spike_blocks])
    return after_spike - before_spike


def degree(trail_type):
    """
    Calculate the degree of the trail based on the trail type
    :param trail_type:
    :return:
    """
    deg = 0
    if 'L' in trail_type:
        deg = int((trail_type.split('_')[0][-1])) * (-5)
    elif 'R' in trail_type:
        deg = int((trail_type.split('_')[0][-1])) * 5
    return deg


def process_files(directory, spike_type, head_intercept=None, head_slope=None):
    """
    Process all the files in the given directory and return a DataFrame with the results
    :param directory:
    :param spike_type:
    :param head_intercept:
    :param head_slope:
    :return:
    """
    results = {}
    head_results = []
    eyes_results = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = preprocess_data(file_path)
            trail_type = filename.split('_')[0]

            # Debugging: Print the first few rows to verify the data
            # print(f"Processing file: {filename}")
            # print(df.head())
            # print(spike_type)

            if spike_type == 'eyes_volt':
                mean_diff = find_spike(df, spike_type)
                if mean_diff is not None:
                    if trail_type not in results:
                        results[trail_type] = []
                    results[trail_type].append(mean_diff)
            if spike_type == 'head_volt':
                head_results.append((trail_type, find_spike(df, spike_type),
                                     find_head_degree(df, trail_type, head_intercept, head_slope)))

    if spike_type == 'eyes_volt':
        for key in results:
            for test in range(len(results[key])):
                eyes_results.append((key, results[key][test], degree(key)))
        results_df = pd.DataFrame(sorted(eyes_results, key=lambda x: x[2]),
                                  columns=['Trail Type', 'Mean Difference', 'Degrees'])

    if spike_type == 'head_volt':
        results_df = pd.DataFrame(sorted(head_results, key=lambda x: x[2]),
                                  columns=['Trail Type', 'Mean Difference', 'Degrees'])

    return results_df


def get_df_from_file(file_path):
    """
    Read a CSV file and return a DataFrame
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ['seconds', 'eyes_volt', 'head_volt', 'chair_volt']
    return df


def calibrate_chair(calibration_file_path, angle=40):
    """
    Calibrate the chair using the calibration data
    :param calibration_file_path:
    :param angle:
    :return:
    """
    calibration_data = get_df_from_file(calibration_file_path)
    left_volt = calibration_data['chair_volt'].min()
    right_volt = calibration_data['chair_volt'].max()
    left_angle = (-angle, left_volt)
    right_angle = (angle, right_volt)
    results_df = pd.DataFrame([left_angle, right_angle],
                              columns=['Angle', 'Voltage'])
    return results_df


def create_graph(df, spike_type):
    """
    Create a graph showing the change in voltage as a function of eye or head movements
    :param df:
    :param spike_type:
    :return:
    """
    spike_name = ''
    if spike_type == 'head_volt':
        spike_name = 'Head Movements'
    elif spike_type == 'eyes_volt':
        spike_name = 'Eye Movements'
    elif spike_type == 'chair_volt':
        spike_name = 'Chair Movements'

    if spike_type in ['head_volt', 'eyes_volt']:
        x_values = df['Degrees']  # Movements in degrees
        y_values = df['Mean Difference']  # Change in voltage
    elif spike_type == 'chair_volt':
        x_values = df['Angle']
        y_values = df['Voltage']

    x_values_with_const = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_values_with_const).fit()
    intercept, slope = model.params

    # Format the equation of the line
    equation = f"y = {slope:.6f}x + {intercept:.6f}"

    if spike_type == 'chair_volt':
        fig = px.scatter(df, x='Angle', y='Voltage', trendline='ols',
                         title='Chair Calibration',
                         color_discrete_sequence=['darkseagreen'], trendline_color_override='darkblue')
    else:
        fig = px.scatter(df, x='Degrees', y='Mean Difference', trendline='ols',
                         title=f'Change in Voltage as a Function of {spike_name}',
                         color_discrete_sequence=['darkseagreen'], trendline_color_override='darkblue')

    fig.update_layout(
        xaxis_title=f'{spike_name} (degrees)',
        yaxis_title='Change in Voltage',
        template='plotly_white'
    )

    # Add the equation of the line as an annotation
    fig.add_annotation(
        x=0.5,
        y=0.95,
        xref="paper",
        yref="paper",
        text=equation,
        showarrow=False,
        font=dict(size=18, color="black", family="Arial, sans-serif"),
        bgcolor="darkseagreen",
        bordercolor="black",
        borderwidth=2, borderpad=6
    )
    # Show the axes in black
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black')

    # Show the plot
    fig.show()
    fig.write_image(f"{spike_type}_graph.png")
    return intercept, slope


def calculate_spike_duration(df, spike_type, block_parts=50):
    """
    Calculate the duration of the spike in the data
    :param spike_type:
    :param df:
    :param block_parts:
    :return:
    """
    if spike_type == 'head_volt':
        block_parts = 20
    spike_block = find_spike_blocks(df, spike_type, block_parts, for_duration=True)

    start_time = spike_block.iloc[0]['seconds'].copy()
    end_time = spike_block.iloc[-1]['seconds'].copy()

    time_diff = end_time - start_time

    return time_diff


def create_speeds_table(directory, spike_type, head_results=None):
    """
    Create a table showing the eye speeds for each trail type
    :param head_results:
    :param spike_type:
    :param directory:
    :return:
    """
    spike_durations_by_trail = {}

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = preprocess_data(file_path)
            trail_type = filename.split('_')[0][-2:]
            if trail_type not in spike_durations_by_trail:
                spike_durations_by_trail[trail_type] = []
            spike_durations_by_trail[trail_type].append(calculate_spike_duration(df, spike_type))

    results = []
    name = ''
    if spike_type == 'eyes_volt':
        name = "Eye"
    elif spike_type == 'head_volt':
        name = "Head"

    for key in spike_durations_by_trail:
        timing_avg = sum(spike_durations_by_trail[key]) / len(spike_durations_by_trail[key])
        degree_moved = get_degrees(key, spike_type, head_results)
        speed = abs(degree_moved / timing_avg)
        direction = f"Right {key[-1]}" if 'R' in key else f"Left {key[-1]}"
        results.append((direction, degree_moved, timing_avg, speed))
    eye_speeds_df = pd.DataFrame(sorted(results, key=lambda x: x[1]),
                                 columns=['Trail Type', 'Saccade amplitude (deg)', 'Mean Time', f'{name} Speed (deg/s)'])

    return eye_speeds_df


def get_degrees(trail_type, spike_type, head_results):
    if spike_type == 'eyes_volt':
        return degree(trail_type)
    elif spike_type == 'head_volt':
        for index, row in head_results.iterrows():
            if trail_type in row['Trail Type']:
                return row['Degrees']
        return None


def create_vor_table(directory):
    """
    Create a table showing the VOR duration for each trail type
    :param directory:
    :return:
    """
    vor_durations_by_trail = {}

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = preprocess_data(file_path)
            trail_type = filename.split('_')[0][-2:]
            if trail_type not in vor_durations_by_trail:
                vor_durations_by_trail[trail_type] = []
            vor_durations_by_trail[trail_type].append(calculate_vor_duration(df))

    results = []

    for key in vor_durations_by_trail:
        timing_avg = sum(vor_durations_by_trail[key]) / len(vor_durations_by_trail[key])
        speed = abs(degree(key) / timing_avg)
        direction = f"Right {key[-1]}" if 'R' in key else f"Left {key[-1]}"
        results.append((direction, degree(key), timing_avg, speed))
    vor_durations_df = pd.DataFrame(sorted(results, key=lambda x: x[1]),
                                    columns=['Trail Type', 'VOR amplitude (deg)', 'Mean Time', 'VOR Speed (deg/s)'])

    return vor_durations_df


def calculate_delay_between_spike_end_to_vor_start(directory):
    """
    Calculate the delay between the end of the spike and the start of the VOR
    :param df:
    :param spike_type:
    :param vor_type:
    :param parts:
    :return:
    """
    window_size = 15
    step_size = 0.001  # Adjust this value to increase the number of x-values shown

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = preprocess_data(file_path)
            trail_type = filename.split('_')[0][-2:]
            volt_smooth = np.convolve(df['eyes_volt'], np.ones(window_size) / window_size, mode='valid')
            time_smooth = np.arange(df['seconds'].min(), df['seconds'].max(), step_size)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_smooth, y=volt_smooth, mode='lines', name=f'{trail_type} Smoothed Data'))

            fig.update_layout(
                title=f'{trail_type} Smoothed Data',
                xaxis_title='Time (s)',
                yaxis_title='Voltage (V)')
            fig.show()

def part_a_1(chair_calibration, eye_directory, head_directory):
    """
    Processes eye movement data, head movement data, and chair calibration data.
    This function computes the mean voltage differences for eye and head movement data
    and calibrates the chair voltage data. It also generates and saves graphs visualizing
    the processed data, and returns the head movement results as a DataFrame.

    Parameters:
    chair_calibration (str): The directory or file containing chair calibration data.
    eye_directory (str): The directory containing eye movement data files.
    head_directory (str): The directory containing head movement data files.

    Returns:
    DataFrame: The DataFrame containing the processed head movement data results.

    This function performs the following tasks:
    1. Processes eye movement data from the given directory using `process_files` and saves the results to 'mean_differences_eyes.csv'.
    2. Uses `create_graph` to generate and save a graph for the eye movement data.
    3. Processes head movement data with `process_files`, applying calibration for eye voltage differences.
    4. Saves the processed head movement results to 'mean_differences_head.csv'.
    5. Generates and saves a graph for the head movement data.
    6. Calibrates the chair data with `calibrate_chair` and saves the results to 'calibration_results.csv'.
    7. Generates and saves a graph for the chair calibration data.

    Example:
    head_results = part_a_1('chair_calib_data', 'eye_data', 'head_data')
    """
    eye_results_df = process_files(eye_directory, 'eyes_volt')
    eye_results_df.to_csv('mean_differences_eyes.csv', index=False)
    eye_intercept, eye_slope = create_graph(eye_results_df, 'eyes_volt')
    head_results_df = process_files(head_directory, 'head_volt', eye_intercept, eye_slope)
    head_results_df.to_csv('mean_differences_head.csv', index=False)
    create_graph(head_results_df, 'head_volt')
    chair_results_df = calibrate_chair(chair_calibration)
    chair_results_df.to_csv('calibration_results.csv', index=False)
    create_graph(chair_results_df, 'chair_volt')
    return head_results_df

def part_a_4(both_directory, head_directory, head_results_df):
    """
    Computes and saves tables related to eye and head movement speeds, as well as
    VOR (Vestibulo-Ocular Reflex) speeds. It generates separate CSV files for
    each type of speed table and processes both eye and head movement data from
    the provided directories.

    Parameters:
    both_directory (str): The directory containing both eye and head movement data files.
    head_directory (str): The directory containing head movement data files.
    head_results_df (DataFrame): The DataFrame containing previously processed head movement data results.

    Returns:
    None: The function generates and saves CSV files, but does not return any value.

    This function performs the following tasks:
    1. Creates an eye saccade speed table from data in the `both_directory` and saves it to 'total_eye_speeds.csv'.
    2. Creates a VOR speed table using data from the `head_directory` and saves it to 'vor_speeds.csv'.
    3. Creates a head movement speed table using data from the `head_directory` and `head_results_df`, then saves it to 'head_speeds.csv'.

    Example:
    part_a_4('both_data', 'head_data', head_results)
    """
    all_saccade_avg_table = create_speeds_table(both_directory, 'eyes_volt')
    all_saccade_avg_table.to_csv('total_eye_speeds.csv', index=False)
    vor_speeds_table = create_vor_table(head_directory)
    vor_speeds_table.to_csv('vor_speeds.csv', index=False)
    head_speed_table = create_speeds_table(head_directory, 'head_volt', head_results_df)
    head_speed_table.to_csv('head_speeds.csv', index=False)


def part_a_2(eye_directory):
    """
    Computes and saves a table of eye movement speeds based on the data from the
    specified directory. It generates a CSV file containing the processed eye speed data.

    Parameters:
    eye_directory (str): The directory containing eye movement data files.

    Returns:
    None: The function generates and saves a CSV file, but does not return any value.

    This function performs the following task:
    1. Creates a table of eye movement speeds from the data in the `eye_directory` and saves it to 'eye_speeds.csv'.

    Example:
    part_a_2('eye_data')
    """
    eye_speeds_table = create_speeds_table(eye_directory, 'eyes_volt')
    eye_speeds_table.to_csv('eye_speeds.csv', index=False)


def main():
    # Replace 'your_*****_directory_path' with the path to the directory containing your CSV files
    # For example: '/Users/your_name/Downloads/Lab Results/part 1 - Saccade/1 - eye'
    eye_directory = 'your_eye_directory_path'
    # For example: '/Users/your_name/Downloads/Lab Results/part 1 - Saccade/2 - head'
    head_directory = 'your_head_directory_path'
    # For example: '/Users/your_name/Downloads/Lab Results/part 1 - Saccade/3 - chair'
    chair_calibration = 'your_chair_directory_path'
    # needs to be a folder with *all the first part* files
    # For example: '/Users/your_name/Downloads/Lab Results/both'
    both_directory = 'your_both_directory_path'

    # part A1
    head_results_df = part_a_1(chair_calibration, eye_directory, head_directory)

    # part A2
    part_a_2(eye_directory)

    #  part A4
    part_a_4(both_directory, head_directory, head_results_df)

    # activate it only when you want to see the graph
    # it opens all of them at once
    # calculate_delay_between_spike_end_to_vor_start(head_directory)

    print("Results saved!")


if __name__ == '__main__':
    main()
