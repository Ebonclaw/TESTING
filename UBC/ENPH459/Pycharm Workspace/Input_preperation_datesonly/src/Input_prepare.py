import pickle
import pandas as pd
import numpy as np
import time
# Time record
start_time = time.time()

# Tuning parameters
how_many_group = 5 # read how many group?
TIME_INTERVAL_DAYS = 10  # days
testing = False

# Define output classifications
OUTPUT_CLASSIFICATION_NUM = 5
class_0 = 0
class_1 = 1
class_2 = 5
class_3 = 10
class_4 = 15 # class 5 is retention that are more than 15 times

# CONSTANTS
data_path = '../Data/'
capture_datapath1 = data_path + '2016-05-18t09.35.32_capture_all.pkl'
output_file_path = data_path + "10days_processed_2016-05-18t09.35.32_capture_all.pkl"
ONE_DAY = 24*60*60 # seconds
TIME_INTERVAL = TIME_INTERVAL_DAYS * ONE_DAY  # seconds
INGROUP_FILTER = 0
INGROUP_TIME = 1


# filter indexing
filter_dict ={
    "NO FILTER": 0,
    "PRISM": 1,
    "WAVY": 2,
    "LIGHTSPEED": 3,
    "HUE": 4,
    "TINT": 5,
    "SLOW": 6,
    "DISSOLVE": 7,
    "TOON": 8,
    "PULP": 9,
    "MASK": 10,
    "POP": 11,
    "KALEIDO": 12,
    "FADED": 13,
    "DILATION": 14,
    "SILO": 15,
    "XY": 16,
    "DOT": 17,
    "PIXEL": 18,
    "POINT": 19,
    "HATCH": 20,
    "DRAW": 21,
    "SOBEL": 22,
    "COLOR": 23,
    "PINK": 24,
    "VHS 01": 25,
    "VHS 02": 26,
    "MELT": 27,
    "TRAILS": 28,
    "LIGHTLEAK": 29,
    "TILE": 30,
    "CHROMA": 31,
    "RGB": 32,
    "SWIRL": 33,
    "CONTRAST": 34,
    "DIAMOND": 35,
    "TWITCH": 36,
    "REFLECT": 37,
    "HAZE": 38,
    "DONT": 39,
    "WARM": 40,
    "VID MIX": 41,
    "Flip": 42,
    "PLIES": 43,
    "AMATORKA": 44,
    "VIGNETTE": 45,
    "CRISPYCOL": 46,
    "BULGE": 47,
    "SATURATION": 48,
    "Rotate": 49,
    "Shadows": 50,
    "Brightness": 51,
    "Highlights": 52,
    "FACEGLITCH": 53,
    "Audio Meter": 54,
    "MOSAIC": 55,
    "COLORFY": 56,
    "MILEY": 57
}
TOTAL_FILTER = len(filter_dict)


df = pd.read_pickle(capture_datapath1)

# group info's
id_based_group = df.groupby('distinct_id')
id_group_size = id_based_group.size()
first_day = id_based_group.nth(1)
comes_back = 0


# process data for each distinct_id, assume they are strictly in ascending order
grp_id = 0
# open pickle file
out_file = open(output_file_path, "wb")

for id, content in id_based_group:

    current_group = id_based_group["filters", "time"].get_group(id)
    
    # create trianing and output matrix of size 1x(filterxday)
    training_record = np.zeros((1,TOTAL_FILTER*TIME_INTERVAL_DAYS))
    training_size = 0 # record training set size
    output_record = np.zeros((1,OUTPUT_CLASSIFICATION_NUM)) # retention class matrix
    retention = 0

    print(int(grp_id/27537*100),'%')
    # print('size: ', id_group_size.iloc[grp_id])
    # Iterate through an id group element by element (with id chopped off) iloc[row, column]
    for element_id in range(0, id_group_size.iloc[grp_id]):

        ## print group info
        # print(current_group.iloc[element_id,0],current_group.iloc[element_id,1],)
        # print(current_group.iloc[element_id])
        capture_filter = current_group.iloc[element_id, INGROUP_FILTER]
        capture_time = current_group.iloc[element_id, INGROUP_TIME]

        # check filter data validity
        if len(capture_filter) == 0:
            capture_filter = 'NO FILTER'
        elif len(capture_filter) > 1 and ''.join(capture_filter[0]) == '':
            capture_filter = ''.join(capture_filter[1])
        else :
            capture_filter = ''.join(capture_filter[0])
        # check if filter in dictionary
        if capture_filter in filter_dict:
            capture_filter = capture_filter
        else:
            capture_filter = 'NO FILTER'

        # if we are in the training range, we store data to training set
        time_elapsed = capture_time - current_group.iloc[0, INGROUP_TIME]
        if time_elapsed < TIME_INTERVAL:
            training_size += 1
            day_number = int(time_elapsed/ONE_DAY) # starts from day 0
            training_record[0,TOTAL_FILTER*day_number+filter_dict[capture_filter]]+=1

        # if we are in output range, we add to retention record
        if time_elapsed > TIME_INTERVAL and time_elapsed < TIME_INTERVAL*2:
            retention += 1

    # Classification
    if retention == class_0:
        output_record[0, 0] = 1
    elif retention > class_1 and retention <= class_2:
        output_record[0, 1] = 1
    elif retention > class_2 and retention <= class_3:
        output_record[0, 2] = 1
    elif retention > class_3 and retention <= class_4:
        output_record[0, 3] = 1
    else :
        output_record[0, 4] = 1

    # pickle
    pickle.dump(training_record,out_file)
    pickle.dump(output_record,out_file)

    # print(output_record)
    # print(training_record)
    grp_id += 1

    if grp_id > how_many_group and testing:
        break
end_time = time.time()

out_file.close()

print('time used:', end_time - start_time)


