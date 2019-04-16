from opshift import Experiment
import pandas as pd
import numpy as np

def main(f, f2):
    shifts1 = Experiment(f)

    build_shift(shifts1, f2)

    #shifts1.to_opshift(f2, duration=15, target_week='2017-11-04T00:00:00')

    # shifts1.subset(attribute_filter=['EDMO','METRO'])
    # shifts1.delete(current_subset=True)

    # pd.set_option('precision', 12)
    # print(shifts1.vs().df.iloc[1, 3])
    # print(shifts1.vs().df.info())
    # shifts1.write('tmp.txt')


def build_shift(shifts, f):

    # Convert matrix to Optima Predict list format
    shift_list = shifts.to_opshift(f, duration=15, target_week='2017-11-04T00:00:00')

    # Add names to each unit
    shifts.assign_unit_names(shift_list, 'TSTA', 'A', np.arange(1, len(shift_list)+1))

    # Add attributes
    shifts.assign_attr(shift_list, ['EDMO', 'ALS', 'METRO'])

    # Add mobilization delay
    shifts.assign_delay(shift_list, 'StandardDelay')

    # Add event types
    shifts.assign_events(shift_list, 'METRO_30')

    # Add base assingment to shifts by distributing shifts in order of start
    # time across a set number of basess
    shifts.assign_bases(shift_list, ['EDMO-400', 'EDMO-36', 'EDMO-42'], False)

    print(shift_list[1])

    shifts.insert(shift_list)


if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt', 'data/interim/test_base_36.csv')
    print('ok')
