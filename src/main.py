from opshift import Experiment
import pandas as pd
import numpy as np

def main(op_shift_file, opt_shift_file, pp_shift_file):

    dur = 15
    week = '2017-10-01T00:00:00'

    # Convert existing shifts to hub model
    opshifts = Experiment(op_shift_file)

    # Carve out EDMO METRO units
    opshifts.subset(attribute_filter=['EDMO','METRO'])
    sm = opshifts.shift_matrix(dur, week)
    smdt = pd.DataFrame(sm)
    smdt.to_csv('data/interim/opshift.csv',
        sep=',',
        header=True,
        index=True,
        float_format='%.1f'
     )

    # shifts1.delete(current_subset=True)
    #
    # build_shift(shifts1, f2)
    #
    # shifts1.write('tmp.txt')


def build_shift(shifts, f, unit_pre):

    # Convert matrix to Optima Predict list format
    shift_list = shifts.to_opshift(f, duration=15, target_week='2017-10-01T00:00:00')

    # Add names to each unit
    shifts.assign_unit_names(shift_list, unit_pre, 'A', np.arange(1, len(shift_list)+1))

    # Add attributes
    shifts.assign_attr(shift_list, ['EDMO', 'ALS', 'METRO'])

    # Add mobilization delay
    shifts.assign_delay(shift_list, 'StandardDelay')

    # Add event types
    shifts.assign_events(shift_list, 'METRO_30')

    # Add base assingment to shifts by distributing shifts in order of start
    # time across a set number of basess
    shifts.assign_bases(shift_list, ['EDMO-400', 'EDMO-36', 'EDMO-39'], False)

    print(shift_list[1])

    shifts.insert(shift_list)

    print(shifts.vs().df.shape)


if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt', 'data/interim/test_base_36.csv',
        'None')
    print('ok')
