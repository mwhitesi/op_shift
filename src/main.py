from opshift import Experiment
import os
import pandas as pd
import numpy as np

def main(op_shift_file, sm_files, sm_unit_names):

    dur = 15
    week = '2017-09-24T00:00:00'
    op_file = 'data/interim/opshift.csv'

    # Convert existing shifts to hub model
    opshifts = Experiment(op_shift_file)

    # Carve out EDMO METRO units
    opshifts.subset(attribute_filter=['EDMO','METRO'])
    sm = opshifts.shift_matrix(dur, week)
    smdt = pd.DataFrame(sm)
    smdt = smdt.T
    smdt.to_csv(op_file,
        sep=',',
        header=True,
        index=True,
        float_format='%.1f'
    )
    sm_files.append(op_file)
    sm_unit_names.append('OPTA')

    for i,f in enumerate(sm_files):
        uname = sm_unit_names[i]
        build_shift(op_shift_file, f, uname, week)



def build_shift(shift_f, f, unit_pre, week):

    shifts = Experiment(shift_f)

    # Replace EDMO 911 shifts with new shifts
    shifts.subset(attribute_filter=['EDMO','METRO'])
    shifts.delete(current_subset=True)

    # Convert matrix to Optima Predict list format
    shift_list = shifts.to_opshift(f, duration=15, target_week=week)

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
    shifts.assign_bases(shift_list, ['EDMO-400', 'EDMO-200', 'EDMO-300'], False)

    # Add newly constructed shifts
    shifts.insert(shift_list)

    # Save shifts
    filenm = os.path.splitext(os.path.basename(f))[0]
    shifts.write(os.path.join('data', 'interim', filenm+'.amb.txt'))


if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt',
         ['data/raw/funded_unit_shift_matrix.csv', 'data/raw/optimized_block_shift_matrix.csv'],
         ['PROF', 'REGR'],
        )
    print('ok')
