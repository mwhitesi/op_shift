from opshift import Experiment
import os
import pandas as pd
import numpy as np
import json
import sys

def main(configfile):

    with open(configfile) as jh:
        config = json.load(jh)

    config['shift_input'] = os.path.abspath(os.path.join('data', 'raw', 'VehicleShifts_20191011.amb.txt'))
    config['shift_output'] = os.path.abspath(os.path.join(config['rootdir'][0],
        'data', 'interim', 'opshift',
        'VehichShifts_'+str(config['scenario_name'][0])+'.amb.txt'))
    config['input'] = os.path.abspath(os.path.join(config['rootdir'][0], config['input'][0]))
    config['rootdir'] = os.path.abspath(config['rootdir'][0])

    build_shift(config)



def build_shift(config):

    shifts = Experiment(config['shift_input'])

    # Replace existing shifts with new shifts
    print(config['target_units'])
    shifts.subset(muni_filter=config['target_muni'],
        attribute_filter=config['target_attrs'])
    print(shifts.mask.sum())
    print(shifts.vs().df.shape)
    shifts.delete(current_subset=True)

    print(shifts.vs().df.shape)

    exit(0)

    # Convert matrix to Optima Predict list format
    shift_list = shifts.to_opshift(config['input'], duration=config['duration'][0],
        target_week=config['origin'][0])

    # Add names to each unit
    start_id = 1000
    shifts.assign_unit_names(shift_list, config['unit_prefix'][0], 'A',
        np.arange(start_id, len(shift_list)+start_id))

    # Add attributes
    shifts.assign_attr(shift_list, config['unit_attrs'])

    # Add mobilization delay
    shifts.auto_assign_delay(shift_list, 'StandardDelay', 'MixedFleetNight')

    # Add event types
    shifts.assign_events(shift_list, config['events'][0])

    # Add base assingment to shifts by distributing shifts in order of start
    # time across a set number of bases
    shifts.assign_bases(shift_list, config['bases'], False)

    # Add newly constructed shifts
    shifts.insert(shift_list)

    # Save shifts
    shifts.write(config['shift_output'])


if __name__ == "__main__":

    if(len(sys.argv) == 1):
        print("Missing config file argument")
        exit()

    main(sys.argv[1])
