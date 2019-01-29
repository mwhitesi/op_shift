from opshift import Experiment


def main(f):
    shifts1 = Experiment(f)

    # print(sorted(shifts1.opshifts.vs.base_search('EDMO-36')))
    # print((shifts1.opshifts.vs.df['Base_Code']=='EDMO-36').values)

    shifts1.subset(attribute_filter=['EDMO', 'METRO'])

    sm = shifts1.shift_matrix(duration=15, target_week='2017-11-04T00:00:00')

    sm.to_csv('data/interim/edmo_metro_shift_matrix.csv')
    #print(sm.sum(axis=1))



if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt')
    print('ok')
