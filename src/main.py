from opshift import Experiment


def main(f):
    shifts1 = Experiment(f)

    # print(sorted(shifts1.opshifts.vs.base_search('EDMO-36')))
    # print((shifts1.opshifts.vs.df['Base_Code']=='EDMO-36').values)

    #shifts1.subset(attribute_filter=['EDMO', 'METRO'], base_filter='EDMO-36')
    #print(sum(shifts1.mask))
    shifts1.vs().shift_matrix(duration=15, target_week=None)


if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt')
    print('ok')
