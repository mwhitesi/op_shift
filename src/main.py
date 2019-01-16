from opshift import Experiment


def main(f):
    shifts1 = Experiment(f)

    # print(sorted(shifts1.opshifts.vs.base_search('EDMO-36')))
    # print((shifts1.opshifts.vs.df['Base_Code']=='EDMO-36').values)

    shifts1.subset(attribute_filter=['EDMO', 'METRO'], base_filter='EDMO-36')

    sm = shifts1.vs().shift_matrix(duration=15, target_week=None, subset=shifts1.mask)

    sids = sm.columns.values
    names = shifts1.vs().df.loc[shifts1.vs().df['Shift_Id'].isin(sids),
                                ['Vehicle_Name']]
    sm.columns = names
    sm.to_csv('data/interim/base_36.csv')
    sm.sum(axis=1).to_csv('data/interim/base_36_metro_sums.csv')



if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt')
    print('ok')
