from opshift import Experiment


def main(f, f2):
    shifts1 = Experiment(f)

    shifts1.to_opshift(f2, duration=15, target_week='2017-11-04T00:00:00')




if __name__ == "__main__":
    main('data/raw/VehicleShifts_20181010.amb.txt', 'data/interim/test_base_36.csv')
    print('ok')
