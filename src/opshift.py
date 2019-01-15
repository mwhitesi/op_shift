
import numpy as np
import pandas as pd
import re

from io import StringIO
from csv import writer
from collections import defaultdict
from datetime import datetime, timedelta


class Experiment:
    # Object to run OP Shift Scenarios

    def __init__(self, file_path):
        self.opshifts = OPShift(file_path)
        self.mask = np.array([True] * self.vs().df.shape[0])

    def subset(self, attribute_filter=None, base_filter=None):

        if base_filter:
            self.mask = self.vs().base_search(base_filter)
        elif attribute_filter:
            self.mask = self.mask & \
                self.vs().attribute_search(attribute_filter)

    def vs(self):
        return(self.opshifts.vs)

    def concurrent_counts(self, duration=15):
        # Weekly
        dailym = 24*60
        i = np.arange(7*24*60/duration)
        d = np.repeat(np.arange(1, 8), 24*60/duration)
        t = [str(timedelta(minutes=int(m % dailym))) for m in i*duration]
        cc = pd.DataFrame({'i': i, 'd': d,  't': t, 'count': 0})
        return(cc)


class OPShift:
    # Object to manipulate OP VehicleShifts input file

    md = None
    vs = None
    e = None

    def __init__(self, file_path):
        self.load(file_path)

    def load(self, file_path):

        # Create temporary in memory CSV files for each data type
        md_output = StringIO()
        vs_output = StringIO()
        e_output = StringIO()
        md_writer = writer(md_output)
        vs_writer = writer(vs_output)
        e_writer = writer(e_output)

        with open(file_path, 'r') as f:

            # Check file type using header
            h = f.readline()
            if not re.match(r'VEHICLE TASKS FILE\s+V2.06', h):
                raise Exception('Unsupported Vehicle Shift file format. '
                                'Expected: V2.06, Got: {}'.format(h))

            # Parse remainder of file
            state = 0
            # 1 = Mobilization Delays, 2 = Vehicle Shifts, 3 = Events
            for i, line in enumerate(f):
                line = line.strip()

                if re.match(r'^$|^#', line):
                    # Skip blank & comment lines
                    next

                elif re.match(r'^\[[a-zA-Z ]+]$', line):
                    # New state
                    if line == "[Mobilisation Delays]":
                        state = 1
                    elif line == "[Vehicle Shifts]":
                        state = 2
                    elif line == "[Events]":
                        state = 3
                    else:
                        raise Exception('Unrecognized section header: {}'.
                                        format(line))
                else:
                    # Data line
                    vals = line.split(sep='\t')
                    if state == 1:
                        # Mobilization Delays
                        md_writer.writerow(vals)
                    elif state == 2:
                        # Vehicle Shifts
                        vs_writer.writerow(vals)
                    elif state == 3:
                        # Events
                        e_writer.writerow(vals)

                # if i > 50:
                #     break

            md_output.seek(0)
            vs_output.seek(0)
            e_output.seek(0)

            self.md = MobilisationDelaysTable(md_output)
            self.vs = VehicleShiftTable(vs_output)
            self.e = DataTable(e_output)


class DataTable:

    df = None

    def __init__(self, csvfile, cols=None):
        if cols:
            self.df = pd.read_csv(csvfile, encoding='utf-8', names=cols,
                                  index_col=False)
        else:
            self.df = pd.read_csv(csvfile, encoding='utf-8', header=None,
                                  index_col=False)

    def datetime(self, flt):
        seconds = (flt - 25569) * 86400.0

        return(datetime.utcfromtimestamp(seconds))

    def time(self, flt):
        seconds = timedelta(seconds=int(flt))
        d = datetime(1, 1, 1) + seconds

        return(d.days-1, d.hours, d.minutes, d.seconds)


class MobilisationDelaysTable(DataTable):

    header = ("[Mobilisation Delays]\n"
              "# Profile Name  Call Class Attributes	Distribution Type"
              "	Offset	Parameter 1	Parameter 2	Lower Bound	Upper Bound\n"
              "# --------------------  --------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------\n")

    profile_names = dict()

    def __init__(self, csvfile):

        cols = ["Profile_Name", "Call_Class_Attributes",
                "Distribution_Type", "Offset", "Parameter_1",
                "Parameter_2", "Lower_Bound", "Upper_Bound"]
        super(MobilisationDelaysTable, self).__init__(csvfile, cols)

        self.profile_names = {name: 1 for name in
                              self.df.Profile_Name.unique()}


class VehicleShiftTable(DataTable):

    header = ("[Vehicle Shifts]\n",
              "# Shift Id	Vehicle Name	Base Code	Start Date/Time	"
              "Shift Duration	Repeat Until Date	Handover Delay	"
              "Repeat Cycle	Repeat Pattern	Vehicle Attributes	"
              "Mobilisation Delay	Dispatch Priority	Dispatch Region	"
              "Associate Vehicle	Allow Deployment	"
              "Response Restrictions	Notes Date	Notes\n"
              "# --------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------	"
              "--------------------	--------------------\n")

    def __init__(self, csvfile):

        cols = [
            "Shift_Id",
            "Vehicle_Name",
            "Base_Code",
            "Start_DateTime",
            "Shift_Duration",
            "Repeat_Until_Date",
            "Handover_Delay",
            "Repeat_Cycle",
            "Repeat_Pattern",
            "Vehicle_Attributes",
            "Mobilization_Delay",
            "Dispatch_Priority",
            "Dispatch_Region",
            "Associate_Vehicle",
            "Allow_Deployment",
            "Response_Restrictions",
            "Notes_Date",
            "Notes"
        ]
        super(VehicleShiftTable, self).__init__(csvfile, cols)

        #self.df.set_index("Shift_Id")

        # Make lookup for attributes
        d = defaultdict(list)
        for i, a in self.df["Vehicle_Attributes"].iteritems():
            alist = VehicleShiftTable._attr_parser(a)
            for a2 in alist:
                d[a2].append(i)

        self.attribute_lookup = {k: set(l) for k, l in d.items()}

    @staticmethod
    def _attr_parser(s):
        a = s[1:-1].split(',')
        return(a)

    def attribute_search(self, attr_list):
        rows = set()
        for a in attr_list:
            if a in self.attribute_lookup:
                s = self.attribute_lookup[a]
                rows = rows | s

        return(self.df['Shift_Id'].isin(rows).values)

    def base_search(self, base):
        return((self.df['Base_Code'] == base).values)

    def shift_matrix(self, duration, target_week):

        # dailym = 24*60
        # dailyi = dailym/duration
        # max = 7*24*60/duration
        # i = np.arange(max)
        # d = np.repeat(np.arange(1, 8), dailyi)
        # t = [str(timedelta(minutes=int(m % dailym))) for m in i*duration]
        #
        # mat = np.zeros((self.df.shape[0], max))
        # ts = pd.DataFrame({'i': i, 'd': d,  't': t})
        #
        cols = ['Start_DateTime', 'Shift_Duration', 'Repeat_Until_Date',
                'Repeat_Cycle', 'Repeat_Pattern']

        target_dt = datetime.strptime('2017-11-04T00:00:00',
                                      "%Y-%m-%dT%H:%M:%S")
        target_end_dt = target_dt + timedelta(days=7)
        duration_str = str(duration) + 'm'
        for index, row in self.df[cols].iterrows():

            first_dt = self.datetime(row['Start_DateTime'])
            last_dt = self.datetime(row['Repeat_Until_Date'])

            if first_dt > target_dt or last_dt < target_dt:
                continue

            f = str(row['Repeat_Cycle'])+'d'
            dt_range = pd.date_range(first_dt, target_dt, freq=f)

            start_dt = dt_range[-1]
            # Go through cycles overlapping range
            while(start_dt < target_dt+)
            end_dt = start_dt + timedelta(days=row['Shift_Duration'])
            pd.date_range(start_dt, end_dt, freq=duration_str)
