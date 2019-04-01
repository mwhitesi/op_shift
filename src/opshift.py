
import numpy as np
import pandas as pd
import re

from io import StringIO
from csv import writer
from collections import defaultdict
from datetime import datetime, timedelta, timezone


class Experiment:
    # Object to run OP Shift Scenarios

    def __init__(self, file_path):
        self.opshifts = OPShift(file_path)
        self.mask = np.array([True] * self.vs().df.shape[0])

    def subset(self, attribute_filter=None, base_filter=None):

        if base_filter is not None:
            self.mask = self.vs().base_search(base_filter)
        if attribute_filter is not None:
            self.mask = self.mask & \
                self.vs().attribute_search(attribute_filter, or_op=False)

    def vs(self):
        return(self.opshifts.vs)

    def insert():
        # Need naming scheme + shift id
        # Need Mobilisation Delay
        # Need Event modifiers
        # Need base
        pass

    def base_planner():
        pass

    def to_opshift(self, f, duration, target_week):

        # Step 1: convert binary matrix to OP shift format
        sm = pd.read_csv(f, encoding='utf-8', index_col=0)
        sm = sm.iloc[:, 0:-1]
        ncol = sm.shape[1]

        if ncol != 24*7*60/duration:
            raise Exception(
                'Incorrect shift matrix dimesions for duration {}'.format(
                    duration
                ))

        target_dt = datetime.strptime(target_week,
                                      "%Y-%m-%dT%H:%M:%S")
        target_end_dt = target_dt + timedelta(days=7)
        duration_str = str(duration) + 'min'
        dts = pd.date_range(
            target_dt, target_end_dt, freq=duration_str)[0:-1]

        daynight_shifts = 0
        starts_totals = ends_totals = np.zeros(sm.shape[1])

        for i, row in sm.iterrows():

            # Special shifts that need to be treated separately
            if np.all(row == 1):
                daynight_shifts += 1
                continue

            # Find status change positions
            idx = np.where(np.roll(row, 1) != row)[0]

            ln = len(idx)

            if ln == 0:
                raise Exception("0% or 100% shift not permitted")
            else:
                if row[idx[0]] != 1:
                    # start on a 'on' block
                    idx = np.roll(idx, -1)

                starts = idx[np.arange(0, ln, 2)]
                ends = idx[np.arange(1, ln, 2)]

            lens = ends - starts
            lens[lens < 0] = lens[lens < 0] + ncol

            if not np.all(lens == lens[0]):
                raise Exception(
                    "Variable shift block lengths: {}\nShift:\n{}\n".format(
                        ends - starts, row))
            if not len(starts) <= 7:
                raise Exception(
                    "Can't have multiple shifts per day.\n" +
                    "Shift starts found: {}".format(
                        starts))

            # Convert to minutes
            thislen = self._opshift_len(lens[0] * duration)
            thisstart = dts[starts[0]]

            # Only consider 2 simple repeat patterns 1 day or 7 day
            if len(starts) == 7:
                # 7 starts in a week can be treated as 1 Day repeat pattern
                thisrc = '1'
                thisrp = '0'

            else:
                # Weekly Repeat
                thisrc = '7'
                # Time deltas from start
                tds = [dts[s] - thisstart for s in starts]

                if(np.any(np.array([td.seconds for td in tds]) != 0)):
                    raise Exception('A shift must have the same start time ' +
                                    'every day')

                thisrp = ','.join([str(td.days) for td in tds])

            shift = {
                'start': self._opshift_datetime(thisstart),
                'duration': thislen,
                'cycle': thisrc,
                'pattern': thisrp
            }

            # Record start/stops
            starts_totals[starts] += 1
            ends_totals[ends] += 1

        self.assign_dn_starts(starts_totals, daynight_shifts)


    def _opshift_datetime(self, dt):
        deltasecs = (dt - datetime(1899, 12, 30))
        deltadays = deltasecs.total_seconds() / float(86400)
        return(self._round(deltadays))

    def _opshift_len(self, l):
        fl = float(l) / (24*60)
        return(self._round(fl))

    def _round(self, fl, n=12):
        return('{0:.{p}f}'.format(fl, p=n))


    def assign_dn_starts(self, st, dn_count):
        # Use periods with low start/stop counts to do shift
        # change on continuous shifts

        



    def shift_matrix(self, duration, target_week):
        # Weekly
        sm = self.vs().shift_matrix(duration, target_week, subset=self.mask)

        # Change column names
        sids = sm.columns.values
        names = self.vs().df.loc[
            self.vs().df['Shift_Id'].isin(sids)
            ]['Vehicle_Name'].tolist()

        sm.columns = names

        return(sm)


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
            self.e = ShiftEventTable(e_output)


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
        return(self._round_time(datetime.utcfromtimestamp(seconds), 60))

    @staticmethod
    def _round_time(dt, round_to=60):
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds+round_to/2) // round_to * round_to
        return(dt + timedelta(0, rounding-seconds, -dt.microsecond))

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

    def attribute_search(self, attr_list, or_op=True):
        rows = set()
        for a in attr_list:
            if a in self.attribute_lookup:
                s = self.attribute_lookup[a]
                if or_op or len(rows) == 0:
                    rows = rows | s
                else:
                    rows = rows & s

        return(self.df['Shift_Id'].isin(rows).values)

    def base_search(self, base):
        return((self.df['Base_Code'] == base).values)

    def shift_matrix(self, duration, target_week, subset=None):

        df = self.df
        if subset is not None:
            df = self.df.iloc[subset]

        target_dt = datetime.strptime(target_week,
                                      "%Y-%m-%dT%H:%M:%S")
        target_end_dt = target_dt + timedelta(days=7)
        duration_str = str(duration) + 'min'
        periods = pd.date_range(target_dt, target_end_dt, freq=duration_str)
        periods = periods[0:-1]
        shiftmat = pd.DataFrame(index=periods,
                                columns=df['Shift_Id'])
        shiftmat = shiftmat.fillna(0)

        cols = ['Shift_Id', 'Start_DateTime', 'Shift_Duration',
                'Repeat_Until_Date', 'Repeat_Cycle', 'Repeat_Pattern']
        for index, row in df[cols].iterrows():

            first_dt = self.datetime(row['Start_DateTime'])
            last_dt = self.datetime(row['Repeat_Until_Date'])

            if first_dt > target_dt or last_dt < target_dt:
                continue

            f = str(row['Repeat_Cycle'])+'d'
            dt_range = pd.date_range(first_dt, target_dt, freq=f)

            start_dt = dt_range[-1]
            cycle_days = [int(d) for d in row['Repeat_Pattern'].split(',')]
            # Go through cycles overlapping range
            while(start_dt < target_end_dt):
                # Iterate through days in cycle
                for d in cycle_days:
                    day_dt = start_dt + timedelta(days=d)
                    end_dt = day_dt + timedelta(days=row['Shift_Duration'])

                    if day_dt < target_end_dt and end_dt >= target_dt:
                        # Build shift range

                        if day_dt < target_dt:
                            day_dt = target_dt
                        if end_dt > target_end_dt:
                            end_dt = target_end_dt

                        # print(day_dt)
                        # print(end_dt)
                        shift_range = pd.date_range(day_dt, end_dt,
                                                    freq=duration_str)
                        shift_range = shift_range[0:-1]  # Drop last value
                        shift_range = shift_range.floor(freq=duration_str)
                        sid = row['Shift_Id']
                        shiftmat.loc[shift_range, [sid]] = 1

                # Next cycle
                start_dt = start_dt + timedelta(days=row['Repeat_Cycle'])

        return(shiftmat)


class ShiftEventTable(DataTable):

    se_lib = {
        'METRO_30': [
            {
                'Event_Type': 1,
                'Offset': 0.020833333333,
                'Offset_From': 0,
                'Parameters-->':
                    '<RspPermitted NonePermitted="0"><Filter><CallPriority><FilteredValueSet><Value>P1</Value><Value>P2</Value></FilteredValueSet></CallPriority></Filter></RspPermitted>',
                '': 0
            },
            {
                'Event_Type': 1,
                'Offset': 0.000694444444,
                'Offset_From': 0,
                'Parameters-->':
                    '<RspPermitted NonePermitted="1"/>',
                '': 0
            }]
    }

    def __init__(self, csvfile):

        cols = [
            "Shift_Id",
            "Event_Type",
            "Offset",
            "Offset_From",
            "Parameters-->",
            ""
        ]
        super(ShiftEventTable, self).__init__(csvfile, cols)

        #print(self.df.loc[1])
