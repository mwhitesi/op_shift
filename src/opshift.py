
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from io import StringIO
from csv import writer, QUOTE_NONE
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal


class Experiment:
    # Object to run OP Shift Scenarios

    def __init__(self, file_path):
        self.opshifts = OPShift(file_path)
        self.vehicle_ids = self.vs().shift_id_list()
        self._reset_mask()

    def subset(self, attribute_filter=None, base_filter=None):

        if base_filter is not None:
            self.mask = self.vs().base_search(base_filter)
        if attribute_filter is not None:
            self.mask = self.mask & \
                self.vs().attribute_search(attribute_filter, or_op=False)

    def vs(self):
        return(self.opshifts.vs)

    def _reset_mask(self):
        self.mask = np.array([True] * self.vs().df.shape[0])

    def delete(self, current_subset=None, id_list=None):

        if current_subset:
            delete_list = self.vehicle_ids[self.mask]
        elif id_list and len(id_list) > 0:
            delete_list = id_list
        else:
            raise Exception('Invalid parameter')

        # Send delete list to shift instance
        self.opshifts.delete(delete_list)

        self._reset_mask()


    def write(self, f):
        self.opshifts.write(f)

    def insert(self, shift_list):
        # Identify available ids
        n2 = len(shift_list)
        i = np.max(self.vs().df['Shift_Id'])
        curr = np.arange(0, i)
        avail = curr[~np.isin(curr, self.vs().df['Shift_Id'])]

        missing = n2 - len(avail)
        if len(avail) > 0:
            ids = np.concatenate((avail, np.arange(i+1, i+missing+1)))
        else:
            ids = np.arange(i+1, i+missing+1)

        for i, s in enumerate(shift_list):
            s['id'] = ids[i]

        self.opshifts.insert(shift_list)

        self._reset_mask()

    def to_opshift(self, f, duration, target_week):

        # Step 1: convert binary matrix to OP shift format
        sm = pd.read_csv(f, encoding='utf-8', index_col=0)
        #sm = sm.iloc[:, 0:-1]
        ncol = sm.shape[1]

        if ncol != 24*7*60/duration:
            raise Exception(
                'Incorrect shift matrix dimensions for duration {}'.format(
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

        shifts = []

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
            thislen = lens[0] * duration
            thisstart = dts[starts[0]]

            # Only consider 2 simple repeat patterns 1 day or 7 day
            if len(starts) == 7:
                # 7 starts in a week can be treated as 1 Day repeat pattern
                thisrc = 1
                thisrp = [0]
                stype = '7d'

            else:
                # Weekly Repeat
                thisrc = 7
                # Time deltas from start
                tds = [dts[s] - thisstart for s in starts]

                if(np.any(np.array([td.seconds for td in tds]) != 0)):
                    raise Exception('A shift must have the same start time ' +
                                    'every day')

                thisrp = [td.days for td in tds]
                stype = str(len(tds))+'d'

            sh = self._shift_hash(thisstart, thislen, thisrc, thisrp, stype)
            shifts.append(sh)

            # Record start/stops
            starts_totals[starts] += 1
            ends_totals[ends] += 1

        if daynight_shifts > 0:
            #self.assign_dn_starts(starts_totals, daynight_shifts, duration, 20, 80)
            shifts.extend(self.assign_dn_starts(starts_totals, 20, duration, 20, 80, dts))

        return(shifts)

    def _opshift_datetime(self, dt):
        deltasecs = (dt - datetime(1899, 12, 30))
        deltadays = Decimal(deltasecs.total_seconds()) / Decimal(86400)
        return(deltadays)

    def _opshift_len(self, l):
        dl = Decimal(int(l)) / Decimal(24*60)
        return(dl)

    def _round(self, fl, n=12):
        return('{0:.{p}f}'.format(fl, p=n))

    def _shift_hash(self, s, l, c, p, t):
        wd = s.weekday()
        mins = s.hour * 60 + s.minute

        if c == 1:
            key = np.repeat(mins, 7)
        else:
            key = np.zeros(7)
            for td in p:
                d = (wd + td) % 7
                key[d] = mins

        shift = {
            'start': self._opshift_datetime(s),
            'duration': self._opshift_len(l),
            'cycle': str(c),
            'pattern': ','.join([str(i) for i in p]),
            'key': key,
            'type': t
        }

        return(shift)

    def assign_dn_starts(self, shift_change_counts, new_shifts, duration,
                         first, last, dts):
        # Use periods with low start/stop counts to do shift change
        # for continuous shifts

        # These shifts are configured to be 12 hrs long
        # Start and stop need to occur between first and last
        hr12 = 12*60/duration

        # First day
        start_window = np.arange(first, last-hr12+1).astype(int)
        end_window = (start_window + hr12).astype(int)

        paired_counts = np.zeros(len(start_window))

        # Totals thru week
        for d in range(8):
            paired_counts = paired_counts + \
                shift_change_counts[start_window*d] + \
                shift_change_counts[end_window*d]

        remaining = new_shifts
        starts = []

        while remaining > 0:
            # Find the best position to assign shifts
            p = np.argmin(paired_counts)
            start = first+p

            day = self._shift_hash(dts[start], 12*60, 1, [0], '7d24')
            i = int(start+hr12)
            night = self._shift_hash(dts[i], 12*60, 1, [0], '7d24')

            starts.extend([day, night])

            paired_counts[p] = paired_counts[p] + 1
            remaining = remaining - 1

        return(starts)

    def assign_bases(self, shifts, base_names, do_plot):
        # Greedy algorithm to assign a shift to one of a number of start/stop
        # bases

        def custom_sort(a_hash, b_hash):
            a = a_hash['key']
            b = b_hash['key']
            i = np.nonzero(a)[0][0]
            j = np.nonzero(b)[0][0]

            if i == j:
                if a[i] < b[j]:
                    return -1
                if a[i] > b[j]:
                    return 1
                else:
                    return 0
            else:
                if i < j:
                    return -1
                if i > j:
                    return 1
                else:
                    return 0

        # Split into types
        shiftsets = defaultdict(list)
        for x in shifts:
            shiftsets[x['type']].append(x)

        n = len(base_names)

        # Sort sets
        for t, l in shiftsets.items():
            shiftsets[t] = sorted(l, key=functools.cmp_to_key(custom_sort))

        # Assign each set 1 by 1
        base_record = defaultdict(list)
        current_base = 0
        for t, l in shiftsets.items():
            for s in l:

                s['base'] = base_names[current_base]
                base_record[current_base].append(s['key'])
                current_base = (current_base + 1) % n

        if do_plot:
            self.plot_base_record(base_record)

    def plot_base_record(self, br):

        def flatten(wk):
            vals = []
            for d, mins in enumerate(wk):
                if mins != 0:
                    vals.append(d*24*60+mins)
            return vals

        for b, vlist in br.items():
            vals = [y for v in vlist for y in flatten(v)]
            br[b] = vals

        bins = np.arange(0,7*24*60,step=60)
        plt.hist(br.values(), bins, label=br.keys())
        plt.legend(loc='upper right')
        plt.show()

    def assign_unit_names(self, shifts, prefix, level, ids, types=None):
        for i, s in enumerate(shifts):
            if not types:
                hr = DataTable.datetime(s['start']).hour
                if hr >= 0 and hr <= 12:
                    t = '1'
                else:
                    t = '2'
            else:
                t = types[i]

            s['unit'] = '{}-{}{}{}'.format(prefix, t, level, ids[i])

    def assign_unit_names(self, shifts, prefix, level, ids, types=None):
        for i, s in enumerate(shifts):
            if not types:
                hr = DataTable.datetime(s['start']).hour
                if hr >= 0 and hr <= 12:
                    t = '1'
                else:
                    t = '2'
            else:
                t = types[i]

            s['unit'] = '{}-{}{}{}'.format(prefix, t, level, ids[i])

    def assign_attr(self, shifts, attr):
        if not isinstance(attr, list):
            raise Exception('Invalid attr parameter')

        for s in shifts:
            s['attr'] = attr

    def assign_delay(self, shifts, dname):
        if not self.opshifts.md.defined(dname):
            raise Exception('Unknown Mobilisation Delay parameter')
        for s in shifts:
            s['delay'] = dname

    def assign_events(self, shifts, ename):
        if not self.opshifts.e.defined(ename):
            raise Exception('Unknown Event Type parameter')
        for s in shifts:
            s['event'] = ename

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

    def delete(self, id_list):
        self.vs.delete(id_list)
        self.e.delete(id_list)

    def insert(self, shift_list):
        self.vs.insert(shift_list)
        self.e.insert(shift_list)

    def write(self, f):
        with open(f, 'w', newline='') as csvfile:
            csvfile.write('VEHICLE TASKS FILE\tV2.06\tUTF-8\r\n')
            csvfile.write('\r\n')
            csvfile.write('\r\n')
            self.md.write(csvfile)
            csvfile.write('\r\n')
            csvfile.write('\r\n')
            self.vs.write(csvfile)
            csvfile.write('\r\n')
            csvfile.write('\r\n')
            self.e.write(csvfile)
            csvfile.write('\r\n')
            csvfile.write('\r\n')


class DataTable:

    df = None

    def __init__(self, csvfile, cols=None, decimal_format_cols=None):
        if cols and decimal_format_cols:
            conv = {}
            for c in decimal_format_cols:
                conv[c] = self.decimal_from_value
            self.df = pd.read_csv(csvfile, encoding='utf-8', names=cols,
                                  index_col=False,
                                  converters=conv
                                  )
        elif cols:
            self.df = pd.read_csv(csvfile, encoding='utf-8', names=cols,
                                  index_col=False,
                                  )
        else:
            self.df = pd.read_csv(csvfile, encoding='utf-8', header=None,
                                  index_col=False)
        print(self.df.info())

    @staticmethod
    def datetime(flt):
        seconds = Decimal(flt - 25569) * Decimal(86400)
        return(DataTable._round_time(datetime.utcfromtimestamp(seconds), 60))

    @staticmethod
    def decimal_from_value(value):
        return Decimal(value)

    @staticmethod
    def _round_time(dt, round_to=60):
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds+round_to/2) // round_to * round_to
        return(dt + timedelta(0, rounding-seconds, -dt.microsecond))

    def time(self, flt):
        seconds = timedelta(seconds=int(flt))
        d = datetime(1, 1, 1) + seconds
        return(d.days-1, d.hours, d.minutes, d.seconds)

    def write(self, fh):

        if 'Shift_id' in self.df:
            self.df.sort_values(by='Shift_Id')

        csvwriter = writer(fh,
                           delimiter='\t',
                           quoting=QUOTE_NONE,
                           lineterminator='\r\n',
                           quotechar='')
        csvwriter.writerows(self.header)

        df = self.df.copy()

        # Convert Decimal object to strings to control formatting
        for c in df.columns:
            if isinstance(df[c].iloc[1], Decimal):
                df[c] = df[c].apply(lambda x: '{0:.12f}'.format(x))

        df.to_csv(fh,
            sep='\t',
            na_rep='',
            header=False,
            index=False,
            quoting=QUOTE_NONE,
            quotechar='',
            line_terminator='\r\n',
            float_format='%.12f'
        )


class MobilisationDelaysTable(DataTable):

    header = (["[Mobilisation Delays]"],
              ("# Profile Name", "Call Class Attributes", "Distribution Type",
               "Offset", "Parameter 1", "Parameter 2", "Lower Bound", "Upper Bound"),
              ("# --------------------", "--------------------",
               "--------------------", "--------------------",
               "--------------------", "--------------------",
               "--------------------", "--------------------"))

    profile_names = dict()

    def __init__(self, csvfile):
        cols = ["Profile_Name", "Call_Class_Attributes",
                "Distribution_Type", "Offset", "Parameter_1",
                "Parameter_2", "Lower_Bound", "Upper_Bound"]
        deci_cols = ["Offset", "Parameter_1",
                     "Parameter_2", "Lower_Bound", "Upper_Bound"]
        super(MobilisationDelaysTable, self).__init__(csvfile, cols, deci_cols)

        self.profile_names = {name: 1 for name in
                              self.df.Profile_Name.unique()}

    def defined(self, profile):
        r = True if profile in self.profile_names else False

        return(r)

class VehicleShiftTable(DataTable):

    header = (["[Vehicle Shifts]"],
              ("# Shift Id", "Vehicle Name", "Base Code", "Start Date/Time",
               "Shift Duration", "Repeat Until Date", "Handover Delay",
               "Repeat Cycle", "Repeat Pattern", "Vehicle Attributes",
               "Mobilisation Delay", "Dispatch Priority", "Dispatch Region",
               "Associate Vehicle", "Allow Deployment",
               "Response Restrictions", "Notes Date", "Notes"),
              ("# --------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------",
                "--------------------", "--------------------"))

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
    deci_cols = [
        "Start_DateTime",
        "Shift_Duration",
        "Repeat_Until_Date",
        "Handover_Delay",
        "Notes_Date"
    ]

    def __init__(self, csvfile):
        super(VehicleShiftTable, self).__init__(csvfile, self.cols, self.deci_cols)

        # Make lookup for attributes
        d = defaultdict(list)
        for i, r in self.df[["Vehicle_Attributes", "Shift_Id"]].iterrows():
            a = r["Vehicle_Attributes"]
            s = r["Shift_Id"]
            alist = VehicleShiftTable._attr_parser(a)
            for a2 in alist:
                d[a2].append(s)

        self.attribute_lookup = {k: set(l) for k, l in d.items()}

    @staticmethod
    def _attr_parser(s):
        a = s[1:-1].split(',')
        return(a)

    def shift_id_list(self):
        return(self.df['Shift_Id'])

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

            # Convert all decimals to float
            for c in self.deci_cols:
                if c in row:
                    row[c] = float(row[c])

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

                        shift_range = pd.date_range(day_dt, end_dt,
                                                    freq=duration_str)
                        shift_range = shift_range[0:-1]  # Drop last value
                        shift_range = shift_range.floor(freq=duration_str)
                        sid = row['Shift_Id']
                        shiftmat.loc[shift_range, [sid]] = 1

                # Next cycle
                start_dt = start_dt + timedelta(days=row['Repeat_Cycle'])

        return(shiftmat)

    def delete(self, id_list):
        # Remove from attribute_lookup
        for id in id_list:
            r = self.df[self.df['Shift_Id'] == id]
            a = r["Vehicle_Attributes"].values[0]
            alist = VehicleShiftTable._attr_parser(a)
            for a2 in alist:
                self.attribute_lookup[a2].remove(id)

        self.df = self.df[~self.df['Shift_Id'].isin(id_list)]


    def insert(self, shift_list):
        for s in shift_list:
            self._insert_shift(s)

    def _insert_shift(self, sft):

        # Required
        new_shift = {
            "Shift_Id": sft['id'],
            "Vehicle_Name": sft['unit'],
            "Base_Code": sft['base'],
            "Start_DateTime": sft['start'],
            "Shift_Duration": sft['duration'],
            "Repeat_Cycle": sft['cycle'],
            "Repeat_Pattern": sft['pattern'],
            "Vehicle_Attributes": '{{{}}}'.format(','.join(sft['attr'])),
            "Mobilization_Delay": sft['delay']
        }

        # Optional with defaults
        new_shift["Handover_Delay"] = sft['handover'] if 'handover' in sft else Decimal(0)
        new_shift["Dispatch_Priority"] = sft['priority'] if 'priority' in sft else int(1)
        new_shift["Dispatch_Region"] = sft['region'] if 'region' in sft else None
        new_shift["Associate_Vehicle"] = sft['vehicle'] if 'vehicle' in sft else int(-1)
        new_shift["Allow_Deployment"] = sft['deploy'] if 'deploy' in sft else int(1)
        new_shift["Response_Restrictions"] = sft['restrict'] if 'restrict' in sft else '<RspPermitted NonePermitted="0"/>'
        new_shift["Notes_Date"] = sft['note_date'] if 'note_date' in sft else Decimal(0)
        new_shift["Notes"] = sft['note_date'] if 'note_date' in sft else None
        new_shift["Repeat_Until_Date"] = sft['until'] if 'until' in sft else Decimal(47484.000000000000)

        self.df = self.df.append(new_shift, ignore_index=True)

        for a in sft['attr']:
            self.attribute_lookup[a].add(sft['id'])


class ShiftEventTable(DataTable):

    lib = {
        'METRO_30': [
            {
                'Event_Type': 1,
                'Offset': Decimal(0.020833333333),
                'Offset_From': 0,
                'Parameters-->':
                    '<RspPermitted NonePermitted="0"><Filter><CallPriority><FilteredValueSet><Value>P1</Value><Value>P2</Value></FilteredValueSet></CallPriority></Filter></RspPermitted>',
                '': 0
            },
            {
                'Event_Type': 1,
                'Offset': Decimal(0.000694444444),
                'Offset_From': 0,
                'Parameters-->':
                    '<RspPermitted NonePermitted="1"/>',
                '': 0
            }]
    }

    header = (["[Events]"],
              ("# Shift Id", "Event Type", "Offset", "Offset From Start", "Parameters-->"),
              ("# --------------------", "--------------------", "--------------------",
               "--------------------", "--------------------", "--------------------"))

    def __init__(self, csvfile):

        cols = [
            "Shift_Id",
            "Event_Type",
            "Offset",
            "Offset_From",
            "Parameters-->",
            ""
        ]
        deci_cols = [
            "Offset"
        ]
        super(ShiftEventTable, self).__init__(csvfile, cols, deci_cols)

    def delete(self, id_list):
        self.df = self.df[~self.df['Shift_Id'].isin(id_list)]

    def defined(self, ename):
        r = True if ename in self.lib else False

        return(r)

    def insert(self, shift_list):
        for s in shift_list:
            e = self.lib[s['event']]
            for r in e:
                r['Shift_Id'] = s['id']
            self.df = self.df.append(e)
