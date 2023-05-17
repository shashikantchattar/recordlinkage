import os
import csv
import re
import logging
import optparse
import pandas as pd

import dedupe
from unidecode import unidecode

def preProcess(column):
#
    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()

    if not column:
        column = None
    return column

def readData(filename):
#
    data_d = {}
    with open(filename, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(row)
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row['Id'])
            data_d[row_id] = dict(clean_row)

    return data_d

if __name__ == '__main__':


    
    optp = optparse.OptionParser()
    optp.add_option('-v', '--verbose', dest='verbose', action='count',
                    help='Increase verbosity (specify multiple times for more)'
                    )
    (opts, args) = optp.parse_args()
    log_level = logging.WARNING
    if opts.verbose:
        if opts.verbose == 1:
            log_level = logging.INFO
        elif opts.verbose >= 2:
            log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)

    file = 'files/sample'
    # file = 'files/training'
    input_file = file+'.csv'
    output_file = file+'_output.csv'
    output_file_sorted = file+'_sorted_output.csv'
    output_file_final = file+'_final_output.csv'
    settings_file = 'settings/learned_settings'
    training_file = 'settings/training_file.json'

    print('importing data ...')
    data_d = readData(input_file)

    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        # fields = [
        #     {'field': 'Id', 'type': 'String'},
        #     {'field': 'name', 'type': 'String'},
        #     {'field': 'address', 'type': 'String'},
        #     {'field': 'country', 'type': 'Exact', 'has missing': True},
        #     {'field': 'salary', 'type': 'String', 'has missing': True},
        #     ]

        fields = [
            {'field': 'Site Name', 'type': 'String'},
            {'field': 'Address', 'type': 'String'},
            {'field': 'ZIP', 'type': 'Exact', 'has missing': True},
            {'field': 'Phone Number', 'type': 'String', 'has missing': True},
            ]
        
        deduper = dedupe.Dedupe(fields)

        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data_d, f)
        else:
            deduper.prepare_training(data_d)

        print('starting active labeling...')

        dedupe.console_label(deduper)

        deduper.train()

        with open(training_file, 'w') as tf:
            deduper.write_training(tf)

        with open(settings_file, 'wb') as sf:
            deduper.write_settings(sf)

    print('clustering...')
    clustered_dupes = deduper.partition(data_d, 0.5)

    print('# duplicate sets', len(clustered_dupes))

    cluster_membership = {}
    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        for record_id, score in zip(records, scores):
            cluster_membership[record_id] = {
                "Cluster ID": cluster_id,
                "confidence_score": score
            }

    with open(output_file, 'w') as f_output, open(input_file, encoding='utf-8-sig') as f_input:

        reader = csv.DictReader(f_input)
        fieldnames = ['Cluster ID', 'confidence_score'] + reader.fieldnames

        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_id = int(row['Id'])
            row.update(cluster_membership[row_id])
            writer.writerow(row)

    # Sorting
    # df = pd.read_csv(output_file, encoding='utf-8-sig')
    df = pd.read_csv(output_file, encoding='cp1252')
    print("Output")
    print(df)
    df.sort_values(by=['Cluster ID','confidence_score'],ascending=[True,False],inplace=True)
    print("Sorted Output")
    print(df)
    df.to_csv(output_file_sorted)
    df = df.drop_duplicates(['Cluster ID'])
    print("Deduplicated Output")
    print(df)
    df.to_csv(output_file_final)

