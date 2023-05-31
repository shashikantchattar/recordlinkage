
import os
import csv
import re
import logging
import optparse
import pandas as pd

import dedupe
from unidecode import unidecode

def preProcess(column):

    column = unidecode(column)
    column = re.sub('\n', ' ', column)
    column = re.sub('-', '', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(",", '', column)
    column = re.sub(":", ' ', column)
    column = re.sub('  +', ' ', column)
    
    column = column.strip().strip('"').strip("'").lower().strip()
    
    if not column:
        column = None
    return column

def readData(filename):

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = dict([(k, preProcess(v)) for (k, v) in row.items()])
            # if clean_row['price']:
            #     clean_row['price'] = float(clean_row['price'][1:])
            data_d[filename + str(i)] = dict(clean_row)
            
    return data_d
            
if __name__ == '__main__':    
            
        optp = optparse.OptionParser() 
              
        optp.add_option('-v', '--verbose', dest='verbose', action='count',
                    help='Increase verbosity (specify multiple times for more)')
        
        (opts, args) = optp.parse_args()
        
        log_level = logging.WARNING
        
        if opts.verbose:
            if opts.verbose == 1:
                log_level = logging.INFO
            
            elif opts.verbose >= 2:
                log_level = logging.DEBUG
            
        logging.getLogger().setLevel(log_level)
    
        output_file = 'files/data_matching_output.csv'
        output_file_sorted = 'files/data_sorted_output.csv'
        settings_file = 'files/data_matching_learned_settings'
        training_file = 'files/data_matching_training.json'

        left_file = 'files/JSE_Instrument_file.csv'
        right_file = 'files/Refinitiv_Instrument_file.csv'

        print('importing data ...')
        data_1 = readData(left_file)
        data_2 = readData(right_file)

        # def descriptions():
        #     for dataset in (data_1, data_2):
        #         for record in dataset.values():
        #             yield record['description']
                
        if os.path.exists(settings_file):
                print('reading from', settings_file)
                with open(settings_file, 'rb') as sf:
                    linker = dedupe.StaticRecordLink(sf)
        else:
                fields = [
                    {'field': 'Stock Name', 'type': 'String'},
                    {'field': 'Company Name', 'type': 'String'},
                ]
                
                
        
                linker = dedupe.RecordLink(fields)
                print('qwertt')
                if os.path.exists(training_file):
                    print('reading labeled examples from ', training_file)
                    with open(training_file) as tf:
                        linker.prepare_training(data_1,
                                        data_2,
                                        training_file=tf,
                                        sample_size=15000)
                else:
                    print('dstr')
                    linker.prepare_training(data_1, data_2, sample_size=100)
                    print('red')
                print('starting active labeling...')

                dedupe.console_label(linker)

                linker.train()
            
                with open(training_file, 'w') as tf:
                    linker.write_training(tf)
                with open(settings_file, 'wb') as sf:
                    linker.write_settings(sf)
        print('clustering...')
        linked_records = linker.join(data_1, data_2, 0.0)
        print('# duplicate sets', len(linked_records))
        
        cluster_membership = {}
        for cluster_id, (cluster, score) in enumerate(linked_records):
            for record_id in cluster:
                cluster_membership[record_id] = {'Cluster ID': cluster_id,
                                             'Link Score': score}

        with open(output_file, 'w') as f:

            header_unwritten = True

            for fileno, filename in enumerate((left_file, right_file)):
                with open(filename) as f_input:
                    reader = csv.DictReader(f_input)

                    if header_unwritten:

                        fieldnames = (['Cluster ID', 'Link Score', 'source file']+
                                  reader.fieldnames)

                        writer = csv.DictWriter(f, fieldnames=fieldnames,extrasaction='ignore')
                        writer.writeheader()

                        header_unwritten = False

                    for row_id, row in enumerate(reader):

                        record_id = filename + str(row_id)
                        cluster_details = cluster_membership.get(record_id, {})
                        row['source file'] = fileno
                        row.update(cluster_details)

                        writer.writerow(row)
                        
            df = pd.read_csv(output_file, encoding='cp1252')
            print("Output")
            print(df)
            df.sort_values(by=['Cluster ID','Currency Code'],ascending=[True,False],inplace=True)
            print("Sorted Output")
            print(df)
            df.to_csv(output_file_sorted)
            df = df.drop_duplicates(['Cluster ID'])
