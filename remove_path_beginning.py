import csv

csv_file_in = 'data/driving_log_in.csv'
csv_file_out = 'data/driving_log.csv'

with open(csv_file_in, 'r') as in_file:
    with open(csv_file_out, 'w') as out_file:
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)
    
        for i, row in enumerate(reader):
            # for dataset with a header
            # if (i == 0): continue

            new_row = row

            for j in range(3):
                path_split = new_row[j].split('/')
                new_row[j] = '/'.join([path_split[-2], path_split[-1]])
                new_row[j] = new_row[j].strip()
        
            writer.writerow(new_row)