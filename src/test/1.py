import csv

for row in csv.DictReader(open('../../Dataset/trainLabels.csv')):
    print("{}->{}".format(row['Id'], row['Class']))
    # if int() == label:
    #     result.append((row['Id']))