import csv


queries = []
count = 0
line = []
fr = open('../data/biker_rank_2.csv', 'r')
reader = csv.reader(fr)
for row in reader:
    if row[3] != '-1' and int(row[3])>int(row[4]):
        # tmp = row[:4]
        # if row[3] == '-1':
        #     tmp.append(-1)
        # else:
        #     tmp.append(row[4])
        # line.append(tmp)
        line.append(row)


fw = open('../data/biker_rank_res_2.csv', 'w', newline='')
writer = csv.writer(fw)
for li in line:
    writer.writerow(li)
fw.close()


