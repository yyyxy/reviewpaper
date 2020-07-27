import csv

query = []
answer = []
fr = open('../data/new_biker_so_repositery1.csv', 'r')
reader = csv.reader(fr)
for row in reader:
    query.append(row[0])
    print(row[1])
    tmp = []
    count = 0
    for i in row[1].split("'"):
        if count%2 == 1:
            tmp.append(i)
            print(i)
        count += 1
    answer.append(tmp)

fw = open('../data/new_biker_so_repositery_new1.csv', 'w', newline='')
writer = csv.writer(fw)
for i in range(len(query)):
    writer.writerow([query[i]]+answer[i])
fw.close()

