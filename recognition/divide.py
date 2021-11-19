num_works = 1152
a = [x for x in range(num_works)]

num_workers = 5
single = num_works // num_workers

for i in range(num_workers):
    begin = i * single
    if i == num_workers - 1:
        end = num_works - 1
    else:
        end = begin + single - 1
    print(begin, end)