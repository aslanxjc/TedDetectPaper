num = 8
h = 33

for i in range(1,9):
    print i,11111111111

new_lst = []
lst = [i*33 for i in range(1,9) ]
print lst

for i,v in enumerate(lst):
    if i==0:
        new_lst.append([0,v])
    else:
        new_lst.append([i*33,v])

print new_lst
