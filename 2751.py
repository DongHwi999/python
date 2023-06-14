n = int(input())
num = [0] * n

for i in range(n):
    num[i] = int(input())

num.sort()
for j in num:
    print(j)