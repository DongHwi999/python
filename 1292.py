n1, n2 = map(int, input().split())

s = []

for i in range(1, n2+1):
    for j in range(i):
        s.append(i)

print(sum(s[n1-1:n2]))