import sys
read = sys.stdin.readline

N = int(read())
A = list(map(int, read().split()))
ans = []

for i,v in enumerate(A):
    ans.insert(v, i+1)
print(*reversed(ans))