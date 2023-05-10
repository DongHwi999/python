import sys
from collections import deque
input = sys.stdin.readline

N, K = map(int, input().split())
q = deque([i for i in range(1, N+1)])
result = []

while q:
    q.rotate(-1*(K-1))
    result.append(q.popleft())

print("<" + ", ".join(map(str, result)) + ">")