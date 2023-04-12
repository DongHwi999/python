import sys

num = int(sys.stdin.readline())
answer = 0

for _ in range(num):
    a, b = map(int,sys.stdin.readline().split())
    answer+=b%a

print(answer)