import sys
from collections import deque


def solution(n):
    queue = deque()

    step = 0
    queue.append((n, step))

    while True:
        now_n, now_step = queue.popleft()
        if now_n == 1:
            return now_step

        if now_n % 3 == 0:
            queue.append((now_n // 3, now_step + 1))
        if now_n % 2 == 0:
            queue.append((now_n // 2, now_step + 1))
        queue.append((now_n - 1, now_step + 1))


n = int(sys.stdin.readline())

result = solution(n)

print(result)