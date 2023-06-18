import sys
from collections import deque

cmd_case_num = int(sys.stdin.readline())

queue = deque()
for _ in range(cmd_case_num):
    cmd_list = sys.stdin.readline().rstrip().split()

    if len(cmd_list) == 2:
        cmd, element = cmd_list
    else:
        cmd = cmd_list[0]

    if cmd == "push":
        queue.append(element)
        continue
    elif cmd == "pop":
        result = -1 if len(queue) == 0 else queue.popleft()
    elif cmd == "size":
        result = len(queue)
    elif cmd == "empty":
        result = 1 if len(queue) == 0 else 0
    elif cmd == "front":
        result = -1 if len(queue) == 0 else queue[0]
    elif cmd == "back":
        result = -1 if len(queue) == 0 else queue[-1]

    print(result)