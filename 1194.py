from collections import deque
import sys
input = sys.stdin.readline

def set_key(key, index):
    return key | (1 << index)

def check_door(key, index):
    return (1 << index) & key

dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]

N, M = map(int, input().split())
miro = [input() for _ in range(N)]

# find start point
def find_start():
    for i in range(N):
        for j in range(M):
            if miro[i][j] == '0':
                return i, j

sx, sy = find_start()

visited = [[[0]*(1<<6) for _ in range(M)] for __ in range(N)]
que = deque() # x,y,dist,keys

que.append([sx, sy, 0, 0])
visited[sx][sy][0] = 1

def bfs():
    while que:
        x, y, dist, key = que.popleft()

        for d in range(4):
            nkey = key
            nx = x + dx[d]
            ny = y + dy[d]
            if 0 <= nx < N and 0 <= ny < M and miro[nx][ny] != '#' and not visited[nx][ny][key]:
                np = ord(miro[nx][ny])
                if miro[nx][ny] == '1':
                    return dist+1
                elif 65 <= np <= 70 and not check_door(key, np - 65): # ord('F'), ord('A')
                        continue
                elif 97 <= np <= 102: # ord('f') ord('a')
                        nkey = set_key(key, np - 97)
                que.append([nx, ny, dist + 1, nkey])
                visited[nx][ny][nkey] = 1
    else:
        return -1

print(bfs())