import sys

N = int(sys.stdin.readline())
word = []
for i in range(N):
    word.append(sys.stdin.readline().strip())
word = list(sset(word))
word.sort()
word.sort(key=lambda x : len(x))
print(*word, sep='\n')