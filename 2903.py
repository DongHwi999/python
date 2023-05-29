N = int(input())

dot = '**'
k = 0

while k != N:
    k += 1
    dot = dot + '*' * (len(dot) - 1)

print(len(dot)*len(dot))