def solution(n):
    count = 1
    li = [0]
    while True:
        if int("1" * count) % n == 0: break
        count += 1
    print(count)

while True:
    try:
        solution(int(input()))
    except:
        break