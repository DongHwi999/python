def gcd(a, b):
    if a % b:
        return gcd(b, a%b)
    return b

t = int(input())
for _ in range(t):
    arr = list(map(int, input().split()))
    result = 0
    for i in range(1, arr[0]+1):
        for j in range(i+1, arr[0]+1):
            result += gcd(max(arr[i], arr[j]), min(arr[i], arr[j]))
    print(result)