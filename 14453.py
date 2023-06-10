# H > S, S > P, P > H
import sys

input = sys.stdin.readline

N = int(input().rstrip())

H = [0] * (N + 1)
P = [0] * (N + 1)
S = [0] * (N + 1)

array = [0]
for i in range(N):
    array.append(input().rstrip())

for i in range(1, N + 1):

    if array[i] == 'H':
        H[i] = H[i - 1] + 1
        P[i] = P[i - 1]
        S[i] = S[i - 1]

    elif array[i] == 'P':
        H[i] = H[i - 1]
        P[i] = P[i - 1] + 1
        S[i] = S[i - 1]

    elif array[i] == 'S':
        H[i] = H[i - 1]
        P[i] = P[i - 1]
        S[i] = S[i - 1] + 1

result = 0
for i in range(1, N + 1):
    p_h = P[i - 1] + (H[N] - H[i - 1])
    p_s = P[i - 1] + (S[N] - S[i - 1])
    h_s = H[i - 1] + (S[N] - S[i - 1])
    h_p = H[i - 1] + (P[N] - P[i - 1])
    s_p = S[i - 1] + (P[N] - P[i - 1])
    s_h = S[i - 1] + (H[N] - H[i - 1])

    result = max(result, p_h, p_s, h_s, h_p, s_p, s_h)

print(result)