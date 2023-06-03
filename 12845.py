# 카드 개수 입력받기
n = int(input())

# 카드별 레벨 입력받기
level_list = list(map(int, input().split()))
# 카드 레별에 따라 내림차순 정렬
level_list.sort(reverse= True)
# 획득할 골드 초기화
answer = 0
# 가장 큰 레벨 카드 2장 합산
answer += level_list[0] + level_list[1]

# 레벨 큰 순서대로 카드 합산
for i in range(2, n):
    answer +=level_list[0] + level_list[i]

print(answer)