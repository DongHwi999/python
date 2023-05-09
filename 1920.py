import sys

def binary_search(array, target):
    start, end = 0, len(array)-1

    while start <= end:
        mid = (start+end)//2
        if array[mid] == target:
            return 1
        elif target < array[mid]:
            end = mid-1
        else :
            start = mid+1
    return 0

read = sys.stdin.readline

N = int(read())
nums = list(map(int, read().split()))
nums.sort()

M = int(read())
nums2 = list(map(int, read().split()))

for target in nums2:
    print(binary_search(nums, target))