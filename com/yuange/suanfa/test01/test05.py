def missingNumber(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    hash_map = {}
    for i in range(len(nums) + 1):
        hash_map[i] = 0
    for num in nums:
        if num in hash_map:
            hash_map[num] = 1
    for key, value in hash_map.items():
        if value == 0:
            return key


nums = [3,0,1]
print(missingNumber(nums))