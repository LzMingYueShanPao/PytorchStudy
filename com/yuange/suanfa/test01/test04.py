def containsNearbyDuplicate(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    hash_map = {}
    for i in range(len(nums)):
        if (nums[i] in hash_map) and (abs(hash_map[nums[i]] - i) <= k):
            return True
        else:
            hash_map[nums[i]] = i
    return False

nums = [1,2,3,1]
k = 3
print(containsNearbyDuplicate(nums, k))