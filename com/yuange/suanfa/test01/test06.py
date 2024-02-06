def moveZeroes(nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    zeroCount = 0
    count = 0
    for i in range(len(nums)):
        if nums[i] == 0:
            zeroCount += 1
        else:
            nums[count] = nums[i]
            count += 1
        i += 1
    for j in range(-zeroCount,0):
        nums[j] = 0

nums = [0,1,0,3,12]
moveZeroes(nums)