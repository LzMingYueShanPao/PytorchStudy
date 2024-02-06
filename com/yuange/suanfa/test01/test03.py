def containsDuplicate(nums):
    bitmask = 0
    for num in nums:
        # 如果num元素的二进制数的最高位已经在位图中，即值为1，则直接返回True
        if (bitmask & (1 << num)) != 0:
            return True
        # 将位图相应的位置设置为1，以填充位图
        bitmask |= (1 << num)
    return False

nums = [1,5,-2,-4,0]
print(containsDuplicate(nums))