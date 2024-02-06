class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None

        mid = len(nums) // 2
        node = TreeNode(nums[mid])

        left = nums[:mid]
        right = nums[mid+1:]

        node.left = self.sortedArrayToBST(left)
        node.right = self.sortedArrayToBST(right)
        return node

def containsDuplicate(nums):
    result = -2
    for num in nums:
        result ^= num
    return result != 0


def hasDuplicate(nums):
    result = 0
    for num in nums:
        # 将当前元素转换为二进制形式
        binary_representation = bin(num)[2:]
        print(binary_representation)
        # 计算当前元素的长度
        length = len(binary_representation)
        print(length)
        # 根据长度确定需要移动多少位
        shift = (length - 1) * 8 + 7
        print(shift)
        # 将当前元素左移shift位，并与result进行按位或操作
        result |= int(binary_representation[::-1], 2) << shift
        print('----------------------')
    return bool(result & (result - 1))




if __name__ == '__main__':
    # 测试样例
    print(hasDuplicate([1, 2, 3]))  # False
    print(hasDuplicate([1, 2, 3, 4, 5]))  # False
    print(hasDuplicate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))  # True
    # s = Solution()
    # nums = [-10,-3,0,5,9]
    # result = containsDuplicate(nums)
    # print(result)
    #
    # result = s.sortedArrayToBST(nums)
