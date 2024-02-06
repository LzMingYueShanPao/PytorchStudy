class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        N = len(nums)
        self.preSum = [0] * (N + 1)
        for i in range(N):
            self.preSum[i + 1] = self.preSum[i] + nums[i]

    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        return self.preSum[left + 1] - self.preSum[right]

# Your NumArray object will be instantiated and called as such:
nums = [-2, 0, 3, -5, 2, -1]
left = 0
right = 2
obj = NumArray(nums)
param_1 = obj.sumRange(left,right)