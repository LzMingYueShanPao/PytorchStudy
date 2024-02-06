import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    """
    Python内置函数range()是用来生成一个整数序列的函数。它常用于循环中，可以方便地遍历一定范围内的整数。
    
    range()函数有三种使用方式：
    
    range(stop)：生成一个0到stop-1之间的整数序列。
    range(start, stop)：生成一个start到stop-1之间的整数序列。
    range(start, stop, step)：生成一个start到stop-1之间以步长为step的整数序列。
    """
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 设置梯度
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
        new_b = b_current - (learningRate * b_gradient)
        new_w = w_current - (learningRate * w_gradient)
        return [new_b, new_w]

# 运行梯度下降
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        """
        np.array()是NumPy库中的一个函数，用于创建多维数组（也称为ndarray对象）
        """
        b, w  = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def run():
    """
    从CSV文件中读取数据，并生成一个数组
    """
    points = np.genfromtxt("data.csv", delimiter=",")
    print("数据=",points)
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    # 迭代次数
    num_itetations = 1000
    print("开始梯度下降 b={0}, m={1}, error={2}"
          .format(initial_b,initial_w,compute_error_for_line_given_points(initial_b,initial_w,points)))
    print("运行...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_itetations)
    print("运行完成后 {0} b={1}, m={2}, error={3}"
          .format(num_itetations,b,w,compute_error_for_line_given_points(b,w,points)))

if __name__ == '__main__':
    run()