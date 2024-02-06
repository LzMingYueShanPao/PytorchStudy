import torch
from matplotlib import pyplot as plt

def plot_curve(data):
    ## plt.figure()是Matplotlib库中的一个函数，用于创建一个新的图形窗口或画布。它可以创建一个空白的图形窗口，然后我们可以在这个窗口上添加图表、子图、图形元素等。
    fig = plt.figure()
    """ 
    plt.plot 是 Matplotlib 库中的一个函数，用于绘制折线图。
    plt.plot 的语法如下：
        plt.plot(x, y, format_string, **kwargs)
    参数说明：
        x：x 轴的数据，可以是一个数组或列表。
        y：y 轴的数据，可以是一个数组或列表。
        format_string：可选参数，用于指定折线图的样式，如颜色、线型和标记等。格式字符串由颜色字符、线型字符和标记字符组成，顺序不限，例如 "ro-" 表示红色圆形标记的实线。
        **kwargs：可选参数，用于设置其他属性，如标签、标题、坐标轴范围等。
    """
    plt.plot(range(len(data)), data, color='blue')
    """ 
    plt.legend 是 Matplotlib 库中的一个函数，用于添加图例到图形中。图例可以用于标识不同线条或数据集的含义，使得图形更易于理解。
    plt.legend 的语法如下：
        plt.legend(*args, **kwargs)
    参数说明：
        *args：可变参数，用于传入图例的标签。可以是字符串、列表或元组等。
        **kwargs：可选参数，用于设置图例的其他属性，如位置、边框、阴影等。
    """
    plt.legend(['value'], loc='upper right')
    """ 
    plt.xlabel 是 Matplotlib 库中的一个函数，用于设置 x 轴的标签。
    plt.xlabel 的语法如下：
        plt.xlabel(xlabel, **kwargs)
    参数说明：
        xlabel：x 轴的标签，可以是一个字符串。
        **kwargs：可选参数，用于设置其他属性，如字体大小、字体样式等。
    """
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(img, lable, name):
    fig = plt.figure()
    for i in range(6):
        """ 
        plt.subplot 是 Matplotlib 库中的一个函数，用于创建子图。子图可以将一个大的图形分割成多个小的图形，每个子图可以有自己的坐标轴和数据。
        plt.subplot 的语法如下：
            plt.subplot(num_rows, num_cols, plot_num)
        参数说明：
            num_rows：子图的行数。
            num_cols：子图的列数。
            plot_num：当前子图的索引，从左上角开始。
        """
        plt.subplot(2, 3, i+1)
        """
        plt.tight_layout()是Matplotlib库中的一个函数，用于自动调整子图或图形中的布局，以使其适应图形窗口或画布。它可以帮助我们消除子图之间的重叠、调整子图的大小和位置，使得绘图更加美观和可读。
        在使用Matplotlib进行绘图时，有时子图之间可能会出现重叠的情况，导致文字或图形相互遮挡，不易阅读。这时可以使用plt.tight_layout()函数来自动调整子图的布局，以解决这个问题。
        """
        plt.tight_layout()
        """ 
        plt.imshow()是Matplotlib库中的一个函数，用于将二维数组或图像显示为彩色或灰度图像。它可以将图像数据显示到Matplotlib窗口或画布中，并提供了很多参数用于控制图像的外观和呈现方式。
        在使用Matplotlib进行图像处理和分析时，我们经常需要将图像数据显示到图形窗口或画布中，以便进行可视化和交互式操作。plt.imshow()函数可以帮助我们实现这个目标，它可以将二维数组或图像显示为彩色或灰度图像，并自动调整图像的大小和比例。
        使用plt.imshow()函数非常简单，只需要传入一个二维数组或图像数据即可
        """
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        """
        plt.title()是Matplotlib库中的一个函数，用于为图形或子图添加标题。它可以在Matplotlib窗口或画布中显示一个文本标题，并提供了很多参数用于控制标题的位置、字体、大小、颜色等属性。
        在使用Matplotlib进行可视化时，我们经常需要为图形或子图添加标题，以便说明图形的内容和目的。plt.title()函数可以帮助我们实现这个目标，它可以在图形窗口或画布中添加一个文本标题，并调整其位置和样式。
        使用plt.title()函数非常简单，只需要传入一个字符串作为标题即可
        """
        plt.title("{}: {}".format(name, lable[i].item()))
        """
        plt.xticks()是Matplotlib库中的一个函数，用于设置和获取X轴刻度的位置和标签。它可以帮助我们自定义X轴的刻度以及对应的标签，以适应特定的需求。
        在使用Matplotlib进行数据可视化时，我们经常需要调整X轴的刻度和标签，以便更好地展示数据。plt.xticks()函数可以帮助我们实现这个目标，它提供了多种方式来设置和获取X轴的刻度位置和标签。
        使用plt.xticks()函数的一种常见用法是通过指定位置和标签来自定义X轴的刻度
        """
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(lable, depth=10):
    out = torch.zeros(lable.size(0), depth)
    idx = torch.LongTensor(lable).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out