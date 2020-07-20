import numpy as np

def Conv2(img, H, W, kernel, n):
    # img：输入图片；W,H：图片的宽和高；kernel：卷积核。
    # return：和输入图像尺寸大小相同的feature map；
    # 卷积大小固定为3*3卷积，这里因为固定了卷积大小，所以写代码前可以直接确定：卷积步长为1，四周个填充一排0
    col = np.zeros(H)
    print(col)
    raw = np.zeros(W + 2)
    print(raw)
    img = np.insert(img, W, values=col, axis=1)
    img = np.insert(img, 0, values=col, axis=1)
    img = np.insert(img, H, values=raw, axis=0)
    img = np.insert(img, 0, values=raw, axis=0)
    print(img)
    res = np.zeros([H,W])##直接新建一个全零数组，省去了后边逐步填充数组的麻烦
    for i in range(H):
        for j in range(W):
            temp = img[i:i + 3, j:j + 3]
            temp = np.multiply(temp,kernel)
            res[i][j] = temp.sum()

    return (res)

if __name__ == '__main__':

    A = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # 4行5列
    ken = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    print(Conv2(A, 5, 5, ken, 3))
