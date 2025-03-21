#第一部分：导入库
import pandas as pd          # 数据处理库，用于读取csv文件
import numpy as np           # 科学计算库，处理多维数组
from sklearn.model_selection import train_test_split  # 数据集划分工具
from sklearn.metrics import accuracy_score  # 准确率计算
from collections import Counter  # 统计工具

#第二部分：决策树节点类
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature    # 用于分裂的特征索引（如"花瓣长度"）
        self.threshold = threshold  # 分裂阈值（如5.0厘米）
        self.left = left          # 左子树（满足<=阈值的样本）
        self.right = right        # 右子树（不满足阈值的样本）
        self.value = value        # 叶节点的预测值（仅叶节点有值）

#第三部分：决策树类
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth# 树的最大深度（防止过拟合）
        self.min_samples_split = min_samples_split# 节点最小分裂样本数
        self.root = None#根节点
        self.feature_importances_ = None  # 存储特征重要性

    def _gini(self, y):
        # 计算Gini不纯度（值越小表示越纯净）
        counts = np.bincount(y) # 统计每个类别的数量
        probabilities = counts / len(y)# 概率
        return 1 - np.sum(probabilities**2) # GINI系数公式

    def _best_split(self, X, y):
        best_gini = float('-inf') # 初始化最佳基尼值为负无穷大
        #在可视化过程中，AI出现了致命的错误，导致准确率直线降低集体原因如下
        #在一开始的版本中，使用了inf来初始化best_gini，对应的判断条件是if gini < best_gini:
        #然而在AI给我可视化版本中，沿用了无穷大的初始化条件，但由于需要计算Gini系数，即用分裂之后的系数与之前的父节点系数作差，
        #导致Line63中的判断条件弄反了
        best_feature, best_threshold = None, None
        parent_gini = self._gini(y)  # 计算父节点的基尼不纯度
         # 遍历所有特征
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            #X[:, feature]：从数据集中提取所有样本的第 feature 个特征的值
            #二维数组的列切片操作
            #<= threshold：将这些特征值与阈值比较，返回一个布尔数组（True/False）。
            #left_indices：最终得到一个布尔数组，
            #True 表示对应样本的特征值满足条件（属于左子树），False 表示不满足（属于右子树）。

            

            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                # 创建布尔掩码划分左右子树
                # #Boolean Masking
                #布尔数组可以作为索引，用于筛选数据。
                #True 对应的位置会被保留，False 会被过滤
                if np.sum(left_indices) < self.min_samples_split or len(y) - np.sum(left_indices) < self.min_samples_split:
                    # 如果任一侧样本数不足则跳过
                    continue
                # 计算子节点的加权基尼不纯度
                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[~left_indices])
                weighted_gini = (np.sum(left_indices)*gini_left + np.sum(~left_indices)*gini_right) / len(y)
                #~表示取反符号 在这里left取反自动变为right
                # 计算不纯度减少量
                gini_reduction = parent_gini - weighted_gini
                if gini_reduction > 0 and gini_reduction > best_gini:  # 修正：寻找最大不纯度减少
                    #但条件 gini_reduction < best_gini 导致程序误选最小的减少量，而非最大
                    best_gini = gini_reduction
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gini  # 返回不纯度减少量

    def _build_tree(self, X, y, depth=0):
        # 终止条件：达到最大深度或样本不足
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            counter = Counter(y) # 统计类别分布
            #counter用于统计 y 当中不同类型的数目，以元组形式输出
            #执行后，counter 的内容为：Counter({0: 2, 1: 4})
            return Node(value=counter.most_common(1)[0][0]) # 返回出现最多的类别
            #most_common(n) 方法返回出现次数最多的前 n 个元素及其计数
            #n=1 时返回一个列表，包含一个元组：[(1, 4)]（元素1出现4次）
            #[0] 获取列表中的第一个元素（即元组 (1, 4)）
            #再取 [0] 获取元组的第一个值（即类别标签 1）
            #表示该叶节点的预测值为类别1
        
        feature, threshold, gini_reduction = self._best_split(X, y)  # 获取不纯度减少量
        if feature is None:# 如果没有找到有效分裂（所有特征都一样）
            return Node(value=Counter(y).most_common(1)[0][0])
        
        # 记录特征重要性
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(X.shape[1])
        self.feature_importances_[feature] += gini_reduction  # 累加不纯度减少量
        #构建递归子树
        left_indices = X[:, feature] <= threshold
        left = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right = self._build_tree(X[~left_indices], y[~left_indices], depth+1)
        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)  # 从根节点开始构建树

    def _predict(self, x, node):
        if node.value is not None:  # 到达叶节点
            return node.value
        # 根据特征值选择分支
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, X):
        return [self._predict(x, self.root) for x in X]  # 对每个样本进行预测

   # 第四部分：随机森林类 
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees                      # 树的数量
        self.max_depth = max_depth                  # 每棵树的深度        
        self.min_samples_split = min_samples_split
        self.n_features = n_features                # 每棵树使用的特征数（None表示全部）
        self.trees = []                             # 存储所有树的列表
        self.feature_importances_ = None  # 新增：存储特征重要性

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.feature_importances_ = np.zeros(X.shape[1])  # 初始化
        
        for _ in range(self.n_trees):

             # 1. 自助采样（有放回抽样）
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            # 2. 随机选择特征（增强多样性）
            features = np.random.choice(X.shape[1], n_features, replace=False)
            #X.shape表示取出维度，及有多少feature, n_feat为实际上抽取的样本个数
            #replace=False TF表示是否允许重复 TRUE表示抽出的n_feat中可以存在重复的元素，False则与之相反
            
            # 3. 构建并训练决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X[indices][:, features], y[indices])
            self.trees.append((features, tree))# 保存特征索引和树
            # 4. 聚合特征重要性（仅考虑该树使用的特征）
            global_importances = np.zeros(X.shape[1])
            global_importances[features] = tree.feature_importances_
            self.feature_importances_ += global_importances# 聚合所有树的特征重要性
        
        # 归一化
        self.feature_importances_ /= self.n_trees# 取平均
        self.feature_importances_ /= np.sum(self.feature_importances_)  # 归一化总和为1

    def predict(self, X):
        # 所有树的预测结果（形状：n_trees × n_samples）
        predictions = np.array([tree.predict(X[:, features]) for features, tree in self.trees])
        # 对每个样本进行多数投票
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions.T])
        #[.T]表示转置，转置后获得众数结果，
    
            
            

import matplotlib.pyplot as plt
#图形化展示部分
def plot_feature_importance(importances, feature_names):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]#np.argsort()[::-1] 实现重要性从高到低排序
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")# 绘制条形图
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])# 特征名标签
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

#可视化的核心思想
#在决策树的每个节点分裂时，计算特征分裂带来的基尼不纯度减少量（Gini Reduction）
#将每个特征在所有树中的所有分裂节点上的基尼不纯度减少量累加，最终归一化得到重要性
#此处使用了竖向的条形图，而横向条形图更适合长特征名的展示，（附上代码plt.barh(range(len(importances)), importances[indices])）


# 加载数据时获取特征名称
def load_iris():
    file_path = "C:/VScodePro/.venv/bezdekIris.data.txt" 
    # 读取数据并添加列名 
    data = pd.read_csv(file_path, header=None)
    data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    # 将类别转换为数字
    label_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    data['class'] = data['class'].map(label_map)
    # 转换为numpy数组
    X = data.iloc[:, :-1].values# 特征矩阵
    y = data.iloc[:, -1].values# 标签向量
    feature_names = data.columns[:-1].tolist()  # 获取特征名称
    # 划分训练集和测试集（70%训练，30%测试）
    return train_test_split(X, y, test_size=0.3, random_state=33), feature_names

if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), feature_names = load_iris()  # 修改返回特征名称
    
    rf = RandomForest(n_trees=100,# 100棵树
                      max_depth=5,# 每棵树最大深度5
                      min_samples_split=2,# 节点最少2个样本才分裂
                      n_features=2)# 每棵树随机选2个特征
    #通过调整以上参数来遏制过拟合100%准确率
    # 训练模型
    rf.fit(X_train, y_train)
    
    # 可视化
    plot_feature_importance(rf.feature_importances_, feature_names)
    y_pred = rf.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")


