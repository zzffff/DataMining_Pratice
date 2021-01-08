# 探索数据前：

1.根据数据内容和目标，探索相应领域的知识（各个特征的常识）

2.在数据简介搜索相关领域知识等，获得知识和个人认知下，观察某些数据是否符合常理

  例如：银行贷款数据年龄有18岁以下的人，但是有个样本的年龄大于300，显然不符合常理
      这时候不一定要删掉这行样本，可以直接创建一个新列，里面都是bool值，和is_ null特征一样，
      是一个关于某某特征的is_incorrect特征，特征的值都是布尔值要么是True，要么是False，可能会给模型带来帮助
        
3.了解数据集是怎么获取的
了解获取从总数据库获取这些样本的方式(是class balance的吗？还是随机从总数据库抽取的样本，等等)

**从总数据库获取Train数据集和Test数据集的方式可能不同，所以可能不能直接用Train的随机一部分作为验证集，
  因为这样的验证集不一定能很好地代表测试集合,因此这样的验证集表现也不能一定代表了模型的表现（所以最好还是使用交叉验证）**


# 数据预处理的步骤：

1.连接训练集和测试集，删除只有一种值的列；
  技巧：dataframe.nunique(axis = 1) == 1，可以得到df中所有只含有一种值的列
  用info，value_counts,describe等pandas方法大概看看各个特征，初步感受。

2.label_encode或者factorize类别特征，然后删除重复列
  技巧：df.T.drop_duplicates()，可以删除df中所有重复列
  注：有时候有些重复列需要LabelEncode之后才能看出来(具体见见Coursera_Note的探索数据集中的预清理数据的技巧)
 （注意pandas的fatorize技巧，以及factorize后数据集过大不能直接转置后删除重复行，要使用循环技巧，
  具体见Coursera笔记的Supplementary notebook中week_2的EDA_Springleaf_screencast）

3.通过nunique和hist等方法，判断个特征的类别（数值型还是类别型；数值型的话是int还是float，int的话可能是count值），
     此外：
     3.1 在使用hist等方法可视化观察特征时，可能会遇到某个特征的分布或是有极端值，或是有某几个值特别多，或是有分布肥尾不均匀，
        该特征在训练集合验证集中的分布不相同，等等问题，要关心探究背后的原因。
     3.2 注意一些非nan的nan值，如-999,9999等等
     3.3 将数值型特征列名以及类别型特征列名分别各自保存在一个list中，可以用以下代码区分和快速找到数值型特征名和类别型特征名：
        #技巧：通过列的类型分辨cat和num
        cat_cols = list(train.select_dtypes(include=['object']).columns)
        num_cols = list(train.select_dtypes(exclude=['object']).columns)
     3.4 Hist方法常见的注意点：
        不能通过单一的一副图就确定自己的某种猜想，如果自己产生了某种猜想，一定要做多几个完全不一样的图来证实这种猜想。
        （比如说划分bins的数目不同作图，对数据取log后作图，做别的类型图等等）
     3.5 其余可视化技巧（见Coursera_Note的探索数据集中的可视化探索）

4.探索数值型特征和类别型特征，并找出所有时间型特征，转换为datetime类型

补充探索技巧：

1.探索train set和test set的数据很重要，可以从两个集合的列数，给的特征，或者各个特征（特别是时间序列特征）的总数，不同的个数等等出发
    （见Coursera笔记的Supplementary notebook中week_2的EDA_video2)
    主要的目的是判断train和test两个数据集是否一致，不一致的话要调整为一致，这样在训练数据上训练的模型才有很好的泛化性。
    如果遇到测试集合和训练集的特征分布很不同，且是二分类问题，则可以使用对抗验证方法做特征衰减
    （也可以用对抗验证方法来判断训练集的模型是否在验证集有很好的泛化性，一般来说采取了对抗验证步骤测试集的AUC高于0.7以上，则泛化效果不好）

2.可以对训练数据进行简单拟合一个随机森林模型（在na已经被替代为-999的情况下），使用随机森林的feature_importance_方法，结合画图来初步判断特征重要性（具体见补充notebook第二周的EDA_video3_screencast）。

3.对一些感兴趣的特征可以求其均值和方差观察，如果均值接近0，方差接近1，那么很有可能这个特征被赛方标准化过了，标准化过后的数据不好处理，将标准化后的数据重新变回原来的数据对数据的理解会有一定帮助（具体操作方法见补充notebook第二周的EDA_video3_screencast）

4.检查数据集是否有重复行，甚至是重复行的特征值相同但是目标值却不同，重复行如果太多，则会严重影响模型的训练效果，所以可以选择删掉重复行。
但是最重要的是要搞清楚为什么会出现重复行？重复特征但是结果却不同合理吗？重复行这么多合理吗？删除重复行训练模型测试集合表现更好了吗？等等问题。

# 检查outliner

一共有5种方法：mad方法，分位点法，三个标准差法，多数决法和box_plot的方法


```python
# 常规的MAD方法，下面另外一个MAD方法用的是平方
from scipy.stats import norm
def mad_based_outlier_origin(points, thresh=3.5):
    if type(points) is list:
        points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[:, None]
    med = np.median(points, axis=0)
    abs_dev = np.absolute(points - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    return mod_z_score > thresh

#假定数据服从正态分布，我们让异常点（outliers）落在两侧的 50% 的面积里，让正常值落在中间的 50% 的区域里
#即让MAD在阈值外（0.25和0.75分位点）的数据点认为是outliner
#MAD，即绝对值差中位数法（Median Absolute Deviation），该方法要求数据服从正态分布前提
#返回一系列布尔值，对应该数据点是否为outliner
def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation #0.6745即为标准正态分布的75%分位点

    return modified_z_score > thresh

#分位点在2.5和97.5以外的视为异常值，有点像winsorize
#返回一系列布尔值，对应该数据点是否为outliner
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    (minval, maxval) = np.percentile(data, [diff, 100 - diff])
    return ((data < minval) | (data > maxval))

#超过三个标准差之外的数据认为是异常值（有待商榷，因为这个方法的前提是数据本身是服从正态分布的）
#返回一系列布尔值，对应该数据点是否为outliner
def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if val/std > threshold:
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier

#箱线图方法检验的outliner
def box_plot_outliner(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    isoutliner =[]
    for val in data:
        if((val<(q1-1.5*iqr))|(val>(q3+1.5*iqr))):
            isoutliner.append(True)
        else:
            isoutliner.append(False)
    return isoutliner

#投票法，三个outliner检测方法多数决，只有大于等于两个True才认为是outliner
#没有加入box_plot检验outliner的方法
def outlierVote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    temp = zip(data.index, x, y, z)
    final = []
    for t in temp:
        if t.count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final

def plotOutlier(x):
    fig, axes = plt.subplots(nrows=4)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote,box_plot_outliner]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False) #标出outliner，x轴红色点处

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=20)
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    axes[2].set_title('STD-based Outliers', **kwargs)
    axes[3].set_title('Majority vote based Outliers', **kwargs)
    axes[3].set_title('box_plot based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=20)
    fig = plt.gcf()
    fig.set_size_inches(15,10)
    
def plotOutlierFree(x):
    fig, axes = plt.subplots(nrows=4)
    nOutliers = []
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]):
        tfOutlier = zip(x, func(x))
        nOutliers.append(len([index for (index, bol) in tfOutlier if bol == True]))
        outlierFree = [index for (index, bol) in tfOutlier if bol == True]
        sns.distplot(outlierFree, ax=ax, rug=True, hist=False)
        
    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=15)
    axes[0].set_title('Percentile-based Outliers, removed: {r}'.format(r=nOutliers[0]), **kwargs)
    axes[1].set_title('MAD-based Outliers, removed: {r}'.format(r=nOutliers[1]), **kwargs)
    axes[2].set_title('STD-based Outliers, removed: {r}'.format(r=nOutliers[2]), **kwargs)
    axes[3].set_title('Majority vote based Outliers, removed: {r}'.format(r=nOutliers[3]), **kwargs)
    fig.suptitle('Outlier Removed By Method with n={}'.format(len(x)), size=20)
    fig = plt.gcf()
    fig.set_size_inches(15,10)

#返回四种方法对应检测出outliner的个数和占数据集的比例
def outlierRatio(data):
    functions = [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]
    outlierDict = {}
    for func in functions:
        funcResult = func(data)
        count = 0
        for val in funcResult:
            if val == True:
                count += 1 
        outlierDict[str(func)[10:].split()[0]] = [count, '{:.2f}%'.format((float(count)/len(data))*100)]
    
    return outlierDict

#使用数据的median或者minUpper代替outliner
#minUpper为outliner中的最小值(如果这个minUpper比数据的mean均值还小，需要注意)
def replaceOutlier(data, method = outlierVote, replace='median'):
    '''replace: median (auto)
                'minUpper' which is the upper bound of the outlier detection'''
    vote = outlierVote(data)
    x = pd.DataFrame(zip(data, vote), columns=['debt', 'outlier'])
    if replace == 'median':
        replace = x.debt.median()
    elif replace == 'minUpper':
        replace = min([val for (val, vote) in zip(data, vote) if vote == True])
        if replace < data.mean():
            return 'There are outliers lower than the sample mean'
    debtNew = []
    for i in range(x.shape[0]):
        if x.iloc[i][1] == True:
            debtNew.append(replace)
        else:
            debtNew.append(x.iloc[i][0])
    
    return debtNew
```

# 处理outliner


```python
1.
#删除极端值，用整个特征列的中位数替代该位置的值，在这里默认极端值为98和96
def removeSpecificAndPutMedian(data, first = 98, second = 96):
    New = []
    med = data.median()
    for val in data:
        if ((val == first) | (val == second)):
            New.append(med)
        else:
            New.append(val)            
    return New

2.
观察特征的各个值(Count,value_counts,hist等方法)，将相对十分少量的那些值（如只出现1到3次）视为outliner
然后用非outliner集合的最大值（或者最小值和中位数，或者是outliner集合的下界或者上界，即minUpperbound）去替代它们。
```

# NA值

**NA值的发现**

1.缺失值的发现
  1.1  缺失值可能是有很多形式，在原始数据里不一定是na，可能是-999，-1，很大的数值或者特殊字符串，空字符串等等
  1.2  寻找缺失值的办法有很多，一般是直接观察法。
     另一种则是使用直方图判断法（变成直方图之前探索之前，可以先把某些怀疑是缺失值的字符串转化为特定数值以便做成直方图），
     一些缺失值往往频数会异常突出（处在分布之间而不是最大值或者最小值）
     或者距离数据大致范围（主要的区间）很远（如-1，若大部分数据范围在0~1）

**NA值的处理**

1.常规处理法：
  使用mean，median，most_frequent方法进行替代特征列的NA值
    
2.简单处理缺失值法：
     直接加一个新的和对应特征相关的is_ null列，列的值均为布尔值，如果该特征不是缺失值就是True，如果是缺失值就是False，
    （除非模型能处理NA值(如数模型)，否则通常在建模前还要再把NA转换为数值）

**注意：** 

1.在生成新特征之前，不要做相关的missing_ value填充，否则容易造成不好的结果，给模型学习带来很大的影响和错误的引导

2.缺失值的填充可以使用-999，均值，中位数,众数等等，但是要注意，填充解决了缺失值的问题，对线性模型等模型有好处，但是对于基于树的模型会带来麻烦，因为缺失值的填充会让树找分裂点更加不真实和困难（如果某一列的缺失值填充有很多的话）

3.**(常见现象）**对于在某特征中某些值训练数据中出现而在测试数据中未出现（或者反过来），可以使用“频数encode”方法。
   根据它们在样本中出现的次数，用次数来作为类别的code，这样的话某些类型即使没有在测试集和训练集中同时出现，也有code值。

**除了常规使用mean，median，most_frequent方法进行替代特征列的NA值，现在还有新的思路，建模预测法**

思路：

将某一有不少NA列，作为待预测的因变量，其他的列（除了y列，因为测试数据的y列未给）作为自变量，拟合模型。
该列已知的值作为训练数据的训练因变量，未知值作为测试数据的因变量。
可以拟合多个模型比较效果最后选择最优的模型（优先选择简单的模型，如线性模型）

**NA占比**


```python
#计算data的各列NA值的数目，和NA值占比
def naCount(data):
    naCount = {}
    for col in data.columns:
        colNa = 0 #该列的NA个数
        for val in data[col].isnull(): #是否is_null可以检验出所有的缺失值，有待商榷
            if val == True:
                colNa += 1
        naCount[col] = [colNa, '{:0.2f}%'.format((float(colNa)/len(data))*100)]

    return naCount
```

# 处理匿名特征


```python
1.通过各种方式，找出特征原来的样子:
        （研究之前：使用pandas的factorize方法快速对类别型数据进行LabelEncode，
        然后对数据集合进行随机森林建模，输出特征的重要性折线图，对匿名特征们都有重要认识。）
        例如：有的特征既没有名字，数字都是小数，通过各种函数调查这行数据，发现这行数据的均值接近0，
        标准差接近1，则这组数据很有可能是被标准化了，那么就想办法看看能不能使得标准化数据回到原样。
        
2.通过各种函数调查特征的类型（数字型，序列型，二元型，或者只能判断为类别型）:
        常使用的方法有pandas的.dtype(),.info(),.value_counts(),.isnull()
        其中.dtype()方法常常返回三种类型，float、int、object，其中object最复杂，可能是各种数据类型，
        需要结合.info(),.value_counts(),.isnull()细看该列特征的内容判断。
```

# 将日期特征转化为datetime格式

好处： 在研究数据的时候，处理和分析样本的时间特征变得更加容易
     使用例子：某data的A特征为datetime特征（年月日），想要获得该data的月特征，只需要：data["A"].dt.month即可

方法：
   1.读取data后再将特征转换为datatime：
     1.1使用pandas的pd.to_datetime（特征, format = "%d.%m.%Y"）可以将特征转换为datetime格式
     1.2如果是包含了时分秒的特征，则format的参数为"%d%b%y:%H:%M:%S"
       (具体见Coursera笔记的Supplementary notebook中week_2的EDA_Springleaf_screencast)
     1.3如果重新读取csv，则需要手动将所有时间型的特征转换为datetime
     
   2.提前知道哪些是datetime特征，读取data的时候使用parse_date将特征转换为datetime
     2.1 使用pd.read_csv读取csv的时候，可以用parse_date参数指定有哪些特征是datetime形式，输出的结果会将它们作为datetime
     2.2 由于使用了parse_date参数，read_csv的读取时间会相对大大增加，因此可以在read_csv中使用参数infer_datetime_format = True减少解析时间
     2.3 read_csv还有一个keep_date_col参数，等于True则保留parse_date参数指定的那些datetime特征，False则不保留


```python

```

# 使用K-means聚类方法删除异常样本


```python

```
