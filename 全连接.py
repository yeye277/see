#导包
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 读取数据
csv_path = r'D:\大二下\人工智能模型与算法\实训\reviews_with_sentiment_clean.csv'
try:
    data = pd.read_csv(csv_path)
    data['review'] = data['review'].astype(str)
    print("数据读取成功！")
except FileNotFoundError:
    print(f"文件 {csv_path} 不存在，请检查文件路径是否正确。")
    exit()  # 如果文件不存在，则退出程序


# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_review'])

# 输出TF-IDF矩阵的形状
print("TF-IDF矩阵的形状:", tfidf_matrix.shape)

# 将文本标签转换为数值标签，例如，假设情感标签为'positive'和'negative'，分别转换为0和1
# 将目标变量（标签）转换为数值型数据
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
# 假设 'positive' 为正类，'negative' 为负类

# TF-IDF矩阵
X = tfidf_matrix.toarray()
y = data['sentiment']  # 'sentiment'列是情感标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型结构
#定义用于构建简单线性堆叠层的模型
model = Sequential()
#隐藏层神经元数量: 第一层256个，第二层128个
model.add(Dense(256, input_dim=X.shape[1], activation='relu'))  # 输入层维度根据TF-IDF矩阵的列数确定
#激活函数: ReLU（隐藏层）和sigmod（输出层）
model.add(Dense(128, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))  # 输出层使用sigmoid激活函数，因为是二分类问题

# 编译模型
optimizer = Adam(learning_rate=0.0001)  # 设置学习率
#模型在训练时将使用稀疏分类交叉熵作为损失函数，使用指定的优化器来更新模型参数，
# #同时监测精确度作为评估指标。这些配置将帮助模型更好地学习数据并进行准确的预测
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 超参数设置
batch_size = 32 #批量大小（batch size）: 32
epochs = 30
# 训练模型
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
#validation_split=0.2表示20%的训练数据将被用作验证集
#verbose=1表示显示进度条

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# 预测测试集
y_pred = model.predict(X_test)
#利用训练好的神经网络模型对测试数据集进行预测，并将输出的概率结果转换为类别标签
y_pred_classes = np.argmax(y_pred, axis=1)

# 打印分类报告
print(classification_report(y_test, y_pred_classes))

# 保存模型
model.save('sentiment_analysis_model.h5')
#
# 使用文本进行情感预测
sample_text = ["我喜欢这部电影!"]
sample_tfidf = tfidf_vectorizer.transform(sample_text).toarray()
sample_pred = model.predict(sample_tfidf)
sentiment_label = np.argmax(sample_pred[0])
sentiment = "positive" if sentiment_label == 1 else "negative"
print(f"The model predicts the sentiment of the text as: {sentiment}")



