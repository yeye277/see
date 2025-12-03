import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
csv_path = 'reviews.csv'  # 如果不在当前目录下，请修改为绝对路径
try:
    data = pd.read_csv(csv_path)
    data['review'] = data['review'].astype(str)
    print("数据读取成功！")
except FileNotFoundError:
    print(f"文件 {csv_path} 不存在，请检查文件路径是否正确。")
    exit()  # 如果文件不存在，则退出程序

# 加载停用词表
with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()
print(len(stopwords))

# 文本预处理函数
#将输入的文本进行分词，并去除停用词，然后返回处理后的文本。
def preprocess_text(text):
    words = jieba.cut(text)
    words = [word for word in words if word.strip() and word not in stopwords]
    return ' '.join(words)

# 对评论文本进行预处理
data['cleaned_review'] = data['review'].apply(preprocess_text)

# 设置阈值，例如3分或4分
threshold = 4

# 根据评分列创建情感标签
# 如果分数高于或等于阈值，则为'positive'，否则为'negative'
data['sentiment'] = np.where(data['rating'] >= threshold, 'positive', 'negative')

# 计算每个类别的数量
pos_count = data['sentiment'].value_counts()['positive']
neg_count = data['sentiment'].value_counts()['negative']
print('正面数量为：',pos_count)
print('负面数量为：',neg_count)

# 如果positive的数量多于negative，随机下采样positive
if pos_count > neg_count:
    # 选择需要剔除的positive数量
    to_remove = pos_count - neg_count
    # 随机选择需要剔除的索引
    remove_indices = data[data['sentiment'] == 'positive'].sample(n=to_remove).index
    # 从DataFrame中剔除这些索引对应的行
    data = data.drop(remove_indices)

# 现在再次查看结果
print(data[['rating', 'sentiment']])
print(data['sentiment'].value_counts())

# 保存数据（如果需要）
data.to_csv('reviews_with_sentiment2.csv', index=False, encoding='utf-8-sig')

#读取处理后的数据
data_clean = pd.read_csv(r'D:\大二下\人工智能模型与算法\实训\reviews_with_sentiment2.csv')

#空值处理
print('处理后数据的空值为：',data_clean['cleaned_review'].isnull().sum())
data_clean.dropna(subset=['cleaned_review'], inplace=True)
# 保存数据（如果需要）
data_clean.to_csv('reviews_with_sentiment_clean.csv', index=False, encoding='utf-8-sig')
print('删除后数据的空值为：',data_clean['cleaned_review'].isnull().sum())

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
tfidf_matrix = tfidf_vectorizer.fit_transform(data_clean['cleaned_review'])

# 输出TF-IDF矩阵的形状
print("TF-IDF矩阵的形状:", tfidf_matrix.shape)

# 如果需要PCA降维
n_components = 2
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# KMeans聚类
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(pca_result)

# 将聚类结果添加到DataFrame中
data_clean['cluster'] = clusters
data_clean['cluster'].value_counts()

# 可视化PCA结果
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
# 添加聚类中心的标记
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of TF-IDF Vectors with KMeans Clusters')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Cluster')
plt.show()