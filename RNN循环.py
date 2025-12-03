import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
csv_path = r'D:\大二下\人工智能模型与算法\实训\reviews_with_sentiment_clean.csv'
data = pd.read_csv(csv_path)
data['cleaned_review'] = data['cleaned_review'].astype(str)

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_review'])

# 将文本标签转换为数值标签
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# 准备数据
X = tfidf_matrix.toarray()
y = data['sentiment']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将Numpy数组转换为PyTorch张量
#为了与PyTorch深度学习框架的兼容性和高效性
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义一个简单的神经网络模型SimpleNN，包括三个全连接层和一个Sigmoid激活函数。
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        #输入input_dim表示输入特征的维度
        #64和16表示每个隐藏层的神经元数量，1表示输出层的神经元数量
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    #定义前向传播函数 forward
    #前向传播是指输入数据通过网络的各层进行计算和传递，最终得到输出结果的过程
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# 初始化模型
#这里的输出特征为1表示的是每个特征可以对应一个词
input_dim = X_train_tensor.size(1)  # 获取输入特征的维度
model = SimpleNN(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类问题使用二元交叉熵损失
# 设置学习率
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
#定义了训练函数train，在每个epoch中对模型进行训练，并输出训练损失
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        #这里的inputs代表一个特征，labels代表分类标签
        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels.float().unsqueeze(1)  # 将标签转换为与输出相同的形状
            # 清除之前的梯度（如果有的话）
            optimizer.zero_grad()
            ## 将输入数据通过模型进行前向传播
            outputs = model(inputs)
            # 计算损失，使用定义的损失函数和模型输出及真实标签
            loss = criterion(outputs, labels)
            ## 反向传播损失，计算梯度
            loss.backward()
            # # 使用优化器更新模型的权重
            optimizer.step()
            ## 累计当前批次的损失
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    # 开始训练
train(model, train_loader, criterion, optimizer, num_epochs=5)
torch.save(model.state_dict(), 'RNN_model.h5')

#定义了评估函数evaluate，对模型在测试集上的表现进行评估，并计算其准确率
def evaluate(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度，节省计算资源
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # 由于我们使用了sigmoid作为最后一层的激活函数，所以我们需要对输出应用阈值来获得预测标签
            # 这里我们使用0.5作为阈值，即如果输出大于0.5，我们预测为正类（1），否则为负类（0）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算准确率
    accuracy = 100 * correct / total
    return accuracy

# 使用评估函数
accuracy = evaluate(model, test_loader)
print(f'Accuracy on test set: {accuracy}%')

# 添加一段文字来进行预测
sample_text = ["我喜欢这部电影"]
sample_tfidf = tfidf_vectorizer.transform(sample_text).toarray()
sample_input = torch.Tensor(sample_tfidf)
model.eval()
with torch.no_grad():
    output = model(sample_input)
    predicted_label = 1 if output.item() > 0.5 else 0

sentiment = "positive" if predicted_label == 1 else "negative"
print(f"The model predicts that the sentiment of the text is {sentiment}.")

