#导包
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

#通常用于HTTP请求的头部信息
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'cookie': 'your_cookie_here',
    'Connection': 'keep-alive',
}

#获取电影ID
def get_homepage_movies():
    movies = set()    # 初始化一个空集合，用于存储电影编号
    with open("D:\大二下\人工智能模型与算法\实训\情感分析\home.html", 'r', encoding='utf-8') as f:
        html = f.read()
        # 使用BeautifulSoup解析html内容，并指定解析器为"html.parser"
        soup = BeautifulSoup(html, "html.parser")
        # 查找html中所有的<a>标签（通常代表链接）
        links = soup.find_all("a")
        for link in links:
            movie_link = link["href"]  # 获取<a>标签的href属性，即链接地址
            # 使用正则表达式查找链接地址中的数字
            numbers = re.findall(r'\d+', movie_link)
            movies.add(numbers[0])   # 如果找到了数字，将第一个数字添加到集合movies中
        with open('movies1.txt', 'w') as f:
            for item in movies:
                f.write(item + '\n')  # 将电影编号写入文件，并在每个编号后添加一个换行符
    # 返回集合movies，其中包含所有找到的电影编号
    return movies

#获取评论文本和评分
def get_movie_reviews(movie_id, page):
    # 构造豆瓣电影评论页面的URL，其中start参数用于分页，limit参数表示每页显示的评论数量
    # status=P表示只获取普通用户的评论，sort=new_score表示按评分的最新时间排序
    url = f"https://movie.douban.com/subject/{movie_id}/comments?start={page * 20}&limit=20&status=P&sort=new_score"
    # 使用requests库发送GET请求到指定的URL，并传入headers头部信息
    response = requests.get(url, headers=headers)
    # 检查HTTP响应的状态码是否为200，表示请求成功
    # 如果不是200，则打印出出错的电影ID，并返回一个空列表
    if response.status_code != 200:
        print(f"获取评论的电影ID为：{movie_id}")
        return []
    # 使用BeautifulSoup库解析响应的HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 查找所有class为'short'的<span>标签，这些标签通常包含评论的简短内容
    reviews = soup.find_all('span', class_='short')
    # 查找所有class以'allstar'开头的<span>标签，这些标签通常包含评分的星星信息
    # 使用lambda函数作为class_参数的值，以匹配以'allstar'开头的class
    ratings = soup.find_all('span', class_=lambda x: x and x.startswith('allstar'))

    reviews_data = []   # 初始化一个空列表，用于存储评论数据和评分
    # 使用zip函数将reviews和ratings两个列表组合起来，然后遍历它们
    for review, rating in zip(reviews, ratings):
        text = review.get_text(strip=True)  # 获取评论的文本内容，并去除两端的空白字符
        # 从rating的class属性中获取评分信息
        # 这里假设class属性的第一个值是以'allstar'开头，并且评分数值位于字符串的倒数第二个位置
        # 例如：'allstar40'中的'4'是评分
        score = int(rating['class'][0][-2])
        # 将评论文本和评分组成一个字典，并添加到reviews_data列表中
        reviews_data.append({'review': text, 'rating': score})

    return reviews_data


if __name__ == '__main__':
    # 调用 get_homepage_movies 函数，获取主页上的电影ID，并存储在 movies 集合中
    movies = get_homepage_movies()
    # 打印找到的电影ID的数量
    print(f"找到{len(movies)}个电影ID。")
    all_data = []  # 初始化一个空列表，用于存储所有电影的评论数据
    for movie_id in movies:  # 遍历 movies 集合中的每一个电影ID
        page = 0  # 初始化页码为0，用于分页请求电影评论
        # 当收集到的评论数据少于 11111 条时，继续循环
        while len(all_data) < 11111:
            # 调用 get_movie_reviews 函数，获取指定电影ID和页码的评论数据
            data = get_movie_reviews(movie_id, page)
            # 如果 get_movie_reviews 返回空列表（表示没有更多评论或请求失败），则跳出当前电影的循环
            if not data:
                break
            all_data.extend(data)   # 将获取到的评论数据添加到 all_data 列表中
            page += 1    # 页码加1，以便获取下一页的评论数据
            # 如果 all_data 列表中的数据数量已经达到或超过 11111 条，则跳出循环
            if len(all_data) >= 11111:
                break

    # 使用 pandas 库，将 all_data 列表转换为 DataFrame
    df = pd.DataFrame(all_data)
    # 将 DataFrame 写入到 CSV 文件中，不包含索引，并使用 'utf-8-sig' 编码
    df.to_csv('reviews.csv', index=False, encoding='utf-8-sig')