import requests
from bs4 import BeautifulSoup
import re

# 请求网页内容
url = 'https://gh.ekyu.moe/mjai-reviewer-demo.html#kyoku-7-1'
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有符合要求的 <section> 标签
    sections = soup.find_all('section', style=re.compile(r'z-index:\d+'))

    # 遍历找到的 <section> 标签并保存内容
    for idx, section in enumerate(sections, start=1):
        content = section.get_text(strip=True)  # 获取去除空格的文本内容
        if content:
            # 保存为不同的 txt 文件
            with open(f'section_{idx}.txt', 'w', encoding='utf-8') as f:
                f.write(content)

    print(f'已成功爬取 {len(sections)} 个 <section> 内容并保存为 txt 文件。')
else:
    print(f'请求失败，状态码: {response.status_code}')

