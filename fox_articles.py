import concurrent.futures
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import openai
import pandas as pd
from bs4 import BeautifulSoup
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

#全局变量声明
"""文件参数"""
path = r'E:\AI Lab\数据封禁线索\线索归档\Fox_Articles'
"""时间参数"""
days = 1
"""大模型参数"""
zhipuai_api_key = "2e261679169b97bdd7d3b35da0d2d3cb.V64QVNdhlKMcR6Er"
model = "glm-4-flash"
base = "https://open.bigmodel.cn/api/paas/v4/"
llm= ChatOpenAI(
    api_key=zhipuai_api_key,
    model=model,
    base_url=base
)
max_workers =20 #线程数


#根据所需新闻天数指定日期范围
def get_earliest_date(int):
    earliest_date = datetime.today().date()-timedelta(days=int)
    return earliest_date

#创建一个文件夹存储当日新闻数据
def create_folder():
    """用日期命名文件夹"""
    date = datetime.today()
    datestr = date.strftime(r'%Y%m%d')
    folder_name = f"FOX_{datestr}"
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.chdir(folder_path)
    current_dir = os.getcwd()  # 获取当前目录
    print(f"当前目录已切换至：{current_dir}")
    return folder_path

#一个html文本清洗函数
def html_to_clean_text(html):
    """解析HTML"""
    soup = BeautifulSoup(html, 'html.parser')

    """处理换行和列表项（示例含<br>但不需要特殊处理）"""
    for br in soup.find_all('br'):
        br.replace_with('\n')  # 将<br>转换为换行

    """清除其他新闻链接"""
    for strong_tag in soup.find_all('strong'):
        strong_tag.decompose()

    """提取文本并处理空格"""
    text = soup.get_text(separator=' ', strip=True)

    # 合并连续空格/换行，处理特殊字符
    text = re.sub(r'\s+', ' ', text)  # 合并连续空格
    text = re.sub(r'(\s*\n\s*)+', '\n', text)  # 清理段落间换行
    text = text.replace('\xa0', ' ')  # 替换&nbsp;

    return text

#获取格式化日期
def get_formated_date(str):
    """fox新闻的时间戳格式 2025-04-02T17:16:08.000Z"""
    dt = datetime.fromisoformat(str.replace('Z', '+00:00'))  # 替换时区标识
    date_obj = dt.date()  # 提取 date 部分
    return date_obj

#调用playwright模拟浏览器访问，从而获取数据
def get_page_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # page.screenshot(path=f'screenshot-{browser_type.name}.png')
            content = page.locator('pre').text_content()
            content = json.loads(content)
            page.close()
        return content
    except Exception as e:
        print(f"访问网页失败：{e}")

#按日期获取api v3中的所有fox article
def get_news_list(earliest_date):
    try:
        """获取api提供的新闻数据"""
        all_news_articles = []
        url = "https://api.foxnews.com/v3/articles?from=0&size=100"
        response = get_page_content(url)
        page_news = response['data']
        page_earliest_date = page_news[-1]['attributes']['last_published_date']
        page_earliest_date = get_formated_date(page_earliest_date)
        print(f"采集至{page_earliest_date}/目标{earliest_date}")

        while page_earliest_date >= earliest_date:
            all_news_articles.extend(page_news)
            url = response['links']['next']
            print(f"已访问:{response['links']['self']}")
            print(f"正在访问{url},已采集新闻条数：{len(all_news_articles)}")
            time.sleep(10)
            response  = get_page_content(url)
            page_news = response['data']
            page_earliest_date = page_news[-1]['attributes']['last_published_date']
            page_earliest_date = get_formated_date(page_earliest_date)
            print(f"采集至{page_earliest_date}/目标{earliest_date}")

        for news in page_news:
            news_published_date = news['attributes']['last_published_date']
            news_published_date = get_formated_date(news_published_date)
            if news_published_date >= earliest_date:
                all_news_articles.append(news)
        return all_news_articles
    except Exception as e:
        print(f"访问错误：{e}")

#解析新闻数据，提取所需字段
def get_articles(list):
    articles_list = []
    for article in list:
        title = article['attributes']['title']
        if "dek" in article['attributes'].keys():
            dek = article['attributes']['dek']
        else:
            dek = ''
        link = article['attributes']['canonical_url']
        if "taxonomy" in article['attributes'].keys():
            tags = article['attributes']['taxonomy']
        else:
            tags = ''
        date = article['attributes']['last_published_date']
        date = get_formated_date(date)
        content = []
        for component in article['attributes']['components']:
            if component['content_type'] == 'text':
                raw_html_text = component['content']['text']
                text = html_to_clean_text(raw_html_text)
                content.append(text)
            if component['content_type'] == 'image':
                caption = component['content']['caption']
                caption = f"\nimage:{caption}\n"
                content.append(caption)
            if component['content_type'] == 'image_gallery':
                captions = []
                images = component['content']['images']
                for image in images:
                    caption = image['caption']
                    caption = f"\nimage:{caption}\n"
                    captions.append(caption)
                captions = "".join(captions)
                content.append(captions)
        content = "".join(content)
        article_info = {"title":title,"dek":dek,"tags":tags, "link":link, "data":date, "content":content}
        articles_list.append(article_info)
        """保存新闻内容"""
    try:
        articles_df = pd.DataFrame(articles_list)
        articles_df.to_csv("articles.csv", index=False)
        print(f"文章数据保存为articles.csv")
    except Exception as e:
        print(f"保存文章数据时出错：{e}")
    return articles_list

#运行文章采集
def scoping_workflow():
    try:
        folder_path = create_folder()
        earliest_date = get_earliest_date(days)
        news_list = get_news_list(earliest_date)
        articles_list = get_articles(news_list)
        print(f"运行成功,共采集文章{len(articles_list)}篇,追溯至日期{earliest_date}")
    except Exception as e:
        print(f'文章采集过程中出现错误：{e}')

#读取文章原文并结构化处理
def get_formatted_article(row):
    fields_list = ["title","tags","dek","content"]
    article_formatted = []
    for field in fields_list:
        if row[field] != "":
            field_str = f"{field}:{row[field]}"
            article_formatted.append(field_str)
    article_formatted_str = "\n".join(article_formatted)
    return article_formatted_str

#读取文章列表,导出series
def formatting_articles(df):
    df['结构化文本'] = df.apply(get_formatted_article, axis=1).tolist()
    return df


#利用文章构建摘要prompt模板
def chat_prompt_briefing():
    prompt_template = ChatPromptTemplate([
        ("system","你是专精政治、经济和科技领域的中英文分析师，负责根据信源为用户提供信息简报"),
        ("user","请为提供的新闻媒体内容提供细节详尽的中文简报，原文如下：\n{text}")
    ])
    # prompt = prompt_template.invoke({
    #     "text": text
    # })
    # return prompt
    return prompt_template

#条目摘要
def row_briefing(row):
    try:
        column_name = "结构化文本"
        text = getattr(row, column_name)
        briefing_chain = chat_prompt_briefing() | llm
        result = briefing_chain.stream(text)
        content = []
        for chunk in result:
            content.append(chunk.content)
        content = "".join(content)
        return content
    except Exception as e:
        print(f"在Index{row.Index}生成摘要时出现错误:{e}")
        return "error"

#并发生成摘要
def generate_briefing(df):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        briefings = []
        for row in df.itertuples():
            index = row.Index
            future = executor.submit(row_briefing, row)
            # print(f"已提交Index{row.Index}摘要生成任务")
            futures[future] = index

        results = {idx : None for idx in futures.values()}

        with tqdm(total=len(futures),desc="摘要生成任务进度",ncols=80,) as pbar:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"为文档Index{idx}生成摘要时发生错误：{e}")
                finally:
                    pbar.update(1)

        for idx in sorted(results):
            briefings.append(results[idx])

        df.loc[:, "AI摘要"] = briefings

        file_name = "archived_briefings.csv"
        df.to_csv(file_name,index=False)

        return df



#利用摘要构建评估prompt
def chat_prompt_evaluation():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "您是一位数据主权分析师，需要根据提供的新闻条目进行分析，提供新闻条目与数据封锁的相关程度"),
        ("user", (
            "【新闻摘要】：\n{text}\n\n"  # 明确换行分隔
            "【限制数据获取的原因】：\n"
            "（1）国家安全风险；（2）国家军事、科技、经济等领域竞争，封锁竞争对手；（3）隐私保护；（4）知识产权保护；"
            "（5）伦理道德约束(如LGBTQ或anti LGBTQ)；（6）数据标准更新；（7）数据机构预算削减；（8）数据机构/人员裁撤；（9）技术故障；"
            "（10）数据老化，停止维护；（11）政府审查数据\n\n"
            "【评估矩阵】:\n"
            "按照1-5等级进行评估（当信息明确针对中国时适当提升评级）\n\n"
            "【输出格式】：请严格使用以下JSON格式：\n"
            "{{\"评估\": int, \"分析\": str}}"  # 使用双大括号转义 + 转义内部引号
        ))
    ])
    return prompt_template

#输出格式--Pydantic类型
class Evaluation(BaseModel):
    评估:int = Field(description="评估")
    分析:str = Field(description="分析")

#条目评估
@retry(
    stop=stop_after_attempt(3),  # 最大重试3次
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"采集失败，重试第 {retry_state.attempt_number} 次\n"
        f"错误行索引：{retry_state.args[0].Index}\n"
        f"错误：{retry_state.outcome.exception()}"
    ),
    retry_error_callback  =  lambda _: {"评估": None, "分析": None}
)
def row_evaluation(row):
    column_name = "AI摘要"
    text = getattr(row, column_name)
    if text == "":
        text = getattr(row, "content")
    json_output_parser = JsonOutputParser(pydantic_object=Evaluation)
    evaluation_chain = chat_prompt_evaluation() | llm | json_output_parser
    result = evaluation_chain.stream(text)
    content = []
    for chunk in result:
        content.append(chunk)
    if len(content)>0:
        content = content[-1]
    else:
        try:
            result = evaluation_chain.invoke(text)
        except openai.BadRequestError as e:
            error_message = e.response.json()["error"]["message"]
            print(f"新闻条目 Index{row.Index} 触发错误：{error_message}")
            content = {"评估": None, "分析": None}
    return content

def generate_evaluations(df):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        evaluations = []
        for row in df.itertuples():
            index = row.Index
            future = executor.submit(row_evaluation, row)
            # print(f"已提交Index{row.Index}摘要生成任务")
            futures[future] = index

        results = {idx: None for idx in futures.values()}

        with tqdm(total=len(futures), desc="评估任务进度", ncols=80, ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"为文档Index{idx}生成评估时发生错误：{e}")
                finally:
                    pbar.update(1)

        for idx in sorted(results):
            evaluations.append(results[idx])

        evaluation_df = pd.DataFrame(evaluations)
        evaluation_df = evaluation_df[["分析","评估"]]
        evaluation_df = evaluation_df.rename(columns={"分析":"AI分析","评估":"相关性评分"})

        result = df.join(evaluation_df)
        articles_evaluation_sorted = result.sort_values(by="相关性评分", ascending=False)
        articles_evaluation_sorted.to_csv("Articles_Evaluation.csv",index=False)

    return result
#文章评估
def evaluation_workflow():
    try:
        articles_df = pd.read_csv("articles.csv")
        articles_df = formatting_articles(articles_df)
        articles_df = generate_briefing(articles_df)
        """下方注释代码实现从存档摘要开始评估"""
        # csv_file = "archived_briefings.csv"
        # all_articles_df = pd.read_csv(csv_file)
        # articles_df=all_articles_df #方便测试时切片
        """评估"""
        try:
            result = generate_evaluations(articles_df)
            news_num = len(result)
            print(type(result))
            unevaluated_num = result["相关性评分"].isna().sum()
            evaluated_num = news_num - unevaluated_num
            print(f"已完成，共{len(result)}条消息，完成评估{evaluated_num}条，未评估{unevaluated_num}条")

        except Exception as e:
            print(e)
    except Exception as e:
        print(f"文章评估执行失败：{e}")

def main():
    try:
        scoping_workflow()
        evaluation_workflow()
    except Exception as e:
        print("执行中出现错误：{e}")

if __name__ == "__main__":
    main()