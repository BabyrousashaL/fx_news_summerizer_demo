import concurrent.futures
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import ftfy
import openai
import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from 线索获取.github_demo.fox_articles import stream_url

#全局变量声明
"""文件参数"""
path = r''#请填写文件存放父文件夹，脚本运行后会在此路径下生成带有时间戳的子文件夹用来存放数据
"""时间参数"""
days = 2 #追溯天数
"""大模型参数"""
zhipuai_api_key = ""#请填写api-key
model = "glm-4-flash"
base = "https://open.bigmodel.cn/api/paas/v4/"
llm= ChatOpenAI(
    api_key=zhipuai_api_key,
    model=model,
    base_url=base
)
max_workers =20 #大模型任务线程数
"""fxvideo api"""
stream_url=''#demo新闻视频流api
player_url=''#demo新闻视频详情api

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
            browser = p.chromium.launch(headless=True)
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

#检测条目id是否为纯数字，非纯数字id为广播，无字幕，不采集
def check_pure_num_id(str):
    return re.search(r'^[0-9]+$', str) is not None


#按日期获取api v3中的所有fox article
def get_news_list(earliest_date):
    try:
        """获取api提供的新闻数据"""
        all_news_articles = []
        url = stream_url
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
        print(f"采集完成，已采集新闻视频{len(all_news_articles)}条")
        return all_news_articles
    except Exception as e:
        print(f"访问错误：{e}")

#解析新闻视频列表，提取所需字段
def get_videos_metadata(list):
    videos = []
    for item in list:
        attributes_dict = item['attributes']
        id = item.get("id")
        if check_pure_num_id(id) is True:
            media_tags = attributes_dict.get('fn__media_tags')
            if all(i not in media_tags for i in ['full_episode','live_stream']): #使用tag去除无法解析的全集和直播
                tags = ';'.join(media_tags)
                title = attributes_dict.get('title')
                if "description" in attributes_dict:
                    description = item['attributes'].get('description')
                else:
                    description = ''
                date = attributes_dict.get('last_published_date')
                url = attributes_dict.get("canonical_url")
                metadata = {
                    "id":id,
                    "title":title,
                    "tags":tags,
                    "description":description,
                    "date":date,
                    "url":url
                }
                videos.append(metadata)
        else:
            continue
    df = pd.DataFrame(videos)
    df.to_csv("fox_videos_metadata.csv", index=False)
    return df

#根据vtt url提取字幕
def get_vtt_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_load_state("networkidle")

            # page.screenshot(path=f'screenshot-{browser_type.name}.png')
            content = page.locator('pre').text_content()
            page.close()
        return content
    except Exception as e:
        print(f"访问网页失败：{e}")

#每行字幕采集
@retry(
    stop=stop_after_attempt(3),  # 最大重试3次
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"字幕采集失败，重试第 {retry_state.attempt_number} 次\n"
        f"错误视频编号：{retry_state.args[0].id}\n"
        f"错误：{retry_state.outcome.exception()}"
    ),
    retry_error_callback  =  lambda _: None
)
def get_row_video_subtitles(row):
    id = getattr(row,'id')
    url = f"{player_url}/{id}"#player_url为全局变量
    player_response = get_page_content(url)
    vtt_url = (
        player_response.get("channel", {})
        .get("item", {})
        .get("media-group", {})
        .get("media-subTitle", {})
        .get("@attributes", {})
        .get("href")
    )
    if vtt_url != "":
        raw_vtt = get_vtt_content(vtt_url)
        #清洗可能的编码错误
        subtitles = ftfy.fix_text(raw_vtt)
        #清洗重复字幕条目
        subtitles_list = subtitles.split("\n")
        # print(subtitles_list)
        cleaned_subtitles_list = []
        for i in subtitles_list:
           if i != "":
                if i[-1] == " ":
                    str_i = i[0:-1]
                else:
                    str_i = i
                cleaned_subtitles_list.append(str_i)
        # print(cleaned_subtitles_list)
        unduplicated_subtitles_list = list(dict.fromkeys(cleaned_subtitles_list))
        # print(unduplicated_subtitles_list)
        unduplicated_subtitles = "\n".join(unduplicated_subtitles_list)
        subtitles = unduplicated_subtitles

    else:
        subtitles = ""
    return subtitles

#全部字幕采集
def get_subtitles(df):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        subtitles = []
        for row in df.itertuples():
            index = row.Index
            future = executor.submit(get_row_video_subtitles, row)
            # print(f"已提交Index{row.Index}摘要生成任务")
            futures[future] = index

        results = {idx : None for idx in futures.values()}

        with tqdm(total=len(futures),desc="字幕获取任务进度",ncols=80,) as pbar:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"为视频{getattr(row,'id')}提取字幕时发生错误：{e}")
                finally:
                    pbar.update(1)

        for idx in sorted(results):
            subtitles.append(results[idx])

        df.loc[:, "字幕文件"] = subtitles

        file_name = "fox_videos_subtitles.csv"
        df.to_csv(file_name,index=False)
    return df

#生成摘要prompt
def chat_prompt_briefing():
    prompt_template = ChatPromptTemplate([
        ("system","你是专精政治、经济和科技领域的中英文分析师，负责根据信源为用户提供信息简报"),
        ("user","请为提供的福克斯新闻视频内容提供细节详尽的中文简报，原文如下：\n{text}")
    ])
    return prompt_template

#基于标题/描述/字幕/tag生成摘要
@retry(
    stop=stop_after_attempt(3),  # 最大重试3次
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"生成摘要失败，重试第 {retry_state.attempt_number} 次\n"
        f"错误行索引：{retry_state.args[0].Index}，视频编号：{getattr(retry_state.args[0], 'id')}\n"
        f"错误：{retry_state.outcome.exception()}"
    ),
    retry_error_callback  =  lambda _: None
)
def row_briefing(row):
    try:
        text = getattr(row, '字幕文件')
        title = getattr(row, 'title')
        dek = getattr(row, 'description')
        tags = getattr(row, 'tags')
        video_info = f'title:{title}\ndek:{dek}\ntags:{tags}\ntext:{text}'
        briefing_chain = chat_prompt_briefing() | llm
        result = briefing_chain.stream(video_info)
        content = []
        for chunk in result:
            content.append(chunk.content)
        content = "".join(content)
        return content
    except Exception as e:
        print(f"在Index{row.Index}生成摘要时出现错误:{e}")
        return None

#为每条联邦公报生成摘要，并写入dataframe
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

        file_name = "archived_video_briefings.csv"
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
            "（5）伦理道德约束(如LGBTQ或anti LGBTQ)；（6）数据标准更新；（7）数据机构预算削减；（8）数据或科研机构/人员裁撤；（9）技术故障；"
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
        videos_evaluation_df = result.sort_values(by="相关性评分",ascending=False)
        videos_evaluation_df.to_csv("Videos_Evaluation.csv",index=False)

    return videos_evaluation_df

def main():
    folder_path = create_folder()
    all_news_videos = get_news_list(get_earliest_date(days))
    df = get_videos_metadata(all_news_videos)
    # df = pd.read_csv("fox_videos_metadata.csv")
    df_subtitles = get_subtitles(df)
    # df_subtitles = pd.read_csv("fox_videos_subtitles.csv")
    df_briefings = generate_briefing(df_subtitles)
    # df_briefings = pd.read_csv("archived_video_briefings.csv")
    df_evaluations = generate_evaluations(df_briefings)
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)