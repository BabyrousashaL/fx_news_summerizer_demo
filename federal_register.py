import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from types import NoneType

import openai
import pandas as pd
import requests
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

#全局变量
"""大模型参数"""
zhipuai_api_key = ""#请使用自己的LLM api-key
model = "glm-4-flash"
base = "https://open.bigmodel.cn/api/paas/v4/"
llm= ChatOpenAI(
    api_key=zhipuai_api_key,
    model=model,
    base_url=base
)
max_workers =20 #大模型任务线程数
"""文件路径"""
path = r""#选择存放路径

#获取日期
def get_previous_workday():
    today = datetime.now()
    # 判断今天星期几，如果是周一，则前一个工作日是上周五
    if today.weekday() == 0:
        return today - timedelta(days=3)
    # 如果是周六或周日，则前一个工作日是上周五
    elif today.weekday() in [5, 6]:
        return today - timedelta(days=1 if today.weekday() == 5 else 2)
    # 其他情况，前一个工作日就是昨天
    else:
        return today - timedelta(days=1)


#获取当日新发布联邦公报列表
def get_fr_list(date):
    fr_formatted_date = date.strftime("%Y-%m-%d")
    fr_issues_url = f"https://www.federalregister.gov/api/v1/issues/{fr_formatted_date}.json"
    fr_issues_response = requests.get(fr_issues_url)
    if fr_issues_response.status_code == 200:
        fr_issues = json.loads(fr_issues_response.text)
        return fr_issues
    else:
        print(f"访问联邦公报列表失败，状态码{fr_issues_response.status_code}")


#提取联邦公报元数据
def fr_issues_extract(data):
    result = {"Document_Number":[], "Document_Subject":[], "type":[], "Agency":[]}
    for agency in data['agencies']:
        if len(agency['document_categories']) > 0:
            agency_name = agency['name']
            if 'see_also' in agency:
                alter_names = []
                for i in agency['see_also']:
                    alter_name = i['name']
                    alter_names.append(alter_name)
                sub_agency_name = '\n'.join(alter_names)
                full_agency_name = f"{agency_name}\n{sub_agency_name}"
            else:
                full_agency_name = agency_name
            for cat in agency['document_categories']:
                document_type = cat['type']
                for doc in cat['documents']:
                    document_subject = doc['subject_1']
                    if "subject_2" in doc:
                        document_subject += doc['subject_2']
                    for num in doc['document_numbers']:
                        result['Document_Number'].append(num)
                        result['Document_Subject'].append(document_subject)
                        result['type'].append(document_type)
                        result['Agency'].append(full_agency_name)
        else:
            continue
    fr_issues_data = pd.DataFrame(result)
    return fr_issues_data

#获取每条联邦公报内容
@retry(
    stop = stop_after_attempt(3),
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"获取联邦公报内容失败，重试第 {retry_state.attempt_number} 次\n"
        f"错误行索引：{retry_state.args[0].Index}，联邦公报编号：{getattr(retry_state.args[0],'Document_Number')}\n"
        f"错误：{retry_state.outcome.exception()}"
    ),
    retry_error_callback=lambda _:  {
        "url": None,
        "pdf_url": None,
        "text_url": None,
        "title": None,
        "abstract": None,
        "content": None
    }
)
def get_row_content(row):
    column_name = "Document_Number"
    fr_num = getattr(row,column_name)
    fr_info_url = f"https://www.federalregister.gov/api/v1/documents/{fr_num}.json"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(fr_info_url)
        page.wait_for_load_state("networkidle")
        fr_info = page.locator('pre').text_content()
        fr_info = json.loads(fr_info)
        page.close()
    url = fr_info['html_url']
    pdf_url = fr_info['pdf_url']
    text_url = fr_info['body_html_url']
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(text_url)
        page.wait_for_load_state("networkidle")
        content = page.locator('body').inner_text()
        page.close()
    title = fr_info['title']
    abstract = fr_info['abstract']
    content_dict = {
        "url": url,
        "pdf_url": pdf_url,
        "text_url": text_url,
        "title": title,
        "abstract": abstract,
        "content": content
    }
    return content_dict

# 获取列表所有联邦公报信息
def fr_content_extract(df):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        fr_content_list = []
        for row in df.itertuples():
            index = row.Index
            issue_number = getattr(row, 'Document_Number')
            future = executor.submit(get_row_content,row)
            futures[future] = index
        #预先初始化，为每行预留写入位置
        results = {idx: None for idx in futures.values()}

        with tqdm(total=len(futures),desc="联邦公报内容提取任务进度",ncols=80,) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"提取联邦公报{issue_number}内容时发生错误：{e}")
                finally:
                    pbar.update(1)
        #按Index顺序将提取的内容写入
        for idx in sorted(results):
            fr_content_list.append(results[idx])
    fr_content_df = pd.DataFrame(fr_content_list)
    fr_df=df.join(fr_content_df)
    return fr_df

#生成摘要prompt
def chat_prompt_briefing():
    prompt_template = ChatPromptTemplate([
        ("system","你是专精政治、经济和科技领域的中英文分析师，负责根据信源为用户提供信息简报"),
        ("user","请为提供的美国联邦公报内容提供细节详尽的中文简报，原文如下：\n{text}")
    ])
    return prompt_template

#生成联邦公报的摘要
@retry(
    stop=stop_after_attempt(3),  # 最大重试3次
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"生成摘要失败，重试第 {retry_state.attempt_number} 次\n"
        f"错误行索引：{retry_state.args[0].Index}，联邦公报编号：{getattr(retry_state.args[0], 'Document_Number')}\n"
        f"错误：{retry_state.outcome.exception()}"
    ),
    retry_error_callback  =  lambda _: None
)
def row_briefing(row):
    try:
        column_name = "content"
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
            for future in as_completed(futures):
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

        file_name = "archived_fr_briefings.csv"
        df.to_csv(file_name,index=False)

        return df

#利用摘要构建评估prompt
def chat_prompt_evaluation():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "您是一位数据主权分析师，需要根据提供的联邦公报摘要进行分析，提供联邦公报与数据封锁的相关程度"),
        ("user", (
            "【联邦公报摘要】：\n{text}\n\n"  # 明确换行分隔
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

#生成联邦条目评估
@retry(
    stop=stop_after_attempt(3),  # 最大重试3次
    wait=wait_exponential(multiplier=4, min=4, max=20),  # 指数退避等待
    before_sleep=lambda retry_state: print(
        f"评估失败，重试第 {retry_state.attempt_number} 次\n"
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

#为每条联邦公报生成评估并写入dataframe
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
            for future in as_completed(futures):
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
        evaluation_sorted = result.sort_values(by="相关性评分", ascending=False)
        evaluation_sorted.to_csv("FR_Evaluation.csv",index=False)

    return evaluation_sorted

#整理工作流-获取联邦公报
def scoping_fr_workflow():
    date = get_previous_workday()
    datestr = date.strftime(r'%Y%m%d')
    folder_name = f"FR_{datestr}"
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.chdir(folder_path)
    current_dir = os.getcwd()
    print(f'当前目录已经切换至{current_dir}')
    content_file_name = f'{datestr}_fr_content.csv'
    if os.path.isfile(content_file_name):
        df = pd.read_csv(content_file_name)
        return df
    else:
        fr_issues_dict = get_fr_list(date)
        print('已获取联邦公报列表')
        fr_issues_data = fr_issues_extract(fr_issues_dict)
        print("已获取联邦公报元数据")
        fr_contents = fr_content_extract(fr_issues_data)
        # 保存联邦公报数据至csv
        fr_contents.to_csv(f'{datestr}_fr_content.csv',index=False)
        return fr_contents
def main():
    try:
        fr_contents = scoping_fr_workflow()
        #生成摘要
        fr_contents_with_briefing = generate_briefing(fr_contents)
        #生成评估
        fr_contents_with_evaluation = generate_evaluations(fr_contents_with_briefing)
    except Exception as e:
        print(f"联邦公报评估执行失败，错误：{e}")

if __name__ == '__main__':
    main()
