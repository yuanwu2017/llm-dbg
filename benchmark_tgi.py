from argparse import RawDescriptionHelpFormatter
from typing import List, Dict, Any, Tuple, Optional
from numpy import average
import requests
from multiprocessing import Pool, TimeoutError
import argparse, time, os
import pandas as pd
import numpy as np
import random
API_REQUEST = "generate"
TEI_API_REQUEST = "embed"
API_STREAM_REQUEST = ""

#queries_file = 'test_set_queries.tsv'
queries_file = 'default.txt'

def load_queries(filename):
    queries = {}
    qid = 0
    with open(filename) as fp:
        for q in fp.readlines():
            #qid, query_text = q.strip().split("\t")
            #query_text = query_text.strip()
            query_text = q.strip()
            queries[int(qid)] = query_text
            qid = qid+1
    return queries



def dump_error(file, query_txt) :
    with open(file, "a") as dump_file:
        dump_file.write(query_txt)
        dump_file.write("\n")

def query(query, idx=0, config=None, queries=None) :
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """
    api_path = API_REQUEST if config.workload == "tgi" else TEI_API_REQUEST
    if config.stream is True:
        api_path = API_STREAM_REQUEST
    url = f"http://{config.ip_address}:{config.port}/{api_path}"
    print(url)
    ret = 1
    pid = str(os.getpid()) + "_" + str(idx)
    rad_max_new_tokens = random.randint(1, config.max_new_tokens)
    queries_len = len(queries)
    query_id = random.randint(0, queries_len-1)
    query_txt = list(queries.values())[query_id]
    params = {}
    if config.workload == "tgi" :
        params = {"max_new_tokens": rad_max_new_tokens, "do_sample": config.do_sample, "seed" : config.seed}
        if config.temperature is not None:
            params.update({"temperature":config.temperature}) 
        if config.top_k is not None:
            params.update({"top_k":config.top_k}) 
        if config.top_p is not None:
            params.update({"top_p":config.top_p}) 
        if config.repetition_penalty is not None:
            params.update({"repetition_penalty":config.repetition_penalty}) 
        if config.typical_p is not None:
            params.update({"typical_p":config.typical_p}) 
    req = {"inputs": query_txt, "parameters": params}
    print(f"req={req}")
    start = time.time()
    response_raw = requests.post(url, json=req, stream=config.stream)
    report = []
    token_num = 0
    first_token_time = None
    second_token_time = None
    if config.stream is True:
        for resp in response_raw.iter_content():
            report.append(resp.decode('utf-8'))
            if token_num == 0:
                first_token_time =  time.time()-start
            if token_num == 1:
                second_token_time = time.time() - start - first_token_time
            result = "".join(report).strip()
            token_num = token_num + 1
    interval=time.time() - start
    print(f"{{pid: {pid}}}, {{time: {interval}}}, query_txt: {query_txt}, params:{params}")
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        ret = 0
        print(f"Error happend! err_num ={response_raw.status_code}, pid={pid}")
        raise Exception(f"{vars(response_raw)}")
        
    #response = response_raw.json()
    # if "errors" in response:
    #     ret = 0
    #     print(f"backend  Error!!!!!!!!!!!!!!!!!!!!!, query_id={query_id}")
    #     #raise Exception(", ".join(response["errors"]))
    err_query = None
    if ret == 0 :
        err_query = query_txt
    # Format response
    return interval, ret, err_query, first_token_time, second_token_time

def benchmark(conig, query_idx, queries):
    question="What is Deep Learning?"
    interval, ret, err_query, first, second = query(question, query_idx, config, queries)
    return interval, ret, err_query, first, second

def parse_cmd():
    desc = 'multi-process benchmark for haystack...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-p', type=int, default=1, dest='processes', help='How many processes are used for the process pool')
    args.add_argument('-n', type=int, default=1, dest='query_number', help='How many querys will be executed.')
    args.add_argument('-c', type=int, default=0, dest='real_concurrent', help='Use the real concurrent', choices=[0, 1])
    args.add_argument('--seed', type=int, default=42, dest='seed', help='random seed')
    args.add_argument('--ip', type=str, default='localhost', dest='ip_address', help='Ip address of backend server')
    args.add_argument('--port', type=int, default=8080, dest='port', help='Ip port of backend server')
    args.add_argument('--do_sample', type=bool, default=False, dest='do_sample', help='do_sample')
    args.add_argument('--stream', type=bool, default=False, dest='stream', help='query the stream interface of tgi')
    args.add_argument('--temperature', type=float, default=None, dest='temperature', help='generation parameters')
    args.add_argument('--repetition_penalty', type=float, default=None, dest='repetition_penalty', help='generation parameters')
    args.add_argument('--top_k', type=int, default=None, dest='top_k', help='generation parameters')
    args.add_argument('--top_p', type=float, default=None, dest='top_p', help='generation parameters')
    args.add_argument('--typical_p', type=float, default=None, dest='typical_p', help='generation parameters')
    args.add_argument('--max_new_tokens', type=int, default=128, dest='max_new_tokens', help='max_new_tokens')
    args.add_argument('--dump_file', type=str, default=None, dest='dump_file', help='dump_file')
    args.add_argument('--workload', type=str, default="tgi", dest='workload', help='Which workload? tgi or tei')
    return args.parse_args()



if __name__ == '__main__':
    config = parse_cmd()
    random.seed(config.seed)
    result = pd.DataFrame()
    start = time.time()
    queries= load_queries(queries_file)
    if config.real_concurrent == 0 :
        with Pool(processes=config.processes) as pool:

            multiple_results = [pool.apply_async(benchmark, (config, i, queries)) for i in range(config.query_number)]
            for res in multiple_results:
                interval, ret, txt, first, second = res.get()
                d = {'time':[interval], 'success':[ret]}
                if first != None:
                    d.update({"first":first})    
                if second != None:
                    d.update({"second":second})    
                if txt != None and config.dump_file is not None:
                    dump_error(config.dump_file, txt)
                df = pd.DataFrame(data=d)
                result = pd.concat([result, df], ignore_index=True)
    else :
        with Pool(processes=config.processes) as pool:
            for num in range(0, int(config.query_number/config.processes)):
                print(f"concurrent index = {num}")
                multiple_results = [pool.apply_async(benchmark, (config, num, queries)) for i in range(config.processes)]
                for res in multiple_results:
                    interval, ret, txt, first, second = res.get()
                    d = {'time':[interval], 'success':[ret]}
                    if first != None:
                        d.update({"first":first})    
                    if second != None:
                        d.update({"second":second})
                    df = pd.DataFrame(data=d)
                    result = pd.concat([result, df], ignore_index=True)    
                    if txt != None and config.dump_file is not None:
                        dump_error(config.dump_file, txt)

    total_time = time.time() - start
    result = result.sort_values(by=['time'])
    average_time = result.apply(np.average, axis=0)
    print(average_time)
    print(f"{{query_number: {config.query_number}}}, {{total_time: {total_time}}}, {{fps: {config.processes/average_time.at['time']}}}")
    print("Benchmark Done!")
