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
#API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
#os.environ['NO_PROXY'] = 'localhost'

random.seed(42)
#queries_file = 'test_set_queries.tsv'
queries_file = 'err3.txt'

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

    url = f"http://{config.ip_address}:8080/{API_REQUEST}"
    ret = 1
    pid = str(os.getpid()) + "_" + str(idx)
    rad_max_new_tokens = random.randint(1, config.max_new_tokens)
    queries_len = len(queries)
    query_id = random.randint(0, queries_len-1)
    query_txt = list(queries.values())[query_id]
    params = {"max_new_tokens": rad_max_new_tokens, "do_sample": config.do_sample, "process_id": pid, "seed" : 42}
    print(f"query_txt = {query_txt}, params = {params}, query_id = {query_id}")
    req = {"inputs": query_txt, "parameters": params}
    start = time.time()
    response_raw = requests.post(url, json=req)
    interval=time.time() - start
    print(f"{{pid: {pid}}}, {{time: {interval}}}")
    
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        ret = 0
        print(f"connection Error!!!!!!!!!!!!!!!!!!!!!,query_id={query_id}")

        #raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        ret = 0
        print(f"backend  Error!!!!!!!!!!!!!!!!!!!!!, query_id={query_id}")
        #raise Exception(", ".join(response["errors"]))
    err_query = None
    if ret == 0 :
        err_query = query_txt
    # Format response
    print(f"response={response}, query_txt={query_txt}, query_id={query_id}")
    return interval, ret, err_query

def benchmark(conig, query_idx, queries):
    print(f"Performance benchmark ! Use the default question")
    question="What is Deep Learning?"
    interval, ret, err_query = query(question, query_idx, config, queries)
    return interval, ret, err_query

def parse_cmd():
    desc = 'multi-process benchmark for haystack...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-p', type=int, default=1, dest='processes', help='How many processes are used for the process pool')
    args.add_argument('-n', type=int, default=1, dest='query_number', help='How many querys will be executed.')
    args.add_argument('-c', type=int, default=0, dest='real_concurrent', help='Use the real concurrent', choices=[0, 1])
    args.add_argument('--ip', type=str, default='localhost', dest='ip_address', help='Ip address of backend server')
    args.add_argument('--do_sample', type=bool, default=False, dest='do_sample', help='do_sample')
    args.add_argument('--max_new_tokens', type=int, default=128, dest='max_new_tokens', help='max_new_tokens')
    args.add_argument('--dump_file', type=str, default="dump.txt", dest='dump_file', help='dump_file')
    return args.parse_args()



if __name__ == '__main__':
    config = parse_cmd()
    # start 4 worker processes
    result = pd.DataFrame()
    start = time.time()
    queries= load_queries(queries_file)
    if config.real_concurrent == 0 :
        with Pool(processes=config.processes) as pool:

            multiple_results = [pool.apply_async(benchmark, (config, i, queries)) for i in range(config.query_number)]
            for res in multiple_results:
                interval,ret, txt = res.get()
                d = {'time':[interval], 'success':[ret]}
                if txt != None:
                    dump_error(config.dump_file, txt)
                df = pd.DataFrame(data=d)
                result = pd.concat([result, df], ignore_index=True)
    else :
        with Pool(processes=config.processes) as pool:
            for num in range(0, int(config.query_number/config.processes)):
                print(f"concurrent index = {num}")
                multiple_results = [pool.apply_async(benchmark, (config, num, queries)) for i in range(config.processes)]
                for res in multiple_results:
                    interval,ret, txt = res.get()
                    d = {'time':[interval], 'success':[ret]}
                    df = pd.DataFrame(data=d)
                    result = pd.concat([result, df], ignore_index=True)    
                    if txt != None:
                        dump_error(config.dump_file, txt)

    total_time = time.time() - start
    result = result.sort_values(by=['time'])
    average_time = result.apply(np.average, axis=0)
    print(average_time)
    print(f"{{query_number: {config.query_number}}}, {{total_time: {total_time}}}, {{fps: {config.processes/average_time.at['time']}}}")
    print("Benchmark Done!")
