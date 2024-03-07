from argparse import RawDescriptionHelpFormatter
from typing import List, Dict, Any, Tuple, Optional
from numpy import average
import requests
from multiprocessing import Pool, TimeoutError
import argparse, time, os
import pandas as pd
import numpy as np
API_REQUEST = "generate"
#API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
#os.environ['NO_PROXY'] = 'localhost'

def query(query, idx=0, config=None) :
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"http://{config.ip_address}:8080/{API_REQUEST}"
    ret = 1
    pid = str(os.getpid()) + "_" + str(idx)
    params = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample}
    req = {"inputs": query, "parameters": params}
    start = time.time()
    response_raw = requests.post(url, json=req)
    interval=time.time() - start
    print(f"{{pid: {pid}}}, {{time: {interval}}}")
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        ret = 0
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        ret = 0
        raise Exception(", ".join(response["errors"]))

    # Format response
    print(f"response={response}")
    return interval, ret

def benchmark(conig, query_idx):
    print(f"Performance benchmark ! Use the default question")
    question="What is Deep Learning?"
    interval, ret = query(question, query_idx, config)
    return interval, ret

def parse_cmd():
    desc = 'multi-process benchmark for haystack...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-p', type=int, default=1, dest='processes', help='How many processes are used for the process pool')
    args.add_argument('-n', type=int, default=1, dest='query_number', help='How many querys will be executed.')
    args.add_argument('-c', type=int, default=0, dest='real_concurrent', help='Use the real concurrent', choices=[0, 1])
    args.add_argument('--ip', type=str, default='localhost', dest='ip_address', help='Ip address of backend server')
    args.add_argument('--do_sample', type=bool, default=False, dest='do_sample', help='do_sample')
    args.add_argument('--max_new_tokens', type=int, default=128, dest='max_new_tokens', help='max_new_tokens')
    return args.parse_args()



if __name__ == '__main__':
    config = parse_cmd()
    # start 4 worker processes
    result = pd.DataFrame()
    start = time.time()
    if config.real_concurrent == 0 :
        with Pool(processes=config.processes) as pool:

            multiple_results = [pool.apply_async(benchmark, (config, i)) for i in range(config.query_number)]
            for res in multiple_results:
                interval,ret = res.get()
                d = {'time':[interval], 'success':[ret]}
                df = pd.DataFrame(data=d)
                result = pd.concat([result, df], ignore_index=True)
    else :
        with Pool(processes=config.processes) as pool:
            for num in range(0, int(config.query_number/config.processes)):
                print(f"concurrent index = {num}")
                multiple_results = [pool.apply_async(benchmark, (config, num)) for i in range(config.processes)]
                for res in multiple_results:
                    interval,ret = res.get()
                    d = {'time':[interval], 'success':[ret]}
                    df = pd.DataFrame(data=d)
                    result = pd.concat([result, df], ignore_index=True)    

    total_time = time.time() - start
    result = result.sort_values(by=['time'])
    average_time = result.apply(np.average, axis=0)
    print(average_time)
    print(f"{{query_number: {config.query_number}}}, {{total_time: {total_time}}}, {{fps: {config.processes/average_time.at['time']}}}")
    print("Benchmark Done!")
