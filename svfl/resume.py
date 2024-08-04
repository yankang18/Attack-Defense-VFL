if __name__ == '__main__':
    pipeline_log = open("./pipeline_main.log").readlines()
    start = []
    finished = []
    for line in pipeline_log:
        line = line.strip()
        if "start" in line:
            start.append(line.split("start")[0].split("&& ")[1])
        if "finished " in line:
            print(line)
            finished.append(line.split("finished")[0].split("&& ")[1])
    resume = [t for t in start if t not in finished]
    print(resume)