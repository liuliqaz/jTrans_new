import gc
from collections import Counter
import os


def read_blocks(data_dir):
    res_lines = []
    # func_list = []
    with open(data_dir, 'r') as f:
        while True:
            # 读取1000行
            lines = f.readlines(10000)

            # 如果已经读取完了所有行，退出循环
            if not lines:
                break

            # 处理读取到的1000行数据
            for line in lines:
                if line[:4] == '[ph]':
                    continue
                # func_list.append(line[:-1].split('\t'))
                res_lines.extend([len(block.split()) for block in line[:-1].split('\t')])
    del lines
    gc.collect()
    # print('[data_read]total_functions=', len(func_list))
    print('[data_read]total_blocks=', len(res_lines))

    # for idx, func in enumerate(res_lines):
    #     res_lines[idx] = [block.split() for block in func]
    #     if idx % 50000 == 0:
    #         gc.collect()

    # lengths = [len(sample) for sample in res_lines]
    lengths = res_lines

    total_count = len(lengths)
    length_count = Counter(lengths)

    del lengths
    gc.collect()

    # 将统计结果转换成list并按照长度从小到大排序
    sorted_length_count = sorted(length_count.items(), key=lambda x: x[0])

    del length_count
    gc.collect()

    # 计算总样本数量和95分位的长度
    cumulative_count = 0
    max_len = 0
    for length, count in sorted_length_count:
        cumulative_count += count
        if cumulative_count / total_count >= 0.95:
            max_len = length
            break

    print("Total samples:", total_count)
    print("95 percentile length:", max_len)

    # new_lines = random.sample(res_lines, 500000)
    # del res_lines
    # gc.collect()
    # print('[data_read]new_lines_len=', len(new_lines))
    # len_map = {}
    # for blocks in new_lines:
    #     for b in blocks:
    #         if len(b) in len_map:
    #             len_map[len(b)] += 1
    #         else:
    #             len_map[len(b)] = 1
    # utils.draw_box(len_map)
    # utils.draw_column(len_map)


def find_all_file(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(os.path.join(dir_path, path))
    return res


if __name__ == '__main__':
    # read_blocks('../jTrans/small_jTrans_proj.txt')
    files = find_all_file('./data_set')
    print(files[0], len(files))

