import torch
from text_process import tokenize, Vocab
import random
import os
import re


def read_blocks(data_dir):
    with open(data_dir, 'r') as f:
        p_lines = f.readlines()
    res_lines = []
    for line in p_lines:
        if line[:4] == '[ph]':
            continue
        res_lines.append(line[:-1].split('\t'))

    return res_lines


def get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.

    Defined in :numref:`sec_bert`"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_nsp_data_from_paragraph(block, blocks, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(block) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(block[i], block[i + 1], blocks)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
        match_jump_id = re.match(r'^JUMP_ADDR_([0-9]*)$', block[i][-1], re.M | re.I)
        if match_jump_id:
            jump_addr = int(match_jump_id.group(1))
            tokens_a, tokens_b, is_next = get_next_sentence(block[i], block[jump_addr], blocks)
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


class _JTransTextDataset(torch.utils.data.Dataset):
    def __init__(self, blocks, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        blocks = [tokenize(block, token='word') for block in blocks]
        instructions = [instruction for block in blocks for instruction in block]
        self.vocab = Vocab(instructions, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for block in blocks:
            examples.extend(_get_nsp_data_from_paragraph(block, blocks, self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids,
         self.all_segments,
         self.valid_lens,
         self.all_pred_positions,
         self.all_mlm_weights,
         self.all_mlm_labels,
         self.nsp_labels) = pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx],
                self.all_segments[idx],
                self.valid_lens[idx],
                self.all_pred_positions[idx],
                self.all_mlm_weights[idx],
                self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_jtrans(data_dir, batch_size, max_len, num_workers):
    blocks = read_blocks(data_dir)
    train_set = _JTransTextDataset(blocks, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    data_path = './data/paragraphs.txt'
    load_data_jtrans(data_path, batch_size, max_len, 4)
