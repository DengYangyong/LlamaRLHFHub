# -*- coding: utf-8 -*-
"""
@author: DengYangyong
@description: Expand chinese vocab and tokenizer from llama-2

Train sentencepiece model from chinese pretrain corpus and merge it with the original tokenizer
Citation: https://github.com/shibing624/MedicalGPT/
"""
import os

import hydra
from omegaconf import OmegaConf
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import random


def is_chinese(uchar):
    """Judge whether a unicode is a Chinese character"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """Judge whether a string is all Chinese characters"""
    return all(is_chinese(c) for c in string)


def load_baichuan_vocab(vocab_file):
    words = set()
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words.add(line.strip().split()[0])
    return words


def build_chinese_tokenizer(cfg):
    """ Build chinese tokenizer from pretrain corpus"""

    # Sample 0.1 ratio of the corpus to train the tokenizer
    with open(cfg["corpus_file"], 'r') as f:
        for line in f:
            if random.random() < 0.1:
                with open(cfg["sampled_corpus_file"], 'a') as f:
                    f.write(line)

    # train sentencepiece model
    spm.SentencePieceTrainer.train(
        input=cfg["sampled_corpus_file"],
        model_prefix=cfg["model_prefix"],
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=cfg["max_sentence_length"],
        pad_id=cfg["pad_id"],
        model_type=cfg["model_type"],
        vocab_size=cfg["vocab_size"],
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=cfg["byte_fallback"],
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    # makes segmenter instance and loads the model file
    sp = spm.SentencePieceProcessor()
    model_file = cfg["model_prefix"] + '.model'
    sp.load(model_file)

    # encode: text => id
    print(sp.encode_as_pieces('CLUECorpusSmall这个语料来自 CLUEBenchmark 社区，包含新闻、社区互动、维基百科、评论语料。'))
    print(sp.encode_as_ids('CLUECorpusSmall这个语料来自 CLUEBenchmark 社区，包含新闻、社区互动、维基百科、评论语料。'))


def merge_tokenizer(cfg):
    """Merge chinese tokenizer with the original one"""

    llama_tokenizer = LlamaTokenizer.from_pretrained(cfg["model_name_or_path"])
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(cfg["model_prefix"] + ".model")

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    # Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    added_set = set()
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            print('picec: ', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)
    print(f"[add domain tokens]New model pieces: {len(llama_spm.pieces)}")

    # Add baichuan tokens to LLaMA tokenizer
    vocab = load_baichuan_vocab(cfg["baichuan_vocab_file"])
    print('baichuan vocab len:', len(vocab))
    baichuan_vocab_set = set([i for i in vocab if is_chinese_string(i)])
    print('baichuan chinese vocab size:', len(baichuan_vocab_set))
    print('baichuan vocab head:', list(baichuan_vocab_set)[:10])
    for p in baichuan_vocab_set:
        piece = p
        if piece not in llama_spm_tokens_set and piece not in added_set:
            print('baichuan picec', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)
    print(f"[add baichuan tokens]New model pieces: {len(llama_spm.pieces)}")

    # Save the new model
    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf'  # the path to save Chinese-LLaMA tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/llama_expand_sp.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/llama_expand_sp.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(cfg["model_name_or_path"])
    llama_expand_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(llama_expand_tokenizer.all_special_tokens)
    print(llama_expand_tokenizer.all_special_ids)
    print(llama_expand_tokenizer.special_tokens_map)
    print('old len:', len(llama_tokenizer), ' new len:', len(llama_expand_tokenizer))
    text = "This is a test: CLUECorpusSmall这个语料来自 CLUEBenchmark 社区，包含新闻、社区互动、维基百科、评论语料。"
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by LLaMA expand tokenizer:{llama_expand_tokenizer.tokenize(text)}")


@hydra.main(version_base=None, config_path="../../config", config_name="expand_tokenizer")
def main(cfg):
    # Build chinese tokenizer
    cfg = OmegaConf.to_container(cfg, resolve=True)
    build_chinese_tokenizer(cfg)

    # Merge chinese tokenizer with the original one
    merge_tokenizer(cfg)


if __name__ == '__main__':
    main()
