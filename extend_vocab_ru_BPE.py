import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import shutil
import json

# Язык всегда "ru"
LANGUAGE = "ru"

def combine_tokenizers(old_tokenizer_dir, new_tokenizer_dir, save_dir):
    # Объединяем vocab.json и merges.txt
    old_vocab_path = os.path.join(old_tokenizer_dir, 'vocab.json')
    new_vocab_path = os.path.join(new_tokenizer_dir, 'vocab.json')

    json1 = json.load(open(old_vocab_path, encoding="utf-8"))
    json2 = json.load(open(new_vocab_path, encoding="utf-8"))

    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab:
            new_vocab[word] = idx
            idx += 1
    for word in json2.keys():
        if word not in new_vocab:
            new_vocab[word] = idx
            idx += 1

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding="utf-8") as fp:
        json.dump(new_vocab, fp, ensure_ascii=False, indent=2)

    # Объединяем merges.txt
    old_merges_path = os.path.join(old_tokenizer_dir, 'merges.txt')
    new_merges_path = os.path.join(new_tokenizer_dir, 'merges.txt')
    combined_merges_path = os.path.join(save_dir, 'merges.txt')

    final_merges_lines = []
    has_old_merges = False

    # Читаем старые merges
    if os.path.exists(old_merges_path):
        with open(old_merges_path, 'r', encoding='utf-8') as f:
            old_lines = f.readlines()
            if old_lines:
                final_merges_lines.extend(old_lines)
                has_old_merges = True

    # Читаем и добавляем новые merges
    if os.path.exists(new_merges_path):
        with open(new_merges_path, 'r', encoding='utf-8') as f:
            new_lines = f.readlines()
        
        if new_lines:
            # Гарантируем переход на новую строку, если есть старые merges и они не заканчиваются \n
            if has_old_merges and final_merges_lines and not final_merges_lines[-1].endswith('\n'):
                final_merges_lines[-1] = final_merges_lines[-1].rstrip('\r\n') + '\n'

            start_index_new = 0
            # Пропускаем заголовок новых merges только если старые merges существуют 
            # и новые merges начинаются с комментария-версии
            if has_old_merges and new_lines[0].startswith("#"): # например, #version: 0.2
                start_index_new = 1
            
            final_merges_lines.extend(new_lines[start_index_new:])

    # Удаляем дубликаты merges, сохраняя комментарии и пустые строки
    if final_merges_lines:
        seen_actual_merges = set()
        deduplicated_lines = []
        for line in final_merges_lines:
            stripped_line = line.strip()
            if stripped_line.startswith("#") or not stripped_line: # Комментарий или пустая строка
                deduplicated_lines.append(line)
            else: # Реальное правило слияния
                if stripped_line not in seen_actual_merges:
                    deduplicated_lines.append(line)
                    seen_actual_merges.add(stripped_line)
        
        with open(combined_merges_path, 'w', encoding='utf-8') as f:
            f.writelines(deduplicated_lines)
    elif not os.path.exists(combined_merges_path): # Если merges вообще нет (ни старых, ни новых)
        # Создаем пустой merges.txt, так как модель BPE его ожидает
        open(combined_merges_path, 'w').close()

def extend_tokenizer(args):
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files")

    # Сохраняем текущий токенайзер
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # Тренируем новый токенайзер на новом корпусе (с ударениями и т.д.)
    traindf = pd.read_csv(args.metadata_path, sep="|")
    texts = [str(text).lower() for text in traindf.text.to_list()] # Приводим к нижнему регистру

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=[f"[{LANGUAGE}]"], vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{LANGUAGE}]"])

    new_tokenizer_path = os.path.join(root, "new_tokenizer")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    # Объединяем словари и правила merges
    merged_tokenizer_path = os.path.join(root, "merged_tokenizer")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    # Записываем итоговый tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(
        os.path.join(merged_tokenizer_path, 'vocab.json'),
        os.path.join(merged_tokenizer_path, 'merges.txt')
    )
    tokenizer.add_special_tokens([f"[{LANGUAGE}]"])
    tokenizer.save(os.path.join(root, "vocab.json"))

    # Чистим временные папки
    if os.path.exists(old_tokenizer_path):
        shutil.rmtree(old_tokenizer_path)
    if os.path.exists(new_tokenizer_path):
        shutil.rmtree(new_tokenizer_path)
    if os.path.exists(merged_tokenizer_path):
        shutil.rmtree(merged_tokenizer_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default=os.path.join("run", "training"), type=str, required=False, help="Путь до папки XTTS_v2.0_original_model_files")
    parser.add_argument("--metadata_path", default=os.path.join("datasets", "metadata_half.csv"), type=str, required=False, help="Путь до корпуса (metadata.csv)")
    parser.add_argument("--extended_vocab_size", default=1200, type=int, required=False, help="Размер расширенного словаря")
    args = parser.parse_args()
    extend_tokenizer(args)