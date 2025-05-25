import argparse
from tokenizers import Tokenizer
import os

def extend_tokenizer_variant3(args):
    old_tokenizer_path = os.path.join("./checkpoints/XTTS_v2_orig", "vocab.json")
    root_tokenizer_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files")
    tokenizer_file_path = os.path.join(root_tokenizer_path, "vocab.json")

    if not os.path.exists(old_tokenizer_path):
        print(f"Ошибка: Файл токенизатора не найден по пути {old_tokenizer_path}")
        print("Пожалуйста, убедитесь, что оригинальные файлы модели XTTS (включая vocab.json) загружены в эту директорию.")
        return

    print(f"Загрузка существующего токенизатора из {old_tokenizer_path}...")
    tokenizer = Tokenizer.from_file(old_tokenizer_path)

    tokens_to_add_default = [
        "а́", "е́", "и́", "о́", "у́", "ы́", "э́", "ю́", "я́",  # Строчные гласные с ударением
        "А́", "Е́", "И́", "О́", "У́", "Ы́", "Э́", "Ю́", "Я́",  # Заглавные гласные с ударением
        " ́"  # Отдельный знак ударения (U+0301 COMBINING ACUTE ACCENT)
       
        # "мо́", "сло́г", "приве́т"
    ]
    
    final_tokens_to_add = list(tokens_to_add_default) 

    # Добавляем токены из файла, если указан
    if args.tokens_to_add_file and os.path.exists(args.tokens_to_add_file):
        print(f"Чтение дополнительных токенов из файла: {args.tokens_to_add_file}")
        with open(args.tokens_to_add_file, 'r', encoding='utf-8') as f:
            additional_tokens_from_file = [line.strip() for line in f if line.strip()]
            final_tokens_to_add.extend(additional_tokens_from_file)
            print(f"Добавлено {len(additional_tokens_from_file)} токенов из файла.")
    
    # Добавляем токены из аргументов командной строки
    if args.tokens_to_add:
        print(f"Добавление токенов из аргументов командной строки: {args.tokens_to_add}")
        final_tokens_to_add.extend(args.tokens_to_add)

    # Удаляем дубликаты и сортируем для консистентности (опционально, но полезно)
    final_tokens_to_add = sorted(list(set(final_tokens_to_add)))

    if not final_tokens_to_add:
        print("Список токенов для добавления пуст. Никаких изменений не будет внесено.")
        return

    print(f"Текущий размер словаря: {tokenizer.get_vocab_size()}")
    print(f"Токены для добавления ({len(final_tokens_to_add)} шт.): {final_tokens_to_add}")

    num_added_tokens = tokenizer.add_tokens(final_tokens_to_add)

    if num_added_tokens > 0:
        print(f"Добавлено {num_added_tokens} новых токенов.")
        print(f"Новый размер словаря: {tokenizer.get_vocab_size()}")

        print(f"Сохранение обновленного токенизатора в {tokenizer_file_path}...")
        tokenizer.save(tokenizer_file_path)
        print("Токенизатор успешно обновлен и сохранен.")
    else:
        print("Новых токенов для добавления не найдено (возможно, они уже существуют в словаре).")
        print("Файл токенизатора не изменен.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расширение словаря существующего XTTS токенизатора путем добавления новых токенов (Вариант 3).")
    parser.add_argument(
        "--output_path", 
        default=os.path.join("run", "training"), 
        type=str, 
        help="Путь до папки, содержащей подпапку XTTS_v2.0_original_model_files с файлом vocab.json."
    )
    
    parser.add_argument(
        "--tokens_to_add_file",
        type=str,
        default=None,
        help="Путь к файлу, где каждая строка - это токен для добавления (например, файл с гласными с ударениями)."
    )
    parser.add_argument(
        "--tokens_to_add",
        nargs="+", # Позволяет передавать несколько токенов: --tokens_to_add "мо́" "сло́г"
        default=[],
        help="Список токенов для добавления, передаваемых напрямую через командную строку."
    )

    args = parser.parse_args()
    extend_tokenizer_variant3(args)