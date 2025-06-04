import argparse
import os
import torch
from pathlib import Path
import sys # Для sys.exit

def clear_gpu_cache():
    """Очищает кэш GPU, если CUDA доступна."""
    if torch.cuda.is_available():
        print("Очистка кэша GPU...")
        torch.cuda.empty_cache()
        print("Кэш GPU очищен.")

def optimize_model_standalone(input_model_path_str: str, delete_original: bool = False):
    """
    Оптимизирует натренированную модель XTTS, удаляя ненужную для инференса информацию.

    Args:
        input_model_path_str (str): Путь к файлу натренированной модели (.pth).
        delete_original (bool): Если True, удаляет исходный неоптимизированный файл модели
                                после успешной оптимизации.
    
    Returns:
        bool: True, если оптимизация прошла успешно, иначе False.
    """
    input_model_path = Path(input_model_path_str)
    output_model_name = "optimized_model.pth"
    # Сохраняем в той же директории, где и входной файл
    output_model_path = input_model_path.parent / output_model_name

    if not input_model_path.is_file():
        print(f"Ошибка: Входной файл модели не найден по пути {input_model_path}")
        return False

    if input_model_path == output_model_path:
        print(f"Предупреждение: Путь к входной модели ({input_model_path}) совпадает с путем "
              f"сохранения оптимизированной модели ('{output_model_name}'). Файл будет перезаписан.")

    try:
        print(f"Загрузка модели из {input_model_path}...")
        # Загружаем на CPU, чтобы избежать проблем с памятью GPU, если модель большая
        checkpoint = torch.load(input_model_path, map_location=torch.device("cpu"))
        print("Модель загружена.")

        # Удаление состояния оптимизатора
        if "optimizer" in checkpoint:
            del checkpoint["optimizer"]
            print("Состояние оптимизатора удалено.")
        else:
            print("Состояние оптимизатора не найдено в чекпоинте.")

        # Удаление весов DVAE
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            keys_to_delete = [key for key in checkpoint["model"].keys() if "dvae" in key.lower()]
            if keys_to_delete:
                for key in keys_to_delete:
                    del checkpoint["model"][key]
                print(f"Удалено {len(keys_to_delete)} ключей, связанных с DVAE, из state_dict модели.")
            else:
                print("Ключи, связанные с DVAE, не найдены в state_dict модели для удаления.")
        else:
            print("Предупреждение: Ключ 'model' не найден или не является словарем в чекпоинте. Пропуск удаления DVAE.")

        print(f"Сохранение оптимизированной модели в {output_model_path}...")
        torch.save(checkpoint, output_model_path)
        print("Оптимизированная модель успешно сохранена.")

        if delete_original:
            if input_model_path == output_model_path:
                print("Исходная модель была перезаписана, отдельное удаление не требуется.")
            else:
                try:
                    os.remove(input_model_path)
                    print(f"Исходный файл модели {input_model_path} удален.")
                except OSError as e:
                    print(f"Ошибка при удалении исходного файла модели {input_model_path}: {e}")
        
        return True

    except Exception as e:
        print(f"Произошла ошибка во время оптимизации модели: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_gpu_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Оптимизирует файл натренированной модели XTTS, удаляя ненужную для инференса информацию. "
                    "Оптимизированная модель сохраняется как 'optimized_model.pth' в той же директории, что и входная модель."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Путь к файлу натренированной модели XTTS (.pth) для оптимизации.",
    )
    parser.add_argument(
        "--delete_original",
        action="store_true",
        help="Если установлен, исходный неоптимизированный файл модели будет удален после успешной оптимизации. "
             "Не имеет эффекта, если входная модель уже называется 'optimized_model.pth'.",
    )

    args = parser.parse_args()

    print(f"Запуск оптимизации модели для: {args.input_file}")
    success = optimize_model_standalone(args.input_file, args.delete_original)

    if success:
        print("Процесс оптимизации успешно завершен.")
        sys.exit(0)
    else:
        print("Процесс оптимизации не удался.")
        sys.exit(1)

if __name__ == "__main__":
    main()
