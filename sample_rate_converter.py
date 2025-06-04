import os
import argparse
import soundfile as sf
import librosa
import numpy as np

def resample_wav(input_path, output_path, target_sr, original_sr=None, force_original_sr=False):
    """
    Ресемплирует один WAV файл.

    Args:
        input_path (str): Путь к исходному WAV файлу.
        output_path (str): Путь для сохранения ресемплированного WAV файла.
        target_sr (int): Целевая частота дискретизации.
        original_sr (int, optional): Ожидаемая исходная частота дискретизации.
                                     Если None, используется фактическая частота файла.
        force_original_sr (bool): Если True и original_sr задан, аудио будет обработано
                                  как будто оно имеет частоту original_sr, даже если
                                  метаданные файла говорят о другом. Используйте с осторожностью.
    """
    try:
        if force_original_sr and original_sr is not None:
            # Читаем как сырые данные, если нужно форсировать original_sr
            # Это менее надежно, если формат файла сложный, но необходимо для librosa.load с sr=None
            # и последующего librosa.resample с orig_sr=original_sr
            # Однако, soundfile.read уже дает нам данные и фактическую sr.
            # Лучше использовать фактическую sr, если force_original_sr не установлен.
            # Если force_original_sr, мы доверяем пользователю, что он знает исходную SR.
            data, sr_actual = librosa.load(input_path, sr=None, mono=True) # librosa.load для форсирования
            sr_to_resample_from = original_sr
            if sr_actual != original_sr:
                print(f"    Предупреждение: Файл {input_path} имеет фактическую SR={sr_actual}, "
                      f"но будет ресемплирован с указанной original_sr={original_sr} из-за force_original_sr=True.")
        else:
            data, sr_actual = sf.read(input_path, dtype='float32')
            # Убедимся, что аудио монофоническое для librosa
            if data.ndim > 1 and data.shape[1] > 1:
                # Просто берем первый канал, если стерео
                print(f"    Предупреждение: Файл {input_path} - стерео. Используется только первый канал.")
                data = data[:, 0]
            elif data.ndim > 1 and data.shape[1] == 1:
                data = data.flatten()


            sr_to_resample_from = sr_actual
            if original_sr is not None and sr_actual != original_sr:
                print(f"    Предупреждение: Файл {input_path} имеет фактическую SR={sr_actual}, "
                      f"ожидалось {original_sr}. Ресемплируем с фактической SR.")


        if sr_to_resample_from == target_sr:
            print(f"    Частота дискретизации файла {input_path} уже {target_sr} Гц. Копирование...")
            sf.write(output_path, data, sr_to_resample_from)
        else:
            print(f"    Ресемплинг {input_path} с {sr_to_resample_from} Гц до {target_sr} Гц...")
            resampled_data = librosa.resample(y=data, orig_sr=sr_to_resample_from, target_sr=target_sr)
            sf.write(output_path, resampled_data, target_sr)
        print(f"    Сохранено в: {output_path}")
        return True
    except Exception as e:
        print(f"    Ошибка при обработке файла {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir, target_sr, original_sr=None, force_original_sr=False):
    """
    Ресемплирует все WAV файлы в указанной директории.

    Args:
        input_dir (str): Директория с исходными WAV файлами.
        output_dir (str): Директория для сохранения ресемплированных WAV файлов.
        target_sr (int): Целевая частота дискретизации.
        original_sr (int, optional): Ожидаемая исходная частота дискретизации.
        force_original_sr (bool): Форсировать использование original_sr.
    """
    if not os.path.isdir(input_dir):
        print(f"Ошибка: Исходная директория '{input_dir}' не найдена.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана выходная директория: {output_dir}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"\nОбработка файла: {filename}")
            if resample_wav(input_path, output_path, target_sr, original_sr, force_original_sr):
                processed_count +=1
            else:
                error_count +=1
        else:
            skipped_count +=1
            # print(f"Пропуск не-WAV файла: {filename}")


    print(f"\n--- Завершено ---")
    print(f"Обработано файлов: {processed_count}")
    print(f"Пропущено (не WAV): {skipped_count}")
    print(f"Файлов с ошибками: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ресемплирует WAV файлы в директории.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Путь к директории с исходными WAV файлами.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Путь к директории для сохранения ресемплированных WAV файлов.")
    parser.add_argument("--target_sr", type=int, default=22050,
                        help="Целевая частота дискретизации (по умолчанию: 22050).")
    parser.add_argument("--original_sr", type=int, default=24000,
                        help="Ожидаемая исходная частота дискретизации. "
                             "Если фактическая отличается, будет выдано предупреждение, "
                             "и использована фактическая, если не указан --force_original_sr "
                             "(по умолчанию: 24000). "
                             "Установите в 0, чтобы всегда использовать фактическую SR файла без предупреждений.")
    parser.add_argument("--force_original_sr", action='store_true',
                        help="Если указано, принудительно использовать значение --original_sr "
                             "для ресемплинга, даже если метаданные файла указывают другую SR. "
                             "Используйте с осторожностью.")


    args = parser.parse_args()

    # Если original_sr установлен в 0, это означает, что мы не хотим проверять
    # и всегда будем использовать фактическую SR файла.
    original_sr_to_pass = args.original_sr if args.original_sr != 0 else None

    process_directory(args.input_dir, args.output_dir, args.target_sr, original_sr_to_pass, args.force_original_sr)
