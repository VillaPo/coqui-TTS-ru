import torch
# import wandb
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from torch.utils.data import DataLoader, Dataset # Добавили Dataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import soundfile as sf # Для загрузки аудио
import numpy as np # Для работы с аудио
import random # Для случайной обрезки в обучении

from tqdm import tqdm
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig

from dataclasses import dataclass, field
from typing import Optional
import os
import datetime
from transformers import HfArgumentParser

# --- Начало добавленного класса DVAEDataset ---
class DVAEDataset(Dataset):
    def __init__(self, samples, sample_rate, is_eval, max_wav_len, min_wav_len=1024): # min_wav_len увеличен
        super().__init__()
        self.samples = samples
        self.sample_rate = sample_rate
        self.is_eval = is_eval
        self.max_wav_len = max_wav_len
        self.min_wav_len = min_wav_len

        original_sample_count = len(samples)
        self.valid_samples = []
        for sample_item in samples:
            audio_file = sample_item[1] # audio_file path
            if not os.path.exists(audio_file):
                continue
            try:
                info = sf.info(audio_file)
                if info.samplerate != self.sample_rate:
                    continue
                if info.frames < self.min_wav_len:
                    continue
                # Опционально: пропустить слишком длинные файлы, чтобы избежать OOM при sf.info или начальной загрузке
                # if info.frames > self.max_wav_len * 10: # Например, файлы длиннее 150 секунд
                #     print(f"Warning: Audio file {audio_file} is extremely long ({info.frames} frames), skipping.")
                #     continue
                self.valid_samples.append(sample_item)
            except Exception: # Более общее исключение для sf.info
                continue
        
        self.samples = self.valid_samples
        if original_sample_count != len(self.samples):
            print(f"Отфильтровано {original_sample_count - len(self.samples)} семплов из-за проблем с путем, SR или длиной.")
        if not self.samples:
            raise ValueError("Не найдено корректных аудио семплов после фильтрации. Проверьте ваш датасет и параметры.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample[1]
        
        wav, sr = sf.read(audio_path, dtype='float32')
        # Проверка sample_rate уже сделана в __init__ при фильтрации

        if wav.ndim > 1: 
            wav = wav[:, 0]
        
        wav = torch.from_numpy(wav).float()

        if wav.size(0) >= self.max_wav_len:
            start = random.randint(0, wav.size(0) - self.max_wav_len) if not self.is_eval else 0
            wav = wav[start : start + self.max_wav_len]
        else: 
            wav = torch.nn.functional.pad(wav, (0, self.max_wav_len - wav.size(0)))
        
        return {"wav": wav, "text": sample[0], "speaker_name": sample[2], "audio_path": audio_path}

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        wavs = [item["wav"] for item in batch]
        # texts = [item["text"] for item in batch] # DVAE не использует текст напрямую
        # speaker_names = [item["speaker_name"] for item in batch]
        # audio_paths = [item["audio_path"] for item in batch] # Для отладки

        wav_batch = torch.stack(wavs)
        # Возвращаем только то, что нужно для DVAE и format_batch
        return {"wav": wav_batch}
# --- Конец добавленного класса DVAEDataset ---

@dataclass
class DVAETrainerArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    output_path: str = field(
        metadata={"help": "Path to the root of your Coqui TTS project (e.g., /path/to/coqui-TTS-ru). Used to locate 'run/training/XTTS_v2.0_original_model_files/' and save fine-tuned DVAE."}
    )
    train_csv_path: str = field(
        # Изменено описание: это должен быть полный metadata.txt
        metadata={"help": "Path to the full metadata file (e.g., metadata.txt) which will be split for training and evaluation."},
    )
    eval_csv_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to eval metadata file"},
    )
    language: Optional[str] = field(
        default="en",
        metadata={"help": "The language you want to train (language in your dataset)"},
    )
    lr: Optional[float] = field(
        default=5e-6,
        metadata={"help": "Learning rate"},
    )
    num_epochs: Optional[int] = field(
        default=5,
    )
    batch_size: Optional[int] = field(
        default=512,
        # Изменено значение по умолчанию на более безопасное для 16GB VRAM
        default=32, 
        metadata={"help": "Batch size for DVAE training. Default: 32. Adjust based on your VRAM."},
    )



def train(output_path, train_csv_path, eval_csv_path="", language="en", lr=5e-6, num_epochs=5, batch_size=512):
    # Уточненные пути к файлам pretrained моделей
    original_model_files_dir = os.path.join(output_path, 'run', 'training', 'XTTS_v2.0_original_model_files')
    if not os.path.isdir(original_model_files_dir):
        print(f"Ошибка: Директория с оригинальными файлами модели не найдена: {original_model_files_dir}")
        print(f"Убедитесь, что `output_path` ({output_path}) является корневой директорией проекта и содержит 'run/training/XTTS_v2.0_original_model_files/'.")
        return

    dvae_pretrained = os.path.join(original_model_files_dir, 'dvae.pth')
    mel_norm_file = os.path.join(original_model_files_dir, 'mel_stats.pth')

    if not os.path.isfile(dvae_pretrained):
        print(f"Ошибка: Файл pretrained DVAE не найден: {dvae_pretrained}")
        return
    if not os.path.isfile(mel_norm_file):
        print(f"Ошибка: Файл mel_stats.pth не найден: {mel_norm_file}")
        return

    now = datetime.datetime.now()
    now_without_ms = now.replace(microsecond=0)
    # CHECKPOINTS_OUT_PATH = os.path.join(output_path, f"DVAE_checkpoint_{now_without_ms}/")
    # os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # --- Разделение metadata.txt на train/eval без перемешивания ---
    all_samples_meta = []
    with open(train_csv_path, 'r', encoding='utf-8') as f_meta:
        all_samples_meta = [line.strip() for line in f_meta if line.strip()]

    if not all_samples_meta:
        raise ValueError(f"Файл метаданных {train_csv_path} пуст или не может быть прочитан.")

    split_idx = int(len(all_samples_meta) * 0.9)
    train_meta_lines = all_samples_meta[:split_idx]
    eval_meta_lines = all_samples_meta[split_idx:]

    dataset_dir = os.path.dirname(train_csv_path)
    temp_train_meta_file = os.path.join(dataset_dir, "metadata_train_temp_dvae.txt")
    temp_eval_meta_file = os.path.join(dataset_dir, "metadata_eval_temp_dvae.txt")
    # --- Конец разделения ---

    config_dataset = BaseDatasetConfig(
        formatter="ljspeech", # Изменен форматтер
        dataset_name="custom_dvae_split",
        path=dataset_dir, # Путь к директории с метафайлами и аудио
        meta_file_train=os.path.basename(temp_train_meta_file),
        meta_file_val=os.path.basename(temp_eval_meta_file) if eval_meta_lines else None,
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]
    GRAD_CLIP_NORM = 0.5
    LEARNING_RATE = lr

    dvae = DiscreteVAE(
                channels=80,
                normalization=None,
                positional_dims=1,
                num_tokens=1024,
                codebook_dim=512,
                hidden_dim=512,
                num_resnet_blocks=3,
                kernel_size=3,
                num_layers=2,
                use_transposed_convs=False,
            )

    dvae.load_state_dict(torch.load(dvae_pretrained), strict=False)
    dvae.cuda()
    opt = Adam(dvae.parameters(), lr = LEARNING_RATE)
    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
                mel_norm_file=mel_norm_file, sampling_rate=22050
            ).cuda()

    try:
        with open(temp_train_meta_file, 'w', encoding='utf-8') as f:
            for line in train_meta_lines:
                f.write(line + '\n')

        if eval_meta_lines:
            with open(temp_eval_meta_file, 'w', encoding='utf-8') as f:
                for line in eval_meta_lines:
                    f.write(line + '\n')
        elif os.path.exists(temp_eval_meta_file): # Ensure it's empty if no eval lines
             open(temp_eval_meta_file, 'w', encoding='utf-8').close()


        train_samples, eval_samples = load_tts_samples(
                DATASETS_CONFIG_LIST,
                eval_split=bool(eval_meta_lines), # eval_split=True только если есть eval данные
                eval_split_max_size=256, # Это значение будет применено, если meta_file_val пуст
                eval_split_size=0.01,    # Аналогично
            )

        eval_dataset = DVAEDataset(eval_samples, 22050, True, max_wav_len=15*22050) if eval_samples else None
        train_dataset = DVAEDataset(train_samples, 22050, False, max_wav_len=15*22050)
    finally:
        if os.path.exists(temp_train_meta_file):
            os.remove(temp_train_meta_file)
        if os.path.exists(temp_eval_meta_file):
            os.remove(temp_eval_meta_file)


    eval_data_loader = DataLoader(
                        eval_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=eval_dataset.collate_fn,
                        num_workers=0,
                        pin_memory=False,
                    ) if eval_dataset else None

    train_data_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=False,
                    )

    torch.set_grad_enabled(True)
    dvae.train()

    # wandb.init(project = 'train_dvae')
    # wandb.watch(dvae)

    def to_cuda(x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.contiguous()
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
        return x

    @torch.no_grad()
    def format_batch(batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]

        try:
            batch['mel'] = torch_mel_spectrogram_dvae(batch['wav'])
            # if the mel spectogram is not divisible by 4 then input.shape != output.shape 
            # for dvae
            remainder = batch['mel'].shape[-1] % 4
            if remainder:
                batch['mel'] = batch['mel'][:, :, :-remainder]
        except NotImplementedError:
            pass
        return batch

    best_loss = 1e6

    for i in range(num_epochs):
        dvae.train()
        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch)
            recon_loss, commitment_loss, out = dvae(batch['mel'])
            recon_loss = recon_loss.mean()
            total_loss = recon_loss + commitment_loss
            # print(f"commitment_loss shape: {commitment_loss.shape}")
            # print(f"recon_loss shape: {recon_loss.shape}")
            # print(f"total_loss shape: {total_loss.shape}")
            total_loss.backward()
            clip_grad_norm_(dvae.parameters(), GRAD_CLIP_NORM)
            opt.step()

            log = {'epoch': i,
                'cur_step': cur_step,
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'commit_loss': commitment_loss.item()}
            print(f"epoch: {i}", print(f"step: {cur_step}"), f'loss - {total_loss.item()}', f'recon_loss - {recon_loss.item()}', f'commit_loss - {commitment_loss.item()}')
            # wandb.log(log)
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            if eval_data_loader:
                dvae.eval()
                eval_loss_sum = 0
                num_eval_batches = 0
                for cur_step, batch in enumerate(eval_data_loader):
                    if batch is None: continue # Пропуск, если collate_fn вернул None
                    batch = format_batch(batch)
                    recon_loss, commitment_loss, out = dvae(batch['mel'])
                    recon_loss = recon_loss.mean()
                    eval_loss_sum += (recon_loss + commitment_loss).item()
                    num_eval_batches += 1
                
                if num_eval_batches > 0:
                    eval_loss = eval_loss_sum / num_eval_batches
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        torch.save(dvae.state_dict(), dvae_pretrained)
                    print(f"#######################################\nepoch: {i}\tEVAL loss: {eval_loss}\n#######################################")
                else:
                    print(f"#######################################\nepoch: {i}\tНет данных для оценки.\n#######################################")
                    # Если нет данных для оценки, можно сохранить модель по последней эпохе или по train loss
                    # torch.save(dvae.state_dict(), dvae_pretrained) # Раскомментируйте, если хотите сохранять в этом случае

    print(f'Checkpoint saved at {dvae_pretrained}')
   

if __name__ == "__main__":
    parser = HfArgumentParser(DVAETrainerArgs)

    args = parser.parse_args_into_dataclasses()[0]

    trainer_out_path = train(
        language=args.language,
        train_csv_path=args.train_csv_path,
        eval_csv_path=args.eval_csv_path,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )