<a href="https://colab.research.google.com/github/MahmoudAshraf97/whisper-diarization/blob/main/Whisper_Transcription_%2B_NeMo_Diarization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Installing Dependencies
https://github.com/MahmoudAshraf97/whisper-diarization
According to github.com/openai/whisper/discussions/55 these are valid values for device: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mps, xla, lazy, vulkan, meta, hpu, privateuseone. Not sure if this list is complete or if there are even more options. 

```python
!pip install nemo_toolkit[asr]==1.17.0
!pip install faster-whisper==0.5.1
!pip install git+https://github.com/m-bain/whisperX.git@4cb167a225c0ebaea127fd6049abfaa3af9f8bb4
!pip install git+https://github.com/facebookresearch/demucs#egg=demucs
!pip install transformers>=4.26.1
!pip install deepmultilingualpunctuation
!pip install wget
```


```python
import os
import wget
from omegaconf import OmegaConf
import json
import shutil
from faster_whisper import WhisperModel
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
```

    /Users/calejohnstone/opt/anaconda3/lib/python3.9/site-packages/pkg_resources/__init__.py:123: PkgResourcesDeprecationWarning: 4.0.0-unsupported is an invalid version and will not be supported in a future release
      warnings.warn(
    [NeMo W 2023-06-06 16:31:45 optimizers:54] Apex was not found. Using the lamb or fused_adam optimizer will error out.
    [NeMo W 2023-06-06 16:31:48 experimental:27] Module <class 'nemo.collections.asr.modules.audio_modules.SpectrogramToMultichannelFeatures'> is experimental, not ready for production and is not fully supported. Use at your own risk.


# Helper Functions


```python
punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
wav2vec2_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "ja",
    "zh",
    "uk",
    "pt",
    "ar",
    "ru",
    "pl",
    "hu",
    "fi",
    "fa",
    "el",
    "tr",
]


def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting or telephonic based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"

    config.num_workers =0  # Workaround for multiprocessing hanging with ipython issue

    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )

    return config


def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (
            int(wrd_dict["start"] * 1000),
            int(wrd_dict["end"] * 1000),
            wrd_dict["text"],
        )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
        )
    return wrd_spk_mapping


sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list)
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence=50
):
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk

    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"]
        f.write(f"\n\n{sp}: {text}")


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_srt(transcript, file):
    """
    Write a transcript to a file in SRT format.

    """
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))
```

# Options


```python
# Name of the audio file
audio_path = 'SydS_IOM_058_Vittorio.wav'

# Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = False

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large')
whisper_model_name = 'small.en'
```

# Processing

## Separating music from speech using Demucs

---

By isolating the vocals from the rest of the audio, it becomes easier to identify and track individual speakers based on the spectral and temporal characteristics of their speech signals. Source separation is just one of many techniques that can be used as a preprocessing step to help improve the accuracy and reliability of the overall diarization process.


```python
if enable_stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file."
        )
        vocal_target = audio_path
    else:
        vocal_target = os.path.join(
            "temp_outputs", "htdemucs", os.path.basename(args.audio[:-4]), "vocals.wav"
        )
else:
    vocal_target = audio_path
```

## Transcriping audio using Whisper and realligning timestamps using Wav2Vec2
---
This code uses two different open-source models to transcribe speech and perform forced alignment on the resulting transcription.

The first model is called OpenAI Whisper, which is a speech recognition model that can transcribe speech with high accuracy. The code loads the whisper model and uses it to transcribe the vocal_target file.

The output of the transcription process is a set of text segments with corresponding timestamps indicating when each segment was spoken.



```python
# Run on GPU with FP16
whisper_model = WhisperModel(whisper_model_name)
#whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = whisper_model.transcribe(
    vocal_target, beam_size=1, word_timestamps=True
)
whisper_results = []
for segment in segments:
    whisper_results.append(segment._asdict())
# clear gpu vram
del whisper_model
torch.cuda.empty_cache()
```

    [2023-06-06 23:24:46.072] [ctranslate2] [thread 561346] [warning] The compute type inferred from the saved model is float16, but the target device or backend do not support efficient float16 computation. The model weights have been automatically converted to use the float32 compute type instead.


## Aligning the transcription with the original audio using Wav2Vec2
---
The second model used is called wav2vec2, which is a large-scale neural network that is designed to learn representations of speech that are useful for a variety of speech processing tasks, including speech recognition and alignment.

The code loads the wav2vec2 alignment model and uses it to align the transcription segments with the original audio signal contained in the vocal_target file. This process involves finding the exact timestamps in the audio signal where each segment was spoken and aligning the text accordingly.

By combining the outputs of the two models, the code produces a fully aligned transcription of the speech contained in the vocal_target file. This aligned transcription can be useful for a variety of speech processing tasks, such as speaker diarization, sentiment analysis, and language identification.

If there's no Wav2Vec2 model available for your language, word timestamps generated by whisper will be used instead.


```python
if info.language in wav2vec2_langs:
    device = "cpu"
    alignment_model, metadata = whisperx.load_align_model(
        language_code=info.language, device=device
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, device
    )
    word_timestamps = result_aligned["word_segments"]
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"text": word[2], "start": word[0], "end": word[1]})
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /var/folders/5t/7hqsqpqd66x83wsqwsy_rytr0000gs/T/ipykernel_32000/1603811820.py in <module>
          4         language_code=info.language, device=device
          5     )
    ----> 6     result_aligned = whisperx.align(
          7         whisper_results, alignment_model, metadata, vocal_target, device
          8     )


    ~/opt/anaconda3/lib/python3.9/site-packages/whisperx/alignment.py in align(transcript, model, align_model_metadata, audio, device, extend_duration, start_from_previous, interpolate_method)
        222             with torch.inference_mode():
        223                 if model_type == "torchaudio":
    --> 224                     emissions, _ = model(waveform_segment.to(device))
        225                 elif model_type == "huggingface":
        226                     emissions = model(waveform_segment.to(device)).logits


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1108         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1109                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1110             return forward_call(*input, **kwargs)
       1111         # Do not call functions when jit is used
       1112         full_backward_hooks, non_full_backward_hooks = [], []


    ~/opt/anaconda3/lib/python3.9/site-packages/torchaudio/models/wav2vec2/model.py in forward(self, waveforms, lengths)
        111                 It indicates the valid length in time axis of the output Tensor.
        112         """
    --> 113         x, lengths = self.feature_extractor(waveforms, lengths)
        114         x = self.encoder(x, lengths)
        115         if self.aux is not None:


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1108         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1109                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1110             return forward_call(*input, **kwargs)
       1111         # Do not call functions when jit is used
       1112         full_backward_hooks, non_full_backward_hooks = [], []


    ~/opt/anaconda3/lib/python3.9/site-packages/torchaudio/models/wav2vec2/components.py in forward(self, x, length)
        107         x = x.unsqueeze(1)  # (batch, channel==1, frame)
        108         for layer in self.conv_layers:
    --> 109             x, length = layer(x, length)  # (batch, feature, frame)
        110         x = x.transpose(1, 2)  # (batch, frame, feature)
        111         return x, length


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1108         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1109                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1110             return forward_call(*input, **kwargs)
       1111         # Do not call functions when jit is used
       1112         full_backward_hooks, non_full_backward_hooks = [], []


    ~/opt/anaconda3/lib/python3.9/site-packages/torchaudio/models/wav2vec2/components.py in forward(self, x, length)
         56             Optional[Tensor]: Shape ``[batch, ]``.
         57         """
    ---> 58         x = self.conv(x)
         59         if self.layer_norm is not None:
         60             x = self.layer_norm(x)


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1108         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1109                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1110             return forward_call(*input, **kwargs)
       1111         # Do not call functions when jit is used
       1112         full_backward_hooks, non_full_backward_hooks = [], []


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/conv.py in forward(self, input)
        300 
        301     def forward(self, input: Tensor) -> Tensor:
    --> 302         return self._conv_forward(input, self.weight, self.bias)
        303 
        304 


    ~/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
        296                             weight, bias, self.stride,
        297                             _single(0), self.dilation, self.groups)
    --> 298         return F.conv1d(input, weight, bias, self.stride,
        299                         self.padding, self.dilation, self.groups)
        300 


    RuntimeError: Calculated padded input size per channel: (1). Kernel size: (2). Kernel size can't be greater than actual input size


## Convert audio to mono for NeMo combatibility


```python
signal, sample_rate = librosa.load(vocal_target, sr=None)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
soundfile.write(os.path.join(temp_path, "mono_file.wav"), signal, sample_rate, "PCM_24")
```

## Speaker Diarization using NeMo MSDD Model
---
This code uses a model called Nvidia NeMo MSDD (Multi-scale Diarization Decoder) to perform speaker diarization on an audio signal. Speaker diarization is the process of separating an audio signal into different segments based on who is speaking at any given time.


```python
# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cpu")
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()
```

    [NeMo I 2023-06-06 17:26:38 msdd_models:1092] Loading pretrained diar_msdd_telephonic model from NGC
    [NeMo I 2023-06-06 17:26:38 cloud:58] Found existing object /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo.
    [NeMo I 2023-06-06 17:26:38 cloud:64] Re-using file from: /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo
    [NeMo I 2023-06-06 17:26:38 common:913] Instantiating model from pre-trained checkpoint


    [NeMo W 2023-06-06 17:26:46 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
        Train config : 
        manifest_filepath: null
        emb_dir: null
        sample_rate: 16000
        num_spks: 2
        soft_label_thres: 0.5
        labels: null
        batch_size: 15
        emb_batch_size: 0
        shuffle: true
        
    [NeMo W 2023-06-06 17:26:46 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s). 
        Validation config : 
        manifest_filepath: null
        emb_dir: null
        sample_rate: 16000
        num_spks: 2
        soft_label_thres: 0.5
        labels: null
        batch_size: 15
        emb_batch_size: 0
        shuffle: false
        
    [NeMo W 2023-06-06 17:26:46 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
        Test config : 
        manifest_filepath: null
        emb_dir: null
        sample_rate: 16000
        num_spks: 2
        soft_label_thres: 0.5
        labels: null
        batch_size: 15
        emb_batch_size: 0
        shuffle: false
        seq_eval_mode: false
        


    [NeMo I 2023-06-06 17:26:46 features:287] PADDING: 16
    [NeMo I 2023-06-06 17:26:47 features:287] PADDING: 16
    [NeMo I 2023-06-06 17:26:50 save_restore_connector:247] Model EncDecDiarLabelModel was successfully restored from /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo.
    [NeMo I 2023-06-06 17:26:50 features:287] PADDING: 16
    [NeMo I 2023-06-06 17:26:53 clustering_diarizer:127] Loading pretrained vad_multilingual_marblenet model from NGC
    [NeMo I 2023-06-06 17:26:53 cloud:58] Found existing object /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo.
    [NeMo I 2023-06-06 17:26:53 cloud:64] Re-using file from: /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo
    [NeMo I 2023-06-06 17:26:53 common:913] Instantiating model from pre-trained checkpoint


    [NeMo W 2023-06-06 17:26:54 modelPT:161] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
        Train config : 
        manifest_filepath: /manifests/ami_train_0.63.json,/manifests/freesound_background_train.json,/manifests/freesound_laughter_train.json,/manifests/fisher_2004_background.json,/manifests/fisher_2004_speech_sampled.json,/manifests/google_train_manifest.json,/manifests/icsi_all_0.63.json,/manifests/musan_freesound_train.json,/manifests/musan_music_train.json,/manifests/musan_soundbible_train.json,/manifests/mandarin_train_sample.json,/manifests/german_train_sample.json,/manifests/spanish_train_sample.json,/manifests/french_train_sample.json,/manifests/russian_train_sample.json
        sample_rate: 16000
        labels:
        - background
        - speech
        batch_size: 256
        shuffle: true
        is_tarred: false
        tarred_audio_filepaths: null
        tarred_shard_strategy: scatter
        augmentor:
          shift:
            prob: 0.5
            min_shift_ms: -10.0
            max_shift_ms: 10.0
          white_noise:
            prob: 0.5
            min_level: -90
            max_level: -46
            norm: true
          noise:
            prob: 0.5
            manifest_path: /manifests/noise_0_1_musan_fs.json
            min_snr_db: 0
            max_snr_db: 30
            max_gain_db: 300.0
            norm: true
          gain:
            prob: 0.5
            min_gain_dbfs: -10.0
            max_gain_dbfs: 10.0
            norm: true
        num_workers: 16
        pin_memory: true
        
    [NeMo W 2023-06-06 17:26:54 modelPT:168] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s). 
        Validation config : 
        manifest_filepath: /manifests/ami_dev_0.63.json,/manifests/freesound_background_dev.json,/manifests/freesound_laughter_dev.json,/manifests/ch120_moved_0.63.json,/manifests/fisher_2005_500_speech_sampled.json,/manifests/google_dev_manifest.json,/manifests/musan_music_dev.json,/manifests/mandarin_dev.json,/manifests/german_dev.json,/manifests/spanish_dev.json,/manifests/french_dev.json,/manifests/russian_dev.json
        sample_rate: 16000
        labels:
        - background
        - speech
        batch_size: 256
        shuffle: false
        val_loss_idx: 0
        num_workers: 16
        pin_memory: true
        
    [NeMo W 2023-06-06 17:26:54 modelPT:174] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
        Test config : 
        manifest_filepath: null
        sample_rate: 16000
        labels:
        - background
        - speech
        batch_size: 128
        shuffle: false
        test_loss_idx: 0
        


    [NeMo I 2023-06-06 17:26:54 features:287] PADDING: 16
    [NeMo I 2023-06-06 17:26:55 save_restore_connector:247] Model EncDecClassificationModel was successfully restored from /Users/calejohnstone/.cache/torch/NeMo/NeMo_1.17.0/vad_multilingual_marblenet/670f425c7f186060b7a7268ba6dfacb2/vad_multilingual_marblenet.nemo.
    [NeMo I 2023-06-06 17:26:55 msdd_models:864] Multiscale Weights: [1, 1, 1, 1, 1]
    [NeMo I 2023-06-06 17:26:55 msdd_models:865] Clustering Parameters: {
            "oracle_num_speakers": false,
            "max_num_speakers": 8,
            "enhanced_count_thres": 80,
            "max_rp_threshold": 0.25,
            "sparse_search_volume": 30,
            "maj_vote_spk_count": false
        }


    [NeMo W 2023-06-06 17:26:55 clustering_diarizer:411] Deleting previous clustering diarizer outputs.


    [NeMo I 2023-06-06 17:26:55 speaker_utils:93] Number of files to diarize: 1
    [NeMo I 2023-06-06 17:26:55 clustering_diarizer:309] Split long audio file to avoid CUDA memory issue


    splitting manifest: 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:09<00:00,  9.67s/it]

    [NeMo I 2023-06-06 17:27:05 vad_utils:101] The prepared manifest file exists. Overwriting!
    [NeMo I 2023-06-06 17:27:05 classification_models:263] Perform streaming frame-level VAD


    


    [NeMo I 2023-06-06 17:27:05 collections:298] Filtered duration for loading collection is 0.000000.
    [NeMo I 2023-06-06 17:27:05 collections:301] Dataset loaded with 1 items, total duration of  0.01 hours.
    [NeMo I 2023-06-06 17:27:05 collections:303] # 1 files loaded accounting to # 1 labels


    vad:   0%|                                                                                                      | 0/1 [00:00<?, ?it/s][NeMo W 2023-06-06 17:27:13 nemo_logging:349] /Users/calejohnstone/opt/anaconda3/lib/python3.9/site-packages/torch/autocast_mode.py:162: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
          warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
        
    vad: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:15<00:00, 15.63s/it]

    [NeMo I 2023-06-06 17:27:21 clustering_diarizer:250] Generating predictions with overlapping input segments


    
                                                                                                                                          

    [NeMo I 2023-06-06 17:27:24 clustering_diarizer:262] Converting frame level prediction to speech/no-speech segment in start and end times format.


    creating speech segments: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.29it/s]

    [NeMo I 2023-06-06 17:27:25 clustering_diarizer:287] Subsegmentation for embedding extraction: scale0, /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale0.json
    [NeMo I 2023-06-06 17:27:25 clustering_diarizer:343] Extracting embeddings for Diarization
    [NeMo I 2023-06-06 17:27:25 collections:298] Filtered duration for loading collection is 0.000000.
    [NeMo I 2023-06-06 17:27:25 collections:301] Dataset loaded with 22 items, total duration of  0.00 hours.
    [NeMo I 2023-06-06 17:27:25 collections:303] # 22 files loaded accounting to # 1 labels


    
    [1/5] extract embeddings: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:18<00:00, 18.01s/it]

    [NeMo I 2023-06-06 17:27:43 clustering_diarizer:389] Saved embedding files to /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings
    [NeMo I 2023-06-06 17:27:43 clustering_diarizer:287] Subsegmentation for embedding extraction: scale1, /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale1.json
    [NeMo I 2023-06-06 17:27:43 clustering_diarizer:343] Extracting embeddings for Diarization
    [NeMo I 2023-06-06 17:27:43 collections:298] Filtered duration for loading collection is 0.000000.


    


    [NeMo I 2023-06-06 17:27:43 collections:301] Dataset loaded with 25 items, total duration of  0.00 hours.
    [NeMo I 2023-06-06 17:27:43 collections:303] # 25 files loaded accounting to # 1 labels


    [2/5] extract embeddings: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:15<00:00, 15.99s/it]

    [NeMo I 2023-06-06 17:27:59 clustering_diarizer:389] Saved embedding files to /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings
    [NeMo I 2023-06-06 17:27:59 clustering_diarizer:287] Subsegmentation for embedding extraction: scale2, /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale2.json
    [NeMo I 2023-06-06 17:27:59 clustering_diarizer:343] Extracting embeddings for Diarization
    [NeMo I 2023-06-06 17:27:59 collections:298] Filtered duration for loading collection is 0.000000.
    [NeMo I 2023-06-06 17:27:59 collections:301] Dataset loaded with 28 items, total duration of  0.00 hours.
    [NeMo I 2023-06-06 17:27:59 collections:303] # 28 files loaded accounting to # 1 labels


    
    [3/5] extract embeddings: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:16<00:00, 16.41s/it]

    [NeMo I 2023-06-06 17:28:16 clustering_diarizer:389] Saved embedding files to /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings
    [NeMo I 2023-06-06 17:28:16 clustering_diarizer:287] Subsegmentation for embedding extraction: scale3, /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale3.json
    [NeMo I 2023-06-06 17:28:16 clustering_diarizer:343] Extracting embeddings for Diarization
    [NeMo I 2023-06-06 17:28:16 collections:298] Filtered duration for loading collection is 0.000000.


    


    [NeMo I 2023-06-06 17:28:16 collections:301] Dataset loaded with 32 items, total duration of  0.00 hours.
    [NeMo I 2023-06-06 17:28:16 collections:303] # 32 files loaded accounting to # 1 labels


    [4/5] extract embeddings: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:14<00:00, 14.66s/it]

    [NeMo I 2023-06-06 17:28:31 clustering_diarizer:389] Saved embedding files to /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings
    [NeMo I 2023-06-06 17:28:31 clustering_diarizer:287] Subsegmentation for embedding extraction: scale4, /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale4.json
    [NeMo I 2023-06-06 17:28:31 clustering_diarizer:343] Extracting embeddings for Diarization


    


    [NeMo I 2023-06-06 17:28:31 collections:298] Filtered duration for loading collection is 0.000000.
    [NeMo I 2023-06-06 17:28:31 collections:301] Dataset loaded with 47 items, total duration of  0.01 hours.
    [NeMo I 2023-06-06 17:28:31 collections:303] # 47 files loaded accounting to # 1 labels


    [5/5] extract embeddings: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.95s/it]

    [NeMo I 2023-06-06 17:28:51 clustering_diarizer:389] Saved embedding files to /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings


    
    [NeMo W 2023-06-06 17:28:51 speaker_utils:464] cuda=False, using CPU for eigen decomposition. This might slow down the clustering process.
    clustering: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.88s/it]

    [NeMo I 2023-06-06 17:28:56 clustering_diarizer:464] Outputs are saved in /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs directory


    
    [NeMo W 2023-06-06 17:28:56 der:106] Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate


    [NeMo I 2023-06-06 17:28:56 msdd_models:960] Loading embedding pickle file of scale:0 at /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings/subsegments_scale0_embeddings.pkl
    [NeMo I 2023-06-06 17:28:56 msdd_models:960] Loading embedding pickle file of scale:1 at /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings/subsegments_scale1_embeddings.pkl
    [NeMo I 2023-06-06 17:28:56 msdd_models:960] Loading embedding pickle file of scale:2 at /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings/subsegments_scale2_embeddings.pkl
    [NeMo I 2023-06-06 17:28:56 msdd_models:960] Loading embedding pickle file of scale:3 at /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings/subsegments_scale3_embeddings.pkl
    [NeMo I 2023-06-06 17:28:56 msdd_models:960] Loading embedding pickle file of scale:4 at /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/embeddings/subsegments_scale4_embeddings.pkl
    [NeMo I 2023-06-06 17:28:56 msdd_models:938] Loading cluster label file from /Users/calejohnstone/Documents/wk/work2023/whisperDiarize/whisper-diarization-main/temp_outputs/speaker_outputs/subsegments_scale4_cluster.label
    [NeMo I 2023-06-06 17:28:56 collections:612] Filtered duration for loading collection is 0.000000.
    [NeMo I 2023-06-06 17:28:56 collections:615] Total 6 session files loaded accounting to # 6 audio clips


    100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.34it/s]

    [NeMo I 2023-06-06 17:28:57 msdd_models:1403]      [Threshold: 0.7000] [use_clus_as_main=False] [diar_window=50]
    [NeMo I 2023-06-06 17:28:57 speaker_utils:93] Number of files to diarize: 1
    [NeMo I 2023-06-06 17:28:57 speaker_utils:93] Number of files to diarize: 1


    
    [NeMo W 2023-06-06 17:28:57 der:106] Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate


    [NeMo I 2023-06-06 17:28:57 speaker_utils:93] Number of files to diarize: 1


    [NeMo W 2023-06-06 17:28:57 der:106] Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate


    [NeMo I 2023-06-06 17:28:57 speaker_utils:93] Number of files to diarize: 1


    [NeMo W 2023-06-06 17:28:57 der:106] Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate


    [NeMo I 2023-06-06 17:28:57 msdd_models:1431]   
        


## Mapping Spekers to Sentences According to Timestamps


```python
# Reading timestamps <> Speaker Labels mapping

speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

## Realligning Speech segments using Punctuation
---

This code provides a method for disambiguating speaker labels in cases where a sentence is split between two different speakers. It uses punctuation markings to determine the dominant speaker for each sentence in the transcription.

```
Speaker A: It's got to come from somewhere else. Yeah, that one's also fun because you know the lows are
Speaker B: going to suck, right? So it's actually it hits you on both sides.
```

For example, if a sentence is split between two speakers, the code takes the mode of speaker labels for each word in the sentence, and uses that speaker label for the whole sentence. This can help to improve the accuracy of speaker diarization, especially in cases where the Whisper model may not take fine utterances like "hmm" and "yeah" into account, but the Diarization Model (Nemo) may include them, leading to inconsistent results.

The code also handles cases where one speaker is giving a monologue while other speakers are making occasional comments in the background. It ignores the comments and assigns the entire monologue to the speaker who is speaking the majority of the time. This provides a robust and reliable method for realigning speech segments to their respective speakers based on punctuation in the transcription.


```python
if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

    

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
else:
    print(
        f'Punctuation restoration is not available for {whisper_results["language"]} language.'
    )

ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/914 [00:00<?, ?B/s]



    Downloading pytorch_model.bin:   0%|          | 0.00/1.11G [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/447 [00:00<?, ?B/s]



    Downloading tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/239 [00:00<?, ?B/s]


    [NeMo W 2023-06-06 17:36:11 nemo_logging:349] /Users/calejohnstone/opt/anaconda3/lib/python3.9/site-packages/transformers/pipelines/token_classification.py:159: UserWarning: `grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to `aggregation_strategy="none"` instead.
          warnings.warn(
        


## Cleanup and Exporing the results


```python
with open(f"{audio_path[:-4]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{audio_path[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
```


```python

```
