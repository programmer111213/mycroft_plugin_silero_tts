import torch
from mycroft.tts import TTS, TTSValidator
from mycroft.util.log import LOG
from os.path import isfile

device = torch.device('cpu')
supported_languages = ["ru-ru"]
model_filename = "v3_1_ru.pt"
sample_rate = 48000


class SileroTTS(TTS):
    def __init__(self, lang, config):
        super().__init__(lang, config, SileroTTSValidator(self), audio_ext='wav',
                         phonetic_spelling=False, ssml_tags=['speak', 'break', 'prosody', 'p', 's'])
        self.speaker = config.get("speaker", "aidar")
        self.model = None

    def get_tts(self, sentence, wav_file):
        if self.model is None:
            LOG.info("Loading model")
            self._load_model()
        if self.speaker not in self.model.speakers:
            raise Exception("Invalid speaker name")
        if self.remove_ssml(sentence) != sentence:
            self.model.save_wav(ssml_text=sentence,
                                sample_rate=sample_rate,
                                speaker=self.speaker,
                                audio_path=wav_file)
        else:
            self.model.save_wav(text=sentence,
                                sample_rate=sample_rate,
                                speaker=self.speaker,
                                audio_path=wav_file)
        return wav_file, None

    def _load_model(self):
        self.model = torch.package.PackageImporter(model_filename).load_pickle(
            "tts_models", "model")
        self.model.to(device)


class SileroTTSValidator(TTSValidator):
    def __init__(self, tts):
        super(SileroTTSValidator, self).__init__(tts)

    def get_tts_class(self):
        return SileroTTS

    def validate_lang(self):
        if self.tts.lang not in supported_languages:
            raise Exception("Language not supported")

    def validate_connection(self):
        if not isfile(model_filename):
            LOG.info("Downloading model")
            url = f'https://models.silero.ai/models/tts/ru/{model_filename}'
            torch.hub.download_url_to_file(url, model_filename)
            LOG.info("Download finished")
