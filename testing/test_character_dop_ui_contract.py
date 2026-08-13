import re
import unittest
from pathlib import Path


class CharacterDOPUIContractTests(unittest.TestCase):
    def test_dataset_video_card_can_open_the_media_viewer(self):
        source = (
            Path(__file__).parents[1] / "ui" / "src" / "components" / "DatasetImageCard.tsx"
        ).read_text(encoding="utf-8")
        video_open_control = re.search(
            r"\{isItAVideo\s*&&\s*onImageClick\s*&&\s*\(.*?"
            r'title="Open video details and Character DOP".*?'
            r"onClick=\{onImageClick\}",
            source,
            re.DOTALL,
        )

        self.assertIsNotNone(
            video_open_control,
            "video cards need a dedicated viewer control so native playback controls remain usable",
        )

    def test_dataset_audio_card_can_open_the_character_dop_annotator(self):
        source = (
            Path(__file__).parents[1] / "ui" / "src" / "components" / "DatasetImageCard.tsx"
        ).read_text(encoding="utf-8")
        audio_open_control = re.search(
            r"\{isItAudio\s*&&\s*onImageClick\s*&&\s*\(.*?"
            r'title="Open audio details and Character DOP".*?'
            r"onClick=\{onImageClick\}",
            source,
            re.DOTALL,
        )

        self.assertIsNotNone(
            audio_open_control,
            "audio cards need a dedicated annotator control so playback remains usable",
        )

    def test_character_annotator_remembers_the_last_sam2_tracker_model(self):
        source = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "components"
            / "CharacterDOPAnnotator.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("CHARACTER_DOP_TRACKER_STORAGE_KEY", source)
        self.assertIn("localStorage.getItem(CHARACTER_DOP_TRACKER_STORAGE_KEY)", source)
        self.assertIn("catalog.trackers.some(model => model.id === storedTrackerModel)", source)
        self.assertIn("localStorage.setItem(CHARACTER_DOP_TRACKER_STORAGE_KEY", source)

    def test_character_dop_form_exposes_positive_visual_and_audio_training_weights(self):
        source = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "app"
            / "jobs"
            / "new"
            / "SimpleJob.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn('label="Character Visual Training Multiplier"', source)
        self.assertIn("train.character_training_visual_multiplier", source)
        self.assertIn('label="Character Audio Training Multiplier"', source)
        self.assertIn("train.character_training_audio_multiplier", source)

    def test_character_annotator_can_create_and_switch_named_identities(self):
        source = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "components"
            / "CharacterDOPAnnotator.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("Character identities", source)
        self.assertIn("request('update-identity'", source)
        self.assertIn("identityId: activeIdentityId", source)
        self.assertIn('aria-label="Character identity"', source)
        self.assertIn('placeholder="Trigger word"', source)
        self.assertIn('placeholder="Generic class prompt"', source)

    def test_character_annotator_can_edit_and_delete_the_selected_identity(self):
        source = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "components"
            / "CharacterDOPAnnotator.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("Update identity", source)
        self.assertIn("request('save-identity'", source)
        self.assertIn("Delete identity", source)
        self.assertIn("request('delete-identity'", source)
        self.assertIn("permanently deletes its masks", source)
        self.assertIn("nextState.cleanup_pending", source)
        self.assertIn("some staged files could not be removed", source)

    def test_character_annotator_persists_legacy_selection_and_ignores_stale_previews(self):
        source = (
            Path(__file__).parents[1]
            / "ui"
            / "src"
            / "components"
            / "CharacterDOPAnnotator.tsx"
        ).read_text(encoding="utf-8")

        self.assertIn("CHARACTER_DOP_LEGACY_IDENTITY", source)
        self.assertIn("storedIdentityId === CHARACTER_DOP_LEGACY_IDENTITY", source)
        self.assertIn("if (!cancelled) setPreview(data.data_url)", source)


if __name__ == "__main__":
    unittest.main()
