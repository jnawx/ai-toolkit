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


if __name__ == "__main__":
    unittest.main()
