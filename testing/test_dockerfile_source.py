import unittest
from pathlib import Path


class DockerfileSourceTests(unittest.TestCase):
    def test_application_source_repository_is_configurable(self):
        dockerfile = (
            Path(__file__).parents[1] / "docker" / "Dockerfile"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "ARG GIT_REPOSITORY=https://github.com/ostris/ai-toolkit.git",
            dockerfile,
        )
        self.assertIn('git clone "${GIT_REPOSITORY}"', dockerfile)


if __name__ == "__main__":
    unittest.main()
