import unittest

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig


class DOPStepIntervalConfigTests(unittest.TestCase):
    def test_interval_defaults_to_every_step(self):
        config = TrainConfig()

        self.assertEqual(config.diff_output_preservation_every_n_steps, 1)

    def test_interval_accepts_positive_integers(self):
        config = TrainConfig(diff_output_preservation_every_n_steps=4)

        self.assertEqual(config.diff_output_preservation_every_n_steps, 4)

    def test_interval_rejects_non_positive_or_fractional_values(self):
        for value in (0, -1, 1.5, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    TrainConfig(diff_output_preservation_every_n_steps=value)


class DOPStepIntervalScheduleTests(unittest.TestCase):
    @staticmethod
    def make_trainer(enabled=True, interval=3):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.train_config = TrainConfig(
            diff_output_preservation=enabled,
            diff_output_preservation_every_n_steps=interval,
        )
        return trainer

    def test_dop_runs_immediately_and_then_every_n_steps(self):
        trainer = self.make_trainer(interval=3)

        actual = []
        for step_num in range(7):
            trainer.step_num = step_num
            actual.append(trainer._should_run_diff_output_preservation())

        self.assertEqual(actual, [True, False, False, True, False, False, True])

    def test_disabled_dop_never_runs(self):
        trainer = self.make_trainer(enabled=False, interval=1)
        trainer.step_num = 0

        self.assertFalse(trainer._should_run_diff_output_preservation())


if __name__ == "__main__":
    unittest.main()
