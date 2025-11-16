import unittest

import services


class ServicesSmokeTests(unittest.TestCase):
    def test_model_configs_have_expected_keys(self):
        self.assertIsInstance(services.MODEL_CONFIGS, dict)
        self.assertIn('deepseek', services.MODEL_CONFIGS)
        conf = services.MODEL_CONFIGS['deepseek']
        self.assertIn('name', conf)
        self.assertIn('config', conf)

    def test_get_gpu_info_structure(self):
        info = services.get_gpu_info()
        self.assertIsInstance(info, dict)
        self.assertIn('cuda_available', info)
        if not info['cuda_available']:
            self.assertIn('message', info)

    def test_detect_model_device_none(self):
        info = services.detect_model_device(None)
        self.assertIsInstance(info, dict)
        self.assertEqual(info.get('status'), 'not_loaded')


if __name__ == '__main__':
    unittest.main()
