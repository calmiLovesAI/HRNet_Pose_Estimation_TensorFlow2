import tensorflow as tf
from core.work_flow import get_model, print_model_summary
from configuration.base_config import Config


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    hrnet = get_model(cfg_name=Config.TRAINING_CONFIG_NAME)
    print_model_summary(hrnet)